"""Machine resource discovery (CPUs/NUMA, GPUs, memory) and allocation.

All CPUs of a job are always allocated from a single NUMA node so that
memory accesses stay local — this keeps benchmark timings stable.
"""

import copy
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

_NODE_DIR = Path("/sys/devices/system/node")
_GPU_LINE = re.compile(r"^GPU (\d+):")


def parse_cpu_list(text: str) -> list[int]:
    """Parse a sysfs cpulist like "0-3,8-11" into [0,1,2,3,8,9,10,11]."""
    cpus: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.extend(range(int(lo), int(hi) + 1))
        else:
            cpus.append(int(part))
    return cpus


def discover_numa_nodes() -> dict[int, list[int]]:
    """Map NUMA node id -> cpu ids. Falls back to one node with every cpu."""
    nodes: dict[int, list[int]] = {}
    try:
        for entry in sorted(_NODE_DIR.glob("node[0-9]*")):
            node_id = int(entry.name[len("node"):])
            cpus = parse_cpu_list((entry / "cpulist").read_text())
            if cpus:
                nodes[node_id] = cpus
    except OSError:
        nodes = {}
    if not nodes:
        nodes = {0: list(range(os.cpu_count() or 1))}
    return nodes


def discover_gpus() -> list[int]:
    """GPU indices reported by `nvidia-smi -L`; empty when unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if out.returncode != 0:
        return []
    gpus = []
    for line in out.stdout.splitlines():
        match = _GPU_LINE.match(line.strip())
        if match:
            gpus.append(int(match.group(1)))
    return gpus


def total_memory_gb() -> float:
    """Total physical memory in GiB."""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1 << 30)
    except (OSError, ValueError):
        return 0.0


@dataclass
class Allocation:
    cpus: list[int]
    numa_node: int
    gpus: list[int]
    mem_gb: float | None
    exclusive: bool


@dataclass
class ResourcePool:
    """Tracks free CPUs (per NUMA node), GPUs and memory."""

    node_cpus: dict[int, list[int]]
    gpu_ids: list[int]
    total_mem_gb: float
    free_cpus: dict[int, set[int]] = field(init=False)
    free_gpus: set[int] = field(init=False)
    free_mem_gb: float = field(init=False)
    active: int = field(init=False, default=0)
    exclusive_active: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.free_cpus = {node: set(cpus) for node, cpus in self.node_cpus.items()}
        self.free_gpus = set(self.gpu_ids)
        self.free_mem_gb = self.total_mem_gb

    # -- validation -----------------------------------------------------

    def validate(self, cpu: int, gpu_cores: int, mem_gb: float | None) -> str | None:
        """Return an error string if the request can never be satisfied."""
        biggest_node = max(len(cpus) for cpus in self.node_cpus.values())
        if cpu > biggest_node:
            return (
                f"--cpu {cpu} exceeds the largest NUMA node ({biggest_node} cpus); "
                "jobs are always confined to a single node"
            )
        if gpu_cores > len(self.gpu_ids):
            return f"--gpu-cores {gpu_cores} exceeds the {len(self.gpu_ids)} gpus on this machine"
        if mem_gb is not None and self.total_mem_gb and mem_gb > self.total_mem_gb:
            return f"--max-mem {mem_gb:g} exceeds total memory ({self.total_mem_gb:.0f} GiB)"
        return None

    # -- allocation -----------------------------------------------------

    def would_fit(
        self, cpu: int, gpu_cores: int, mem_gb: float | None, exclusive: bool
    ) -> bool:
        """True if this request could be allocated right now (non-mutating)."""
        if self.exclusive_active:
            return False
        if exclusive and self.active > 0:
            return False
        if gpu_cores > len(self.free_gpus):
            return False
        if mem_gb is not None and self.total_mem_gb and mem_gb > self.free_mem_gb:
            return False
        return any(len(free) >= cpu for free in self.free_cpus.values())

    def free_totals(self) -> tuple[int, int, float | None]:
        """Currently-free resources: (cpu count across all nodes, gpu count,
        memory in GiB or None when memory is not tracked)."""
        cpu = sum(len(free) for free in self.free_cpus.values())
        mem = self.free_mem_gb if self.total_mem_gb else None
        return cpu, len(self.free_gpus), mem

    def clone(self) -> "ResourcePool":
        """A copy with independent free-lists, for what-if simulation."""
        return copy.deepcopy(self)

    def allocate(
        self, cpu: int, gpu_cores: int, mem_gb: float | None, exclusive: bool
    ) -> Allocation | None:
        """Try to allocate; None means the job must keep waiting."""
        if not self.would_fit(cpu, gpu_cores, mem_gb, exclusive):
            return None
        # Best-fit: the node with the fewest free cpus that still fits,
        # keeping bigger contiguous nodes available for bigger jobs.
        candidates = [
            (len(free), node)
            for node, free in self.free_cpus.items()
            if len(free) >= cpu
        ]
        _, node = min(candidates)
        cpus = sorted(self.free_cpus[node])[:cpu]
        gpus = sorted(self.free_gpus)[:gpu_cores]
        alloc = Allocation(cpus=cpus, numa_node=node, gpus=gpus, mem_gb=mem_gb, exclusive=exclusive)
        self.reserve(alloc)
        return alloc

    def reserve(self, alloc: Allocation) -> None:
        """Mark resources as used (also used to re-adopt jobs after a reload)."""
        free = self.free_cpus.get(alloc.numa_node)
        if free is not None:
            free.difference_update(alloc.cpus)
        self.free_gpus.difference_update(alloc.gpus)
        if alloc.mem_gb is not None:
            self.free_mem_gb = max(0.0, self.free_mem_gb - alloc.mem_gb)
        self.active += 1
        if alloc.exclusive:
            self.exclusive_active = True

    def release(self, alloc: Allocation) -> None:
        free = self.free_cpus.get(alloc.numa_node)
        if free is not None:
            owned = set(self.node_cpus.get(alloc.numa_node, []))
            free.update(c for c in alloc.cpus if c in owned)
        self.free_gpus.update(g for g in alloc.gpus if g in self.gpu_ids)
        if alloc.mem_gb is not None:
            self.free_mem_gb = min(self.total_mem_gb, self.free_mem_gb + alloc.mem_gb)
        self.active = max(0, self.active - 1)
        if alloc.exclusive:
            self.exclusive_active = False
