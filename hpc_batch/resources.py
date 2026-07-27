"""Machine resource discovery (CPUs/NUMA, GPUs, memory) and allocation.

A job's CPUs come from a single NUMA node so that memory accesses stay
local — this keeps benchmark timings stable. Memory follows the same rule:
`cpuset.mems` confines a job to the nodes it was charged for, so a budget is
only meaningful against a node, never against the machine total.

Placement therefore works like this:

- The budget fits the free memory on some node: the job goes there and is
  charged there. This is the ordinary case, and the job gets purely local
  memory without anyone having to ask for it.
- It does not fit any single node: the charge is spread across nodes, home
  node first, and `cpuset.mems` lists exactly those nodes. Access to the
  spilled part is slower, so this is reported rather than done silently.
- ``numa_local`` forbids the second case: the job waits for a node that can
  hold the whole budget locally.
- ``exclusive`` owns the machine, so it takes cores from every node and may
  use all of the memory.

Because every job is charged to exactly the nodes in its `cpuset.mems`, the
books and the kernel can never disagree about where a job's memory lives.
When we cannot set `cpuset.mems` at all (no cpuset controller, or
`--no-cgroups`) a job really can allocate anywhere, and the pool tracks one
machine-wide domain instead of pretending otherwise.
"""

import copy
import logging
import os
import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

_NODE_DIR = Path("/sys/devices/system/node")
_GPU_LINE = re.compile(r"^GPU (\d+):")
_MEMTOTAL_LINE = re.compile(r"^Node \d+ MemTotal:\s+(\d+) kB")

#: Memory-domain key used when jobs are not confined to one node's memory.
SHARED_MEM = -1

#: Slack for comparing GiB floats, well below anything a user can express.
_EPS = 1e-9


def format_id_list(ids: Iterable[int]) -> str:
    """Render cpu/node ids the way cpuset and CUDA_VISIBLE_DEVICES want them."""
    return ",".join(str(i) for i in ids)


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


def discover_node_memory_gb(nodes: dict[int, list[int]]) -> dict[int, float]:
    """Map NUMA node id -> that node's memory in GiB, from sysfs.

    Returns {} when any node's memory cannot be read: partial data is worse
    than none here, because a node we failed to read would look like it had
    no memory at all and no job could ever be placed on it. The caller
    treats {} as "fall back to a machine-wide pool".
    """
    mem: dict[int, float] = {}
    for node_id in nodes:
        try:
            text = (_NODE_DIR / f"node{node_id}" / "meminfo").read_text()
        except OSError:
            return {}
        for line in text.splitlines():
            match = _MEMTOTAL_LINE.match(line.strip())
            if match:
                mem[node_id] = int(match.group(1)) * 1024 / (1 << 30)
                break
        else:
            return {}
    return mem


def apply_reserve(
    node_cpus: dict[int, list[int]],
    node_mem_gb: dict[int, float],
    reserve_cpu: int,
    reserve_mem_gb: float,
) -> tuple[dict[int, list[int]], dict[int, float]]:
    """Hold back cores and memory for the OS and the daemon itself.

    Both come off the lowest-numbered node (spilling to the next only if one
    node cannot cover it). The OS is not pinned to them — this is headroom,
    not a fence — but its memory follows its cpus, so taking both from the
    same end is the honest guess, and it leaves the other nodes full width
    so a wide `--cpu` request can still be placed.

    That does leave the reserved node memory-rich relative to its cores.
    Whatever a node-local job cannot reach there is exactly what a job whose
    budget spans nodes will pick up.
    """
    cpus = {node: list(ids) for node, ids in node_cpus.items()}
    left = max(0, reserve_cpu)
    for node in sorted(cpus):
        if left <= 0:
            break
        # Never strand a node: one core has to remain allocatable.
        take = min(left, max(0, len(cpus[node]) - 1))
        if take:
            cpus[node] = cpus[node][take:]  # the low cpus, where the OS lives
            left -= take

    mem = dict(node_mem_gb)
    left_mem = max(0.0, reserve_mem_gb)
    for node in sorted(mem):
        if left_mem <= _EPS:
            break
        # Never halve a node; on a small machine a fixed reserve would
        # otherwise swallow most of it.
        take = min(left_mem, mem[node] / 2)
        mem[node] -= take
        left_mem -= take
    return cpus, mem


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


def charged_nodes(mem_by_node: dict[int, float]) -> list[int]:
    """The real NUMA nodes a memory charge names, lowest first.

    Filters the machine-wide domain out: SHARED_MEM is bookkeeping, never a
    node id. One definition because both `cpuset.mems` and "does this job
    span nodes?" depend on it, in different modules.
    """
    return sorted(k for k in mem_by_node if k != SHARED_MEM)


@dataclass(frozen=True)
class Request:
    """What a job is asking for. Kept separate from ``Job`` so the pool has
    no idea what a job is."""

    cpu: int
    gpu_cores: int = 0
    mem_gb: float | None = None
    exclusive: bool = False
    numa_local: bool = False


@dataclass
class Allocation:
    """Resources handed to one job.

    ``mem_by_node`` is the exact per-domain charge and its keys are what goes
    into `cpuset.mems`, so the job physically cannot allocate memory we did
    not account for.
    """

    cpus: list[int]
    numa_nodes: list[int]  # every node the cpus came from, lowest first
    gpus: list[int]
    mem_gb: float | None
    mem_by_node: dict[int, float]
    exclusive: bool

    @property
    def numa_node(self) -> int:
        """Home node: where the cpus are, or the first of them when spanning."""
        return self.numa_nodes[0]

    @property
    def spans_nodes(self) -> bool:
        """True when the budget did not fit one node and had to be spread."""
        return len(charged_nodes(self.mem_by_node)) > 1

    def mem_nodes(self) -> list[int]:
        """Nodes for `cpuset.mems`. Falls back to the cpu nodes when memory
        is untracked or pooled machine-wide, where confinement is moot."""
        return charged_nodes(self.mem_by_node) or sorted(self.numa_nodes)


@dataclass
class ResourcePool:
    """Tracks free CPUs and memory (both per NUMA node) and free GPUs.

    ``node_mem_gb`` is each node's memory available to jobs. With
    ``mem_confined`` (the normal case, `cpuset.mems` is set) every node is
    its own memory domain; without it the nodes share one domain holding
    their combined memory.
    """

    node_cpus: dict[int, list[int]]
    gpu_ids: list[int]
    node_mem_gb: dict[int, float]
    mem_confined: bool = True
    free_cpus: dict[int, set[int]] = field(init=False)
    free_gpus: set[int] = field(init=False)
    mem_capacity: dict[int, float] = field(init=False)
    free_mem_gb: dict[int, float] = field(init=False)
    # False when memory is unknown (discovery failed); requests then pass the
    # memory checks unexamined. Fixed once: only free_mem_gb ever moves.
    tracks_memory: bool = field(init=False, default=False)
    active: int = field(init=False, default=0)
    exclusive_active: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.free_cpus = {node: set(cpus) for node, cpus in self.node_cpus.items()}
        self.free_gpus = set(self.gpu_ids)
        if self.mem_confined and set(self.node_mem_gb) != set(self.node_cpus):
            # A node with no memory figure could never host a job with a
            # budget; degrade to machine-wide rather than silently stranding it.
            log.warning(
                "per-node memory %s does not cover every NUMA node %s; "
                "tracking memory machine-wide instead",
                sorted(self.node_mem_gb), sorted(self.node_cpus),
            )
            self.mem_confined = False
        if self.mem_confined:
            self.mem_capacity = {n: float(m) for n, m in self.node_mem_gb.items()}
        else:
            self.mem_capacity = {SHARED_MEM: float(sum(self.node_mem_gb.values()))}
        self.free_mem_gb = dict(self.mem_capacity)
        self.tracks_memory = any(cap > 0 for cap in self.mem_capacity.values())

    # -- memory domains -------------------------------------------------

    def mem_key(self, node: int) -> int:
        """The memory domain a job on ``node`` draws from."""
        return node if self.mem_confined else SHARED_MEM

    def _free_mem(self, node: int) -> float:
        return self.free_mem_gb.get(self.mem_key(node), 0.0)

    def largest_node_mem_gb(self) -> float:
        """Most memory one node can give a job."""
        return max(self.mem_capacity.values(), default=0.0)

    def usable_mem_gb(self) -> float:
        """Total memory available to jobs, across all nodes."""
        return sum(self.mem_capacity.values())

    def total_cpus(self) -> int:
        return sum(len(cpus) for cpus in self.node_cpus.values())

    def proportional_share(self, cpu: int) -> float:
        """The memory ``cpu`` cores are worth, on the least generous node.

        Resolved against the stingiest node because a defaulted budget has to
        be a concrete number at submit time — the node a job lands on is not
        chosen until it starts — and this way it can always be placed. Comes
        out at 0 when memory is untracked, since every capacity is then 0.
        """
        if not self.mem_confined:
            total = self.total_cpus()
            cap = self.mem_capacity[SHARED_MEM]
            return cap * min(cpu, total) / total if total else 0.0
        shares = [
            self.mem_capacity.get(node, 0.0) * min(cpu, len(cpus)) / len(cpus)
            for node, cpus in self.node_cpus.items()
            if cpus
        ]
        return min(shares, default=0.0)

    # -- validation -----------------------------------------------------

    def validate(self, req: Request) -> str | None:
        """Return an error string if the request can never be satisfied.

        Only impossibilities belong here. Anything merely unavailable right
        now should queue instead, so a request that fits the biggest node is
        accepted even when no node can host it at this moment.
        """
        if self._max_nodes(req) > 1:
            total = self.total_cpus()
            if req.cpu > total:
                return f"--cpu {req.cpu} exceeds the {total} cores available to jobs"
        else:
            biggest = max(len(cpus) for cpus in self.node_cpus.values())
            if req.cpu > biggest:
                return (
                    f"--cpu {req.cpu} exceeds the largest NUMA node ({biggest} cpus); "
                    "jobs are always confined to a single node"
                )
        if req.gpu_cores > len(self.gpu_ids):
            return f"--gpu-cores {req.gpu_cores} exceeds the {len(self.gpu_ids)} gpus on this machine"
        if req.mem_gb is not None and self.tracks_memory:
            if req.numa_local:
                ceiling = self.largest_node_mem_gb()
                if req.mem_gb > ceiling + _EPS:
                    return (
                        f"--max-mem {req.mem_gb:g} exceeds the largest NUMA node "
                        f"({ceiling:.0f} GiB) and --numa-local keeps a job on one "
                        "node; drop --numa-local to let the budget span nodes, or "
                        "use --exclusive"
                    )
            else:
                ceiling = self.usable_mem_gb()
                if req.mem_gb > ceiling + _EPS:
                    return (
                        f"--max-mem {req.mem_gb:g} exceeds the {ceiling:.0f} GiB "
                        "available to jobs"
                    )
        return None

    # -- allocation -----------------------------------------------------

    def _fits_locally(self, node: int, mem_gb: float | None) -> bool:
        if mem_gb is None or not self.tracks_memory:
            return True
        return mem_gb <= self._free_mem(node) + _EPS

    def _max_nodes(self, req: Request) -> int:
        """How many NUMA nodes this request may draw cpus from.

        One for an ordinary job — the invariant the whole design rests on. An
        exclusive job owns the machine and may span every node, unless it
        asked to stay local. `validate` and `_plan` both read the ceiling from
        here: if they disagreed, a job could pass submission and then never be
        placed.
        """
        return len(self.node_cpus) if req.exclusive and not req.numa_local else 1

    def _cpu_node_order(self, req: Request, only: int | None, max_nodes: int) -> list[int]:
        """The nodes to draw cpus from, best first. Empty means no placement."""
        candidates = [n for n in self.free_cpus if only is None or n == only]
        if max_nodes > 1:
            # Spanning the machine: there is no one else to keep a node free
            # for, so plain node order is as good as any.
            return sorted(candidates)
        # One node has to hold the whole cpu request.
        whole = [n for n in candidates if len(self.free_cpus[n]) >= req.cpu]
        local = [n for n in whole if self._fits_locally(n, req.mem_gb)]
        if local:
            # Best fit: fewest free cpus, then least free memory, keeping
            # roomier nodes available for bigger jobs. Node id breaks ties so
            # placement stays deterministic.
            return sorted(local, key=lambda n: (len(self.free_cpus[n]), self._free_mem(n), n))
        if req.numa_local:
            return []
        # The budget has to span nodes. Here memory leads and cpu best-fit
        # only breaks ties: this job is already going to pay for remote
        # access, so the node with the most free memory keeps that bill as
        # small as possible. Whether it fits at all is _charge_plan's call.
        return sorted(whole, key=lambda n: (-self._free_mem(n), len(self.free_cpus[n]), n))

    def _pick_cpus(
        self, req: Request, only: int | None, max_nodes: int
    ) -> tuple[list[int], list[int]] | None:
        """Choose this request's cpus: which nodes, and which cores on them.

        Nodes are ranked by policy and drawn from in order until the request
        is satisfied, using at most ``max_nodes`` of them. At ``max_nodes==1``
        this is the ordinary "one node holds the whole job" case; a larger
        ceiling is what lets an exclusive job take the machine. One mechanism
        either way, so the ``only`` pin and the memory-aware node ranking
        apply uniformly.
        """
        cpus: list[int] = []
        nodes: list[int] = []
        for node in self._cpu_node_order(req, only, max_nodes):
            if len(cpus) >= req.cpu or len(nodes) >= max_nodes:
                break
            take = sorted(self.free_cpus[node])[: req.cpu - len(cpus)]
            if take:
                cpus.extend(take)
                nodes.append(node)
        return (cpus, nodes) if len(cpus) == req.cpu else None

    def _charge_plan(self, req: Request, home: int) -> dict[int, float] | None:
        """How much of the budget to charge to each memory domain, home node
        first. None when it cannot be covered.

        Each domain is charged everything it has free before the next is
        touched. That matters: a job whose charge on a node equals that
        node's whole free capacity has no room to quietly grow past it, which
        is what keeps the books and the kernel in agreement.
        """
        if req.mem_gb is None or not self.tracks_memory:
            return {}
        home_key = self.mem_key(home)
        order = [home_key]
        if not req.numa_local:
            order += [k for k in sorted(self.free_mem_gb) if k != home_key]
        charge: dict[int, float] = {}
        remaining = req.mem_gb
        for key in order:
            if remaining <= _EPS:
                break
            take = min(remaining, self.free_mem_gb[key])
            if take > 0:
                charge[key] = take
                remaining -= take
        return None if remaining > _EPS else charge

    def _plan(self, req: Request, node: int | None = None) -> Allocation | None:
        """Build the allocation this request would get, without taking it."""
        if self.exclusive_active:
            return None
        if req.exclusive and self.active > 0:
            return None
        if req.gpu_cores > len(self.free_gpus):
            return None
        picked = self._pick_cpus(req, node, self._max_nodes(req))
        if picked is None:
            return None
        cpus, cpu_nodes = picked
        charge = self._charge_plan(req, cpu_nodes[0])
        if charge is None:
            return None
        return Allocation(
            cpus=cpus,
            numa_nodes=cpu_nodes,
            gpus=sorted(self.free_gpus)[: req.gpu_cores],
            mem_gb=req.mem_gb,
            mem_by_node=charge,
            exclusive=req.exclusive,
        )

    def would_fit(self, req: Request) -> bool:
        """True if this request could be allocated right now (non-mutating)."""
        return self._plan(req) is not None

    def allocate(self, req: Request, node: int | None = None) -> Allocation | None:
        """Try to allocate; None means the job must keep waiting. ``node``
        pins the choice to one NUMA node instead of letting the pool pick."""
        alloc = self._plan(req, node)
        if alloc is not None:
            self.reserve(alloc)
        return alloc

    # -- bookkeeping ----------------------------------------------------

    def free_cpus_by_node(self) -> dict[int, int]:
        """Free cpu count per NUMA node — the unit a job is actually placed
        in. Summing this across nodes would overstate what one job can get."""
        return {node: len(free) for node, free in self.free_cpus.items()}

    def clone(self) -> "ResourcePool":
        """A copy with independent free-lists, for what-if simulation."""
        return copy.deepcopy(self)

    def _charge_domains(self, alloc: Allocation) -> dict[int, float]:
        """The allocation's charge, re-keyed onto the domains in force now.

        A charge read back from the state file was keyed by whatever layout
        was in force when the job started, and a reload can change that
        layout (cpuset newly delegated, `--no-cgroups` toggled). Anything
        unrecognised is folded onto the home node's domain: over-charging one
        domain is safe, while dropping the charge would let it be handed out
        twice.
        """
        charge: dict[int, float] = {}
        for key, gb in alloc.mem_by_node.items():
            domain = key if key in self.free_mem_gb else self.mem_key(alloc.numa_node)
            charge[domain] = charge.get(domain, 0.0) + gb
        return {k: v for k, v in charge.items() if k in self.free_mem_gb}

    def reserve(self, alloc: Allocation) -> None:
        """Mark resources as used (also used to re-adopt jobs after a reload)."""
        for node in alloc.numa_nodes:
            free = self.free_cpus.get(node)
            if free is not None:
                free.difference_update(alloc.cpus)
        self.free_gpus.difference_update(alloc.gpus)
        for key, gb in self._charge_domains(alloc).items():
            self.free_mem_gb[key] = max(0.0, self.free_mem_gb[key] - gb)
        self.active += 1
        if alloc.exclusive:
            self.exclusive_active = True

    def release(self, alloc: Allocation) -> None:
        for node in alloc.numa_nodes:
            free = self.free_cpus.get(node)
            if free is not None:
                owned = set(self.node_cpus.get(node, []))
                free.update(c for c in alloc.cpus if c in owned)
        self.free_gpus.update(g for g in alloc.gpus if g in self.gpu_ids)
        for key, gb in self._charge_domains(alloc).items():
            self.free_mem_gb[key] = min(self.mem_capacity[key], self.free_mem_gb[key] + gb)
        self.active = max(0, self.active - 1)
        if alloc.exclusive:
            self.exclusive_active = False
