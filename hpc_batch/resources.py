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

GPUs follow the same principle. A multi-GPU job's devices talk to each
other, and taking the lowest free indices hands it a pair on opposite sides
of the machine whenever the index between them is busy. So the pool hands
out the closest-connected free set that `nvidia-smi topo -m` describes, and
puts the job's cpus on the NUMA node those GPUs hang off. Where that file
tells us nothing, GPU choice is index order, as it was before.

Closest is found by walking the machine's own structure rather than by
scoring every combination of GPUs. Each level of the interconnect cuts the
free GPUs into islands — the GPUs behind one switch, one host bridge, one
socket — and a job is placed in the finest island that can hold it, because
a better set would have to be a whole island one level down, where there was
none big enough. Which island, when several would do, is a best-fit decision
exactly like the one CPU nodes get: take from the island with least left to
give, so that whole islands stay whole for the jobs that need one.
"""

import copy
import itertools
import logging
import math
import os
import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger(__name__)

_NODE_DIR = Path("/sys/devices/system/node")
_GPU_LINE = re.compile(r"^GPU (\d+):")
_GPU_NAME = re.compile(r"^GPU(\d+)$")
_NVLINK = re.compile(r"^NV(\d+)$")
_MEMTOTAL_LINE = re.compile(r"^Node \d+ MemTotal:\s+(\d+) kB")

#: Link classes from the legend of `nvidia-smi topo -m`, closest first.
_LINK_CLASSES = ("NV", "PIX", "PXB", "PHB", "NODE", "SYS")

#: Links we will score to choose between the GPU sets one island offers. Past
#: this the island is taken apart into the finer ones inside it instead, which
#: is both cheaper and what the answer looks like anyway at that width. No
#: island of a real machine comes close: it is reached only by a topology with
#: no structure to take apart.
_MAX_GPU_LINK_SCANS = 20_000

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


def _nvidia_smi(*args: str) -> str:
    """Output of an `nvidia-smi` query, or "" if it cannot be asked."""
    try:
        out = subprocess.run(
            ["nvidia-smi", *args], capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout if out.returncode == 0 else ""


def discover_gpus() -> list[int]:
    """GPU indices reported by `nvidia-smi -L`; empty when unavailable."""
    gpus = []
    for line in _nvidia_smi("-L").splitlines():
        match = _GPU_LINE.match(line.strip())
        if match:
            gpus.append(int(match.group(1)))
    return gpus


@lru_cache(maxsize=None)
def _link_rank(label: str | None) -> int:
    """How far apart a link class puts two GPUs, closest = 0.

    Anything we do not recognise — a missing entry, a class from a newer
    driver — sorts last, so a pair we know nothing about is never preferred
    to one we do.
    """
    if label and _NVLINK.match(label):
        label = "NV"
    return _LINK_CLASSES.index(label) if label in _LINK_CLASSES else len(_LINK_CLASSES)


@lru_cache(maxsize=None)
def _nvlink_width(label: str | None) -> int:
    """Number of bonded NVLinks a label describes; 0 if it is not NVLink."""
    match = _NVLINK.match(label or "")
    return int(match.group(1)) if match else 0


class LinkQuality(NamedTuple):
    """A set of GPUs scored as a group, in the order the fields are compared."""

    worst: int  # rank of the link pacing the set
    total: int  # every pair's rank, summed
    width: int  # NVLink lanes, negated so that more of them sorts first


@dataclass(frozen=True)
class GpuTopology:
    """The link class between each pair of GPUs, and the NUMA node each one
    hangs off.

    Empty when `nvidia-smi topo -m` is unavailable or unreadable; every
    method then answers "no idea".
    """

    links: dict[tuple[int, int], str] = field(default_factory=dict)
    numa_node: dict[int, int] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.links or self.numa_node)

    def __deepcopy__(self, memo) -> "GpuTopology":
        # Shared, not copied: the machine's wiring cannot change under us,
        # and ResourcePool.clone() deep-copies the pool on every tick.
        return self

    def link(self, a: int, b: int) -> str | None:
        return self.links.get((min(a, b), max(a, b)))

    def _pairs(self, gpus: Iterable[int]) -> list[str | None]:
        return [self.link(a, b) for a, b in itertools.combinations(sorted(gpus), 2)]

    def quality(self, gpus: Iterable[int]) -> "LinkQuality":
        """How good a set of GPUs is as a group, best first.

        The worst link leads because a collective runs at the speed of its
        slowest pair: four NVLinked GPUs and one across the machine is a
        five-GPU job bound by that one hop.
        """
        pairs = self._pairs(gpus)
        ranks = [_link_rank(link) for link in pairs]
        return LinkQuality(
            worst=max(ranks, default=0),
            total=sum(ranks),
            width=-sum(_nvlink_width(link) for link in pairs),
        )

    def islands(self, gpus: Iterable[int], level: int) -> list[list[int]]:
        """``gpus`` split into groups joined by links no worse than ``level``.

        For the PCIe classes this is the machine's own structure: a GPU sits
        under a switch, under a host bridge, under a socket, so "no worse than
        L" is transitive and the groups are exactly those units. NVLink is a
        mesh and not a hierarchy — on a DGX-1 all eight GPUs are one NVLinked
        group but only its quads are wired pairwise — so a group is somewhere
        the answer may be, never proof that the answer is the whole group.
        """
        rest = sorted(gpus)
        groups: list[list[int]] = []
        while rest:
            group = [rest.pop(0)]
            frontier = list(group)
            while frontier:
                gpu = frontier.pop()
                joined = [g for g in rest if _link_rank(self.link(gpu, g)) <= level]
                for g in joined:
                    rest.remove(g)
                group += joined
                frontier += joined
            groups.append(sorted(group))
        return groups

    def worst_link(self, gpus: Iterable[int]) -> str | None:
        """The link class pacing this set. None when it has no pair, or none
        we know anything about."""
        return max((p for p in self._pairs(gpus) if p), key=_link_rank, default=None)

    def nodes_for(self, gpus: Iterable[int]) -> set[int]:
        """The NUMA nodes these GPUs hang off; empty when unknown."""
        return {self.numa_node[g] for g in gpus if g in self.numa_node}


def parse_gpu_topology(text: str) -> GpuTopology:
    """Read the matrix printed by `nvidia-smi topo -m`.

    Only the header says how wide the matrix is, and it has to: on a machine
    with InfiniBand the NICs get columns too, and without the width a row's
    trailing affinity fields are indistinguishable from its links. The header
    is the indented line; rows start flush left with the device name.
    """
    lines = text.splitlines()
    header = next(
        (line for line in lines if line[:1].isspace() and line.strip().startswith("GPU")),
        None,
    )
    if header is None:
        return GpuTopology()
    columns = header.split()
    if "CPU" in columns:
        columns = columns[: columns.index("CPU")]  # "CPU Affinity" and what follows

    links: dict[tuple[int, int], str] = {}
    numa: dict[int, int] = {}
    for line in lines:
        if line[:1].isspace():
            continue
        cells = line.split()
        match = _GPU_NAME.match(cells[0]) if cells else None
        if match is None:
            continue
        row = int(match.group(1))
        for name, label in zip(columns, cells[1 : 1 + len(columns)]):
            col = _GPU_NAME.match(name)
            if col and int(col.group(1)) != row:
                pair = (min(row, int(col.group(1))), max(row, int(col.group(1))))
                links[pair] = label
        affinity = cells[1 + len(columns):]  # CPU affinity, then NUMA affinity
        if len(affinity) > 1 and affinity[1].isdigit():
            numa[row] = int(affinity[1])
    return GpuTopology(links=links, numa_node=numa)


def discover_gpu_topology() -> GpuTopology:
    """GPU interconnect per `nvidia-smi topo -m`; empty when unavailable."""
    return parse_gpu_topology(_nvidia_smi("topo", "-m"))


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
    gpu_topology: GpuTopology = field(default_factory=GpuTopology)
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

    def _cpu_node_order(
        self, req: Request, only: int | None, max_nodes: int, near: set[int]
    ) -> list[int]:
        """The nodes to draw cpus from, best first. Empty means no placement.

        ``near`` are the nodes this job's devices already sit on, if any.
        """
        candidates = [n for n in self.free_cpus if only is None or n == only]
        if max_nodes > 1:
            # Spanning the machine: there is no one else to keep a node free
            # for, so plain node order is as good as any.
            return sorted(candidates)
        # One node has to hold the whole cpu request.
        whole = [n for n in candidates if len(self.free_cpus[n]) >= req.cpu]
        local = [n for n in whole if self._fits_locally(n, req.mem_gb)]
        if local:
            # A node the job's own devices sit on leads: everything it feeds
            # them crosses the interconnect otherwise. Then best fit — fewest
            # free cpus, then least free memory — which keeps roomier nodes
            # available for bigger jobs. Node id breaks ties so placement stays
            # deterministic.
            return sorted(
                local,
                key=lambda n: (n not in near, len(self.free_cpus[n]), self._free_mem(n), n),
            )
        if req.numa_local:
            return []
        # The budget has to span nodes. Here memory leads, ahead of even
        # ``near``: this job is already going to pay for remote access, so the
        # node with the most free memory keeps that bill as small as possible,
        # and cpu best-fit only breaks ties. Whether it fits at all is
        # _charge_plan's call.
        return sorted(whole, key=lambda n: (-self._free_mem(n), len(self.free_cpus[n]), n))

    def _pick_cpus(
        self, req: Request, only: int | None, max_nodes: int, near: set[int]
    ) -> tuple[list[int], list[int]] | None:
        """Choose this request's cpus: which nodes, and which cores on them.

        Nodes are ranked by policy and drawn from in order until the request
        is satisfied, using at most ``max_nodes`` of them. At ``max_nodes==1``
        this is the ordinary "one node holds the whole job" case; a larger
        ceiling is what lets an exclusive job take the machine. One mechanism
        either way, so the ``only`` pin and the node ranking apply uniformly.
        """
        cpus: list[int] = []
        nodes: list[int] = []
        for node in self._cpu_node_order(req, only, max_nodes, near):
            if len(cpus) >= req.cpu or len(nodes) >= max_nodes:
                break
            take = sorted(self.free_cpus[node])[: req.cpu - len(cpus)]
            if take:
                cpus.extend(take)
                nodes.append(node)
        return (cpus, nodes) if len(cpus) == req.cpu else None

    def _pick_gpus(self, count: int, *, search: bool = True) -> list[int]:
        """``count`` free GPUs: the closest-connected set, or the lowest free
        indices when there is no topology to consult, when the free GPUs are
        exactly used up either way, or when ``search`` is off.

        The set is found by climbing the machine's own structure rather than
        by scoring every combination: at each level of the interconnect, only
        the islands wide enough to hold the job are looked inside, and the
        first level that really delivers a set that good is the best the job
        can get — anything better would have been a whole island one level
        down, where there was no island big enough.
        """
        free = sorted(self.free_gpus)
        if not search or count <= 0 or count >= len(free) or not self.gpu_topology:
            return free[:count]
        topo = self.gpu_topology
        for level in range(len(_LINK_CLASSES) + 1):
            roomy = [i for i in topo.islands(free, level) if len(i) >= count]
            if not roomy:
                continue
            picks = [self._within_island(island, count, level) for island in roomy]
            # Between islands that would serve the job equally well, take from
            # the one with least left to give, so whole islands stay whole for
            # the jobs that need one. Same best-fit as cpu nodes get, and the
            # ids only settle what that leaves tied.
            quality, _, gpus = min(
                (topo.quality(pick), len(island), pick)
                for island, pick in zip(roomy, picks)
            )
            # Reaching this level joined an island's GPUs; it did not make them
            # pairwise close, because NVLink is a mesh and not a hierarchy. So
            # take the set only if it really is this good, and otherwise let the
            # wider islands one level up be searched.
            if quality.worst <= level:
                return gpus
        # Unreachable: the last level joins every pair, known or not, so one of
        # them always matched. Here to keep the function total.
        return free[:count]

    def _within_island(self, island: list[int], count: int, level: int) -> list[int]:
        """``count`` GPUs from one island, best set first."""
        if count >= len(island):
            return list(island)
        if math.comb(len(island), count) * math.comb(count, 2) <= _MAX_GPU_LINK_SCANS:
            # Islands are small, and this is the only exact answer where the
            # links inside one are a mesh rather than a hierarchy.
            return list(min(itertools.combinations(island, count), key=self.gpu_topology.quality))
        # Too wide to enumerate, which means it is a coarse level holding
        # several finer islands (a finer one alone would have been chosen
        # instead). Every pair that crosses between them costs this level, and
        # taking whole islands, biggest first, is what leaves fewest such
        # pairs.
        chosen: list[int] = []
        finer = sorted(self.gpu_topology.islands(island, level - 1), key=lambda i: (-len(i), i))
        for sub in finer:
            if len(chosen) >= count:
                break
            chosen += self._within_island(sub, min(len(sub), count - len(chosen)), level - 1)
        return sorted(chosen)

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

    def _plan(
        self, req: Request, node: int | None = None, *, place_gpus: bool = True
    ) -> Allocation | None:
        """Build the allocation this request would get, without taking it.

        Without ``place_gpus`` the gpus are taken in index order instead of
        searched for: *which* gpus a job gets never decides whether it fits,
        so a caller only asking that question need not pay for the search.
        """
        if self.exclusive_active:
            return None
        if req.exclusive and self.active > 0:
            return None
        if req.gpu_cores > len(self.free_gpus):
            return None
        # GPUs first: they are the fixed points here — a job cannot be moved
        # to the other end of the machine, but its cpus can be placed near it.
        gpus = self._pick_gpus(req.gpu_cores, search=place_gpus)
        near = self.gpu_topology.nodes_for(gpus)
        picked = self._pick_cpus(req, node, self._max_nodes(req), near)
        if picked is None:
            return None
        cpus, cpu_nodes = picked
        charge = self._charge_plan(req, cpu_nodes[0])
        if charge is None:
            return None
        return Allocation(
            cpus=cpus,
            numa_nodes=cpu_nodes,
            gpus=gpus,
            mem_gb=req.mem_gb,
            mem_by_node=charge,
            exclusive=req.exclusive,
        )

    def would_fit(self, req: Request) -> bool:
        """True if this request could be allocated right now (non-mutating)."""
        return self._plan(req, place_gpus=False) is not None

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
