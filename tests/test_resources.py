from hpc_batch import resources
from hpc_batch.resources import (
    SHARED_MEM,
    Allocation,
    GpuTopology,
    Request,
    ResourcePool,
    apply_reserve,
    discover_node_memory_gb,
    discover_numa_nodes,
    parse_cpu_list,
    parse_gpu_topology,
)

#: `nvidia-smi topo -m` on a machine with two NVLinked GPU pairs, one pair per
#: NUMA node and a SYS hop between them, plus an InfiniBand NIC (which takes a
#: matrix column of its own without being a GPU).
TOPO_OUTPUT = "\n".join(
    "\t".join(row)
    for row in [
        ["", "GPU0", "GPU1", "GPU2", "GPU3", "mlx5_0",
         "CPU Affinity", "NUMA Affinity", "GPU NUMA ID"],
        ["GPU0", " X ", "NV2", "SYS", "SYS", "PIX", "0-3", "0", "N/A"],
        ["GPU1", "NV2", " X ", "SYS", "SYS", "PIX", "0-3", "0", "N/A"],
        ["GPU2", "SYS", "SYS", " X ", "NV2", "SYS", "4-7", "1", "N/A"],
        ["GPU3", "SYS", "SYS", "NV2", " X ", "SYS", "4-7", "1", "N/A"],
        ["mlx5_0", "PIX", "PIX", "SYS", "SYS", " X ", "", "", ""],
    ]
) + "\n\nLegend:\n\n  X    = Self\n  SYS  = Connection traversing PCIe\n"


def make_pool() -> ResourcePool:
    """Two nodes, 4 cpus and 32 GiB each: 64 GiB machine-wide, but no single
    node can give one job more than 32."""
    return ResourcePool(
        node_cpus={0: [0, 1, 2, 3], 1: [4, 5, 6, 7]},
        gpu_ids=[0, 1, 2, 3],
        node_mem_gb={0: 32.0, 1: 32.0},
    )


def topo_pool() -> ResourcePool:
    """`make_pool` on a machine whose GPU wiring we know: see TOPO_OUTPUT."""
    pool = make_pool()
    pool.gpu_topology = parse_gpu_topology(TOPO_OUTPUT)
    return pool


def two_quads() -> GpuTopology:
    """Eight GPUs as two NVLinked quads, one per NUMA node, a SYS hop apart."""
    return GpuTopology(
        links={
            (a, b): ("NV2" if a // 4 == b // 4 else "SYS")
            for a in range(8) for b in range(a + 1, 8)
        },
        numa_node={gpu: gpu // 4 for gpu in range(8)},
    )


def quad_pool() -> ResourcePool:
    """A `two_quads` machine, 8 cpus and 64 GiB behind each quad."""
    return ResourcePool(
        node_cpus={0: list(range(8)), 1: list(range(8, 16))},
        gpu_ids=list(range(8)),
        node_mem_gb={0: 64.0, 1: 64.0},
        gpu_topology=two_quads(),
    )


def req(cpu=1, gpu=0, mem=None, exclusive=False,
        numa_local_mem=False, numa_local_gpu=False) -> Request:
    return Request(cpu=cpu, gpu_cores=gpu, mem_gb=mem, exclusive=exclusive,
                   numa_local_mem=numa_local_mem, numa_local_gpu=numa_local_gpu)


def held(cpus, node=0, gpus=(), mem=None) -> Allocation:
    """An allocation already handed out, as re-adoption would rebuild it."""
    return Allocation(
        cpus=list(cpus), numa_nodes=[node], gpus=list(gpus),
        mem_gb=mem, mem_by_node={node: mem} if mem else {}, exclusive=False,
    )


def untracked_pool() -> ResourcePool:
    """A pool whose memory could not be discovered, so it is not tracked."""
    return ResourcePool(node_cpus={0: [0, 1]}, gpu_ids=[], node_mem_gb={0: 0.0})


class TestParseCpuList:
    def test_ranges_and_singles(self):
        assert parse_cpu_list("0-3,8-11") == [0, 1, 2, 3, 8, 9, 10, 11]
        assert parse_cpu_list("5") == [5]
        assert parse_cpu_list("0,2-3") == [0, 2, 3]


class TestDiscovery:
    def test_node_memory_covers_every_node_or_nothing(self):
        # Environment-dependent, so assert the contract rather than figures:
        # either sysfs gave us every node, or we report nothing at all.
        nodes = discover_numa_nodes()
        mem = discover_node_memory_gb(nodes)
        assert mem == {} or set(mem) == set(nodes)
        assert all(gb > 0 for gb in mem.values())


class TestApplyReserve:
    def test_takes_both_from_the_lowest_node(self):
        cpus, mem = apply_reserve(
            {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}, {0: 32.0, 1: 32.0}, 2, 8.0
        )
        # The low cpus go, where the OS lives; node 1 keeps its full width.
        assert cpus == {0: [2, 3], 1: [4, 5, 6, 7]}
        assert mem == {0: 24.0, 1: 32.0}

    def test_spills_to_the_next_node_when_one_cannot_cover_it(self):
        cpus, _ = apply_reserve({0: [0, 1], 1: [2, 3, 4, 5]}, {0: 8.0, 1: 8.0}, 4, 0.0)
        # Node 0 can only give up 1 (never strand a node); node 1 gives the rest.
        assert cpus == {0: [1], 1: [5]}

    def test_never_strands_a_node_or_halves_its_memory(self):
        cpus, mem = apply_reserve({0: [0, 1]}, {0: 8.0}, 99, 99.0)
        assert cpus == {0: [1]}
        assert mem == {0: 4.0}

    def test_zero_reserve_is_a_no_op(self):
        nodes, gb = {0: [0, 1], 1: [2, 3]}, {0: 8.0, 1: 8.0}
        assert apply_reserve(nodes, gb, 0, 0.0) == (nodes, gb)


class TestAllocation:
    def test_cpus_come_from_a_single_node(self):
        pool = make_pool()
        alloc = pool.allocate(req(cpu=3))
        assert alloc is not None
        assert set(alloc.cpus) <= set(pool.node_cpus[alloc.numa_node])
        assert alloc.numa_nodes == [alloc.numa_node]

    def test_never_spans_nodes(self):
        pool = make_pool()
        a, b = pool.allocate(req(cpu=3)), pool.allocate(req(cpu=3))
        assert a is not None and b is not None
        assert a.numa_node != b.numa_node
        # 2 cpus free on each node, but never 3 on one: must wait.
        assert pool.allocate(req(cpu=3)) is None

    def test_gpu_exhaustion(self):
        pool = make_pool()
        a = pool.allocate(req(gpu=3))
        assert a is not None and len(a.gpus) == 3
        assert pool.allocate(req(gpu=2)) is None
        b = pool.allocate(req(gpu=1))
        assert b is not None
        assert not (set(a.gpus) & set(b.gpus))

    def test_memory_is_charged_to_one_node(self):
        pool = make_pool()
        alloc = pool.allocate(req(mem=24.0))
        assert alloc is not None
        assert alloc.mem_by_node == {alloc.numa_node: 24.0}
        assert alloc.spans_nodes is False
        assert alloc.mem_nodes() == [alloc.numa_node]
        other = 1 - alloc.numa_node
        assert pool.free_mem_gb[other] == 32.0

    def test_two_big_jobs_land_on_different_nodes(self):
        # Regression: cpu best-fit alone would pack both onto node 0 (it has
        # the fewest free cpus after the first), and a machine-wide memory
        # counter would agree, because 24+24 <= 64. But both jobs would then
        # be confined to node 0's 32 GiB and collide there.
        pool = make_pool()
        a, b = pool.allocate(req(mem=24.0)), pool.allocate(req(mem=24.0))
        assert a is not None and b is not None
        assert a.numa_node != b.numa_node
        assert not a.spans_nodes and not b.spans_nodes

    def test_memory_pressure_steers_placement(self):
        # Node 0 has fewer free cpus, so cpu best-fit prefers it; its memory
        # is spoken for, so the job must go to node 1 anyway.
        pool = make_pool()
        pool.reserve(held([0], node=0, mem=30.0))
        alloc = pool.allocate(req(cpu=2, mem=16.0))
        assert alloc is not None and alloc.numa_node == 1

    def test_allocate_pinned_to_a_node(self):
        pool = make_pool()
        alloc = pool.allocate(req(cpu=2), node=1)
        assert alloc is not None and alloc.numa_node == 1
        # Pinning to a node that cannot take it fails rather than falling back.
        assert pool.allocate(req(cpu=4), node=1) is None

    def test_release_restores_everything(self):
        pool = make_pool()
        alloc = pool.allocate(req(cpu=4, gpu=2, mem=10.0))
        assert alloc is not None
        pool.release(alloc)
        assert pool.free_cpus[alloc.numa_node] == set(pool.node_cpus[alloc.numa_node])
        assert pool.free_gpus == set(pool.gpu_ids)
        assert pool.free_mem_gb == pool.mem_capacity
        assert pool.active == 0

    def test_exclusive_waits_for_idle_machine(self):
        pool = make_pool()
        a = pool.allocate(req())
        assert a is not None
        assert pool.allocate(req(cpu=1, exclusive=True)) is None
        pool.release(a)
        assert pool.allocate(req(cpu=1, exclusive=True)) is not None

    def test_exclusive_blocks_others(self):
        pool = make_pool()
        excl = pool.allocate(req(cpu=1, exclusive=True))
        assert excl is not None
        assert pool.allocate(req()) is None
        pool.release(excl)
        assert pool.allocate(req()) is not None

    def test_reserve_for_adopted_jobs(self):
        pool = make_pool()
        pool.reserve(held([0, 1], node=0, gpus=[0], mem=8.0))
        assert pool.free_mem_gb[0] == 24.0
        alloc = pool.allocate(req(cpu=4))
        assert alloc is not None
        assert alloc.numa_node == 1  # node 0 only has 2 cpus left


class TestSpanningNodes:
    """A budget that fits no single node is spread across nodes, and the
    charge is exactly what goes into cpuset.mems."""

    def test_budget_larger_than_a_node_spans(self):
        pool = make_pool()
        alloc = pool.allocate(req(mem=48.0))
        assert alloc is not None
        assert alloc.spans_nodes
        # Home node is drained first, so the charge cannot drift past it.
        assert alloc.mem_by_node == {0: 32.0, 1: 16.0}
        assert alloc.mem_nodes() == [0, 1]
        assert pool.free_mem_gb == {0: 0.0, 1: 16.0}

    def test_release_returns_every_node_it_took(self):
        pool = make_pool()
        alloc = pool.allocate(req(mem=48.0))
        pool.release(alloc)
        assert pool.free_mem_gb == {0: 32.0, 1: 32.0}

    def test_spanning_prefers_the_node_with_most_free_memory(self):
        pool = make_pool()
        pool.reserve(held([0], node=0, mem=20.0))
        # Node 0 has 12 GiB free, node 1 has 32. Neither fits 40, so home
        # should be node 1 to leave as little as possible remote.
        alloc = pool.allocate(req(cpu=2, mem=40.0))
        assert alloc is not None
        assert alloc.numa_node == 1
        assert alloc.mem_by_node == {1: 32.0, 0: 8.0}

    def test_numa_local_waits_instead_of_spanning(self):
        pool = make_pool()
        assert pool.allocate(req(mem=48.0, numa_local_mem=True)) is None
        # ...and the same budget without the flag runs immediately.
        assert pool.allocate(req(mem=48.0)) is not None

    def test_numa_local_picks_a_node_that_fits(self):
        pool = make_pool()
        pool.reserve(held([0], node=0, mem=20.0))
        alloc = pool.allocate(req(cpu=2, mem=30.0, numa_local_mem=True))
        assert alloc is not None
        assert alloc.numa_node == 1 and not alloc.spans_nodes

    def test_refused_when_even_the_whole_machine_is_short(self):
        pool = make_pool()
        assert pool.allocate(req(mem=65.0)) is None


class TestExclusiveSpansTheMachine:
    def test_takes_cores_from_every_node(self):
        pool = make_pool()
        alloc = pool.allocate(req(cpu=8, exclusive=True))
        assert alloc is not None
        assert sorted(alloc.cpus) == list(range(8))
        assert alloc.numa_nodes == [0, 1]

    def test_may_use_all_the_memory(self):
        pool = make_pool()
        alloc = pool.allocate(req(cpu=8, mem=64.0, exclusive=True))
        assert alloc is not None
        assert alloc.mem_by_node == {0: 32.0, 1: 32.0}
        assert alloc.mem_nodes() == [0, 1]

    def test_honours_a_node_pin_like_any_other_job(self):
        # Placement is one mechanism with a node ceiling, so the pin applies
        # on the spanning path too. When it was a separate routine the pin was
        # silently dropped and the job took cores from everywhere.
        pool = make_pool()
        alloc = pool.allocate(req(cpu=4, exclusive=True), node=1)
        assert alloc is not None
        assert alloc.numa_nodes == [1]
        assert set(alloc.cpus) <= set(pool.node_cpus[1])

    def test_a_pin_that_cannot_hold_the_request_fails(self):
        pool = make_pool()
        # 8 cores exist, but not on node 1 alone.
        assert pool.allocate(req(cpu=8, exclusive=True), node=1) is None

    def test_numa_local_keeps_an_exclusive_job_on_one_node(self):
        pool = make_pool()
        alloc = pool.allocate(req(cpu=4, mem=32.0, exclusive=True, numa_local_mem=True))
        assert alloc is not None
        assert alloc.numa_nodes == [0]
        assert alloc.mem_by_node == {0: 32.0}


class TestSharedMemory:
    """Without the cpuset controller a job can allocate from any node, so
    memory is tracked as one machine-wide pool."""

    def shared(self) -> ResourcePool:
        return ResourcePool(
            node_cpus={0: [0, 1, 2, 3], 1: [4, 5, 6, 7]},
            gpu_ids=[],
            node_mem_gb={0: 32.0, 1: 32.0},
            mem_confined=False,
        )

    def test_one_job_may_exceed_a_single_node(self):
        pool = self.shared()
        assert pool.validate(req(mem=48.0)) is None
        alloc = pool.allocate(req(mem=48.0))
        assert alloc is not None
        assert pool.free_mem_gb == {SHARED_MEM: 16.0}
        # Both nodes draw from the same pool, so the node choice cannot dodge it.
        assert pool.allocate(req(mem=24.0)) is None

    def test_cpuset_mems_falls_back_to_the_cpu_nodes(self):
        pool = self.shared()
        alloc = pool.allocate(req(mem=8.0))
        assert alloc is not None
        # SHARED_MEM is bookkeeping, never a node id to write to cpuset.mems.
        assert alloc.mem_nodes() == [alloc.numa_node]
        assert alloc.spans_nodes is False

    def test_a_charge_from_the_other_layout_is_still_honoured(self):
        # A charge read back from the state file was keyed by whatever layout
        # was in force when the job started, and a reload can change that
        # (cpuset newly delegated, --no-cgroups toggled). Dropping the charge
        # would hand the same memory out twice.
        confined = make_pool()
        shared_charge = Allocation(
            cpus=[0], numa_nodes=[0], gpus=[], mem_gb=24.0,
            mem_by_node={SHARED_MEM: 24.0}, exclusive=False,
        )
        confined.reserve(shared_charge)
        assert confined.free_mem_gb[0] == 8.0  # folded onto the home node
        confined.release(shared_charge)
        assert confined.free_mem_gb == confined.mem_capacity

        pool = self.shared()
        per_node_charge = Allocation(
            cpus=[0], numa_nodes=[0], gpus=[], mem_gb=48.0,
            mem_by_node={0: 32.0, 1: 16.0}, exclusive=False,
        )
        pool.reserve(per_node_charge)
        assert pool.free_mem_gb == {SHARED_MEM: 16.0}  # summed into one domain
        pool.release(per_node_charge)
        assert pool.free_mem_gb == pool.mem_capacity

    def test_incomplete_per_node_memory_degrades(self):
        # Node 1 has no memory figure: rather than stranding it (nothing with
        # a budget could be placed there), fall back to machine-wide.
        pool = ResourcePool(
            node_cpus={0: [0, 1], 1: [2, 3]}, gpu_ids=[], node_mem_gb={0: 32.0}
        )
        assert pool.mem_confined is False
        assert pool.free_mem_gb == {SHARED_MEM: 32.0}


class TestGpuTopologyParsing:
    def test_reads_the_matrix_past_the_nic_columns(self):
        topo = parse_gpu_topology(TOPO_OUTPUT)
        assert topo
        assert topo.link(0, 1) == "NV2" and topo.link(1, 0) == "NV2"
        assert topo.link(0, 2) == "SYS"
        assert topo.link(2, 3) == "NV2"
        # The NIC has a row and a column, but it is not a GPU.
        assert set(topo.links) == {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
        # Which needs the header's width: without it the NIC column would be
        # read as the CPU affinity field and the NUMA one would be lost.
        assert topo.numa_node == {0: 0, 1: 0, 2: 1, 3: 1}

    def test_output_without_a_numa_affinity_column(self):
        text = "\n".join(
            "\t".join(row)
            for row in [
                ["", "GPU0", "GPU1", "CPU Affinity"],
                ["GPU0", " X ", "PHB", "0-3"],
                ["GPU1", "PHB", " X ", "0-3"],
            ]
        )
        topo = parse_gpu_topology(text)
        assert topo.link(0, 1) == "PHB"
        assert topo.numa_node == {}

    def test_unreadable_output_yields_no_topology(self):
        for text in ["", "No devices found.", "Legend:\n\n  X = Self\n"]:
            topo = parse_gpu_topology(text)
            assert not topo and topo.numa_node == {}


class TestGpuTopologyRanking:
    def test_the_worst_link_leads(self):
        # {0,1,2} is NVLinked twice but has one SYS hop; {0,1,3} is PIX at
        # worst. A collective runs at the speed of its slowest pair.
        topo = GpuTopology(links={
            (0, 1): "NV1", (0, 2): "NV1", (1, 2): "SYS",
            (0, 3): "PIX", (1, 3): "PIX", (2, 3): "PIX",
        })
        assert topo.quality([0, 1, 3]) < topo.quality([0, 1, 2])

    def test_wider_nvlink_breaks_a_tie(self):
        topo = GpuTopology(links={(0, 1): "NV1", (0, 2): "NV4", (1, 2): "NV1"})
        assert topo.quality([0, 2]) < topo.quality([0, 1])

    def test_unknown_links_sort_behind_every_known_one(self):
        topo = GpuTopology(links={(0, 1): "SYS", (0, 2): "NVLINK-C2C"})
        assert topo.quality([0, 1]) < topo.quality([0, 2])

    def test_worst_link_names_the_pacing_hop(self):
        topo = parse_gpu_topology(TOPO_OUTPUT)
        assert topo.worst_link([0, 1]) == "NV2"
        assert topo.worst_link([0, 1, 2]) == "SYS"
        assert topo.worst_link([1]) is None


class TestGpuIslands:
    def test_each_level_recovers_the_machine_that_built_it(self):
        topo = two_quads()
        every = list(range(8))
        assert topo.islands(every, 0) == [[0, 1, 2, 3], [4, 5, 6, 7]]  # NVLink
        assert topo.islands(every, 5) == [every]  # SYS joins the machine

    def test_only_the_free_gpus_are_grouped(self):
        assert two_quads().islands([1, 2, 5], 0) == [[1, 2], [5]]

    def test_a_level_below_the_finest_link_leaves_singletons(self):
        assert two_quads().islands([0, 1], -1) == [[0], [1]]


class TestGpuPlacement:
    def test_prefers_an_adjacent_pair_to_the_lowest_indices(self):
        # The reported bug: with GPU1 busy, index order hands out 0+2, a pair
        # on opposite sides of the machine, while 2+3 sit on one NVLink.
        pool = topo_pool()
        pool.reserve(held([0], node=0, gpus=[1]))
        alloc = pool.allocate(req(gpu=2))
        assert alloc is not None and alloc.gpus == [2, 3]

    def test_index_order_when_the_topology_is_unknown(self):
        pool = make_pool()
        pool.reserve(held([0], node=0, gpus=[1]))
        alloc = pool.allocate(req(gpu=2))
        assert alloc is not None and alloc.gpus == [0, 2]

    def test_takes_a_distant_pair_rather_than_waiting(self):
        # Adjacency is a preference, never a reason to leave a job queued.
        pool = topo_pool()
        pool.reserve(held([0], node=0, gpus=[1, 3]))
        alloc = pool.allocate(req(gpu=2))
        assert alloc is not None and alloc.gpus == [0, 2]

    def test_every_gpu_when_the_job_asks_for_all_of_them(self):
        pool = topo_pool()
        alloc = pool.allocate(req(cpu=4, gpu=4))
        assert alloc is not None and alloc.gpus == [0, 1, 2, 3]

    def test_takes_the_broken_island_and_leaves_the_whole_one(self):
        # Both quads would serve a 2-gpu job equally well, so index order
        # would carve up the untouched one and leave nothing for a job that
        # needs four. Best fit spends the quad that is already spent.
        pool = quad_pool()
        pool.reserve(held([8], node=1, gpus=[4]))
        alloc = pool.allocate(req(cpu=2, gpu=2))
        assert alloc is not None and alloc.gpus == [5, 6]
        # ...so the 4-gpu job behind it still gets a whole NVLinked quad.
        after = pool.allocate(req(cpu=2, gpu=4))
        assert after is not None and after.gpus == [0, 1, 2, 3]

    def test_the_finest_level_with_room_decides_the_pacing_link(self):
        pool = quad_pool()
        pool.reserve(held([0], node=0, gpus=[0, 1]))
        # Six gpus free but no NVLinked island holds five, so this job pays
        # for SYS wherever it goes. Taking a whole quad and one straggler
        # leaves four crossing pairs; splitting it 3+2 would leave six.
        alloc = pool.allocate(req(cpu=2, gpu=5))
        assert alloc is not None and alloc.gpus == [2, 4, 5, 6, 7]
        assert pool.gpu_topology.worst_link(alloc.gpus) == "SYS"

    def test_islands_are_taken_apart_when_too_wide_to_score(self, monkeypatch):
        # Forces the path for an island holding more sets than we will score:
        # it is spread over the finer islands inside, biggest first.
        monkeypatch.setattr(resources, "_MAX_GPU_LINK_SCANS", 0)
        pool = quad_pool()
        pool.reserve(held([8], node=1, gpus=[4]))
        assert pool.allocate(req(cpu=2, gpu=2)).gpus == [5, 6]
        # Same answer the scored path gives, for the same reason.
        pool = quad_pool()
        pool.reserve(held([0], node=0, gpus=[0, 1]))
        assert pool.allocate(req(cpu=2, gpu=5)).gpus == [2, 4, 5, 6, 7]

    def test_cpus_land_on_the_node_the_gpus_hang_off(self):
        pool = topo_pool()
        pool.reserve(held([0], node=0, gpus=[0, 1]))
        alloc = pool.allocate(req(cpu=2, gpu=1))
        # Node 0 has fewer free cpus, so best fit alone would put it there;
        # the only free gpus are on node 1, and feeding one from node 0 would
        # cross the interconnect.
        assert alloc is not None and alloc.gpus == [2]
        assert alloc.numa_node == 1

    def test_numa_local_gpu_waits_rather_than_span_nodes(self):
        pool = quad_pool()
        pool.reserve(held([0], node=0, gpus=[0, 1, 2]))
        # One gpu left on node 0 and four on node 1, so a 2-gpu job can be
        # served, but only by node 1.
        alloc = pool.allocate(req(cpu=2, gpu=2, numa_local_gpu=True))
        assert alloc is not None
        assert alloc.gpus == [4, 5] and alloc.numa_node == 1

    def test_numa_local_gpu_queues_when_no_node_has_enough(self):
        pool = quad_pool()
        pool.reserve(held([0], node=0, gpus=[0, 1, 2]))
        pool.reserve(held([8], node=1, gpus=[4, 5, 6]))
        # Two gpus free, one per node: enough for the job, but not on one node.
        assert pool.allocate(req(cpu=2, gpu=2, numa_local_gpu=True)) is None
        # ...and without the flag the same job runs, spanning the nodes.
        alloc = pool.allocate(req(cpu=2, gpu=2))
        assert alloc is not None and alloc.gpus == [3, 7]

    def test_numa_local_gpu_puts_the_cpus_on_the_gpus_node(self):
        pool = quad_pool()
        # Node 0 has every core free, so cpu best-fit would prefer it; the
        # job's gpus are on node 1, and the flag makes that binding.
        pool.reserve(held([0], node=0, gpus=[0, 1, 2, 3]))
        alloc = pool.allocate(req(cpu=2, gpu=2, numa_local_gpu=True))
        assert alloc is not None and alloc.numa_node == 1
        assert set(alloc.cpus) <= set(pool.node_cpus[1])

    def test_numa_local_gpu_is_inert_without_gpus(self):
        pool = quad_pool()
        assert pool.allocate(req(cpu=2, numa_local_gpu=True)) is not None

    def test_a_node_pin_that_fights_the_gpus_finds_nothing(self):
        pool = quad_pool()
        pool.reserve(held([0], node=0, gpus=[0, 1, 2, 3]))
        assert pool.allocate(req(cpu=2, gpu=2, numa_local_gpu=True), node=0) is None

    def test_would_fit_agrees_with_allocate(self):
        # would_fit skips the search for the best set, on the grounds that
        # which gpus a job gets never decides whether it fits. If that ever
        # stopped holding, jobs would be told they fit and then not start.
        pool = topo_pool()
        pool.reserve(held([0], node=0, gpus=[1]))
        for request in [
            req(gpu=2), req(cpu=4, gpu=3), req(cpu=4, gpu=4),
            req(cpu=2, gpu=1, mem=64.0), req(cpu=2, gpu=1, mem=48.0, numa_local_mem=True),
            req(cpu=2, gpu=2, numa_local_gpu=True), req(cpu=2, gpu=3, numa_local_gpu=True),
        ]:
            assert pool.would_fit(request) == (pool.clone().allocate(request) is not None)

    def test_gpu_affinity_never_overrides_the_memory_budget(self):
        pool = topo_pool()
        pool.reserve(held([4], node=1, gpus=[0, 1], mem=30.0))
        # Both best fit and the free gpus point at node 1, but it has 2 GiB
        # left, so the budget still decides.
        alloc = pool.allocate(req(cpu=2, gpu=1, mem=16.0))
        assert alloc is not None and alloc.numa_node == 0


class TestProportionalShare:
    def test_share_tracks_the_fraction_of_a_node_asked_for(self):
        pool = make_pool()  # 4 cpus / 32 GiB per node
        assert pool.proportional_share(1) == 8.0
        assert pool.proportional_share(2) == 16.0
        assert pool.proportional_share(4) == 32.0

    def test_resolved_against_the_stingiest_node(self):
        # Node 1 gives 4 GiB per cpu, node 0 gives 16. A budget fixed at
        # submit time has to assume the worse node so it fits either.
        pool = ResourcePool(
            node_cpus={0: [0, 1], 1: [2, 3]}, gpu_ids=[], node_mem_gb={0: 32.0, 1: 8.0}
        )
        assert pool.proportional_share(1) == 4.0

    def test_zero_when_memory_is_untracked(self):
        assert untracked_pool().proportional_share(1) == 0.0


class TestFreeViewsAndClone:
    def test_free_views_track_allocation(self):
        pool = make_pool()
        assert pool.free_cpus_by_node() == {0: 4, 1: 4}
        assert pool.free_mem_gb == {0: 32.0, 1: 32.0}
        assert len(pool.free_gpus) == 4
        alloc = pool.allocate(req(cpu=2, gpu=1, mem=10.0))
        assert alloc is not None and alloc.numa_node == 0
        assert pool.free_cpus_by_node() == {0: 2, 1: 4}
        assert pool.free_mem_gb == {0: 22.0, 1: 32.0}
        assert len(pool.free_gpus) == 3

    def test_untracked_memory_is_a_zero_capacity_domain(self):
        pool = untracked_pool()
        assert pool.tracks_memory is False
        assert pool.free_mem_gb == {0: 0.0}
        assert pool.free_cpus_by_node() == {0: 2}

    def test_clone_is_independent(self):
        pool = make_pool()
        twin = pool.clone()
        twin.allocate(req(cpu=4, gpu=4, mem=32.0, exclusive=True))
        assert pool.free_cpus_by_node() == {0: 4, 1: 4}
        assert pool.free_mem_gb == {0: 32.0, 1: 32.0}
        assert twin.free_cpus_by_node() == {0: 0, 1: 4}
        assert twin.free_mem_gb == {0: 0.0, 1: 32.0}


class TestValidate:
    def test_cpu_larger_than_biggest_node(self):
        pool = make_pool()
        assert pool.validate(req(cpu=5)) is not None
        assert pool.validate(req(cpu=4)) is None

    def test_exclusive_may_ask_for_every_core(self):
        pool = make_pool()
        assert pool.validate(req(cpu=8, exclusive=True)) is None
        assert pool.validate(req(cpu=9, exclusive=True)) is not None
        # ...but not once it is pinned to a single node.
        assert pool.validate(req(cpu=8, exclusive=True, numa_local_mem=True)) is not None

    def test_too_many_gpus(self):
        pool = make_pool()
        assert pool.validate(req(gpu=5)) is not None
        assert pool.validate(req(gpu=4)) is None

    def test_budget_may_span_nodes_by_default(self):
        pool = make_pool()
        # 48 fits the machine but no single node: allowed, it will spread.
        assert pool.validate(req(mem=48.0)) is None
        assert pool.validate(req(mem=64.0)) is None
        problem = pool.validate(req(mem=65.0))
        assert problem is not None and "available to jobs" in problem

    def test_numa_local_is_capped_at_one_node(self):
        pool = make_pool()
        assert pool.validate(req(mem=32.0, numa_local_mem=True)) is None
        problem = pool.validate(req(mem=48.0, numa_local_mem=True))
        # Unsatisfiable however long it waits, so it is refused at submit
        # time -- and the message has to name both ways out.
        assert problem is not None
        assert "--numa-local" in problem and "--exclusive" in problem

    def test_numa_local_gpu_is_capped_at_one_node_of_gpus(self):
        pool = quad_pool()
        assert pool.validate(req(cpu=2, gpu=4, numa_local_gpu=True)) is None
        # 8 gpus exist, but never 5 on one node: refused now rather than
        # queued for a machine that can never satisfy it.
        problem = pool.validate(req(cpu=2, gpu=5, numa_local_gpu=True))
        assert problem is not None and "--numa-local-gpu" in problem
        assert pool.validate(req(cpu=2, gpu=5)) is None

    def test_numa_local_gpu_needs_a_topology_that_names_the_nodes(self):
        pool = make_pool()  # gpus, but nothing says where they hang off
        problem = pool.validate(req(gpu=1, numa_local_gpu=True))
        assert problem is not None and "nvidia-smi topo -m" in problem
        # It only bites a job that actually asked for gpus.
        assert pool.validate(req(cpu=2, numa_local_gpu=True)) is None

    def test_asymmetric_nodes_are_judged_by_the_largest(self):
        pool = ResourcePool(
            node_cpus={0: [0, 1], 1: [2, 3]}, gpu_ids=[], node_mem_gb={0: 32.0, 1: 64.0}
        )
        # 48 fits node 1 but not node 0: satisfiable, so it must queue rather
        # than be rejected.
        assert pool.validate(req(mem=48.0, numa_local_mem=True)) is None
        assert pool.validate(req(mem=65.0, numa_local_mem=True)) is not None

    def test_memory_unchecked_when_untracked(self):
        assert untracked_pool().validate(req(mem=999.0)) is None
