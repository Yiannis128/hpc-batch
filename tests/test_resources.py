from hpc_batch.resources import Allocation, ResourcePool, parse_cpu_list


def make_pool() -> ResourcePool:
    return ResourcePool(
        node_cpus={0: [0, 1, 2, 3], 1: [4, 5, 6, 7]},
        gpu_ids=[0, 1, 2, 3],
        total_mem_gb=64.0,
    )


class TestParseCpuList:
    def test_ranges_and_singles(self):
        assert parse_cpu_list("0-3,8-11") == [0, 1, 2, 3, 8, 9, 10, 11]
        assert parse_cpu_list("5") == [5]
        assert parse_cpu_list("0,2-3") == [0, 2, 3]


class TestAllocation:
    def test_cpus_come_from_a_single_node(self):
        pool = make_pool()
        alloc = pool.allocate(cpu=3, gpu_cores=0, mem_gb=None, exclusive=False)
        assert alloc is not None
        for node, cpus in pool.node_cpus.items():
            if alloc.numa_node == node:
                assert set(alloc.cpus) <= set(cpus)

    def test_never_spans_nodes(self):
        pool = make_pool()
        a = pool.allocate(cpu=3, gpu_cores=0, mem_gb=None, exclusive=False)
        b = pool.allocate(cpu=3, gpu_cores=0, mem_gb=None, exclusive=False)
        assert a is not None and b is not None
        assert a.numa_node != b.numa_node
        # 2 cpus free on each node, but never 3 on one: must wait.
        assert pool.allocate(cpu=3, gpu_cores=0, mem_gb=None, exclusive=False) is None

    def test_gpu_exhaustion(self):
        pool = make_pool()
        a = pool.allocate(cpu=1, gpu_cores=3, mem_gb=None, exclusive=False)
        assert a is not None and len(a.gpus) == 3
        assert pool.allocate(cpu=1, gpu_cores=2, mem_gb=None, exclusive=False) is None
        b = pool.allocate(cpu=1, gpu_cores=1, mem_gb=None, exclusive=False)
        assert b is not None
        assert not (set(a.gpus) & set(b.gpus))

    def test_memory_exhaustion(self):
        pool = make_pool()
        assert pool.allocate(cpu=1, gpu_cores=0, mem_gb=48.0, exclusive=False) is not None
        assert pool.allocate(cpu=1, gpu_cores=0, mem_gb=32.0, exclusive=False) is None
        assert pool.allocate(cpu=1, gpu_cores=0, mem_gb=16.0, exclusive=False) is not None

    def test_release_restores_everything(self):
        pool = make_pool()
        alloc = pool.allocate(cpu=4, gpu_cores=2, mem_gb=10.0, exclusive=False)
        assert alloc is not None
        pool.release(alloc)
        assert pool.free_cpus[alloc.numa_node] == set(pool.node_cpus[alloc.numa_node])
        assert pool.free_gpus == set(pool.gpu_ids)
        assert pool.free_mem_gb == pool.total_mem_gb
        assert pool.active == 0

    def test_exclusive_waits_for_idle_machine(self):
        pool = make_pool()
        a = pool.allocate(cpu=1, gpu_cores=0, mem_gb=None, exclusive=False)
        assert a is not None
        assert pool.allocate(cpu=1, gpu_cores=0, mem_gb=None, exclusive=True) is None
        pool.release(a)
        assert pool.allocate(cpu=1, gpu_cores=0, mem_gb=None, exclusive=True) is not None

    def test_exclusive_blocks_others(self):
        pool = make_pool()
        excl = pool.allocate(cpu=1, gpu_cores=0, mem_gb=None, exclusive=True)
        assert excl is not None
        assert pool.allocate(cpu=1, gpu_cores=0, mem_gb=None, exclusive=False) is None
        pool.release(excl)
        assert pool.allocate(cpu=1, gpu_cores=0, mem_gb=None, exclusive=False) is not None

    def test_reserve_for_adopted_jobs(self):
        pool = make_pool()
        pool.reserve(Allocation(cpus=[0, 1], numa_node=0, gpus=[0], mem_gb=8.0, exclusive=False))
        alloc = pool.allocate(cpu=4, gpu_cores=0, mem_gb=None, exclusive=False)
        assert alloc is not None
        assert alloc.numa_node == 1  # node 0 only has 2 cpus left


class TestFreeTotalsAndClone:
    def test_free_totals_tracks_allocation(self):
        pool = make_pool()
        assert pool.free_totals() == (8, 4, 64.0)
        pool.allocate(cpu=2, gpu_cores=1, mem_gb=10.0, exclusive=False)
        assert pool.free_totals() == (6, 3, 54.0)

    def test_free_totals_mem_none_when_untracked(self):
        pool = ResourcePool(node_cpus={0: [0, 1]}, gpu_ids=[], total_mem_gb=0.0)
        assert pool.free_totals() == (2, 0, None)

    def test_clone_is_independent(self):
        pool = make_pool()
        twin = pool.clone()
        twin.allocate(cpu=4, gpu_cores=4, mem_gb=64.0, exclusive=True)
        # Mutating the clone leaves the original untouched.
        assert pool.free_totals() == (8, 4, 64.0)
        assert twin.free_totals() == (4, 0, 0.0)


class TestValidate:
    def test_cpu_larger_than_biggest_node(self):
        pool = make_pool()
        assert pool.validate(5, 0, None) is not None
        assert pool.validate(4, 0, None) is None

    def test_too_many_gpus(self):
        pool = make_pool()
        assert pool.validate(1, 5, None) is not None
        assert pool.validate(1, 4, None) is None

    def test_too_much_memory(self):
        pool = make_pool()
        assert pool.validate(1, 0, 65.0) is not None
        assert pool.validate(1, 0, 64.0) is None
