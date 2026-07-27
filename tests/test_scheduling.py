"""Scheduling-policy tests, driven through the pure `plan` function."""

from hpc_batch.jobs import QUEUED, RUNNING, Job
from hpc_batch.resources import ResourcePool
from hpc_batch.scheduling import (
    EASY_BACKFILL,
    FIFO_STRICT,
    STRICT_BACKFILL,
    Reservation,
    plan,
)
from test_resources import held

NOW = 1000.0


def gpu_pool(n_gpus: int) -> ResourcePool:
    """One roomy NUMA node so GPUs are the only scarce resource."""
    return ResourcePool(
        node_cpus={0: list(range(64))},
        gpu_ids=list(range(n_gpus)),
        node_mem_gb={0: 256.0},
    )


def job(jid: int, gpu: int, max_time: int = 3600, state: str = QUEUED,
        start: float | None = None, cpu: int = 1) -> Job:
    j = Job(
        id=jid, user="u", uid=1000, gid=1000, argv=["x"], cwd="/",
        cpu=cpu, gpu_cores=gpu, max_mem_gb=None, max_time_s=max_time,
        exclusive=False, state=state, start_time=start,
    )
    if state == RUNNING:
        j.cpus = [jid]
        j.numa_node = 0
        j.gpus = []  # set by the caller when the running job holds GPUs
    return j


def started_ids(result) -> set[int]:
    to_start, _ = result
    return {j.id for j, _ in to_start}


# The queue from the user's question: GPU demands [1, 4, 2, 1, 2] on a
# 4-GPU machine. Jobs are numbered 1..5 in submission order.
def example_queue() -> list[Job]:
    return [job(1, 1), job(2, 4), job(3, 2), job(4, 1), job(5, 2)]


class TestFifoStrict:
    def test_head_of_line_blocking(self):
        pool = gpu_pool(4)
        result = plan(FIFO_STRICT, example_queue(), pool, [], NOW, None)
        # Only job 1 runs; job 2 (needs 4) blocks everything behind it.
        assert started_ids(result) == {1}
        assert result[1] is None

    def test_starts_everything_that_fits_in_order(self):
        pool = gpu_pool(8)
        result = plan(FIFO_STRICT, example_queue(), pool, [], NOW, None)
        # 1+4+2+1 = 8 exactly; job 5 (needs 2) then blocks.
        assert started_ids(result) == {1, 2, 3, 4}


class TestEasyBackfill:
    def test_backfills_idle_capacity_past_the_blocked_head(self):
        pool = gpu_pool(4)
        to_start, reservation = plan(EASY_BACKFILL, example_queue(), pool, [], NOW, None)
        # Jobs 1, 3, 4 run (1+2+1 = 4); job 2 is the reserved head; job 5
        # does not fit. This is exactly the user's expectation.
        assert {j.id for j, _ in to_start} == {1, 3, 4}
        assert reservation.head_id == 2

    def test_backfills_a_job_submitted_after_the_head_blocked(self):
        # Mirrors incremental submission: the head blocks first, and a small
        # job that fits in the still-idle capacity arrives on a later tick.
        pool = gpu_pool(4)
        # Tick 0: job 1 (needs 1) runs; job 2 (needs 4) blocks -> 3 GPUs idle.
        to_start, reservation = plan(EASY_BACKFILL, [job(1, 1), job(2, 4)], pool, [], NOW, None)
        assert {j.id for j, _ in to_start} == {1}
        assert reservation.head_id == 2 and len(reservation.budget.free_gpus) == 3
        # Tick 1: job 3 (needs 2) arrives; it fits within the frozen budget.
        result = plan(EASY_BACKFILL, [job(2, 4), job(3, 2)], pool, [], NOW, reservation)
        assert started_ids(result) == {3}

    def test_holds_freed_resources_for_the_head(self):
        pool = gpu_pool(4)
        # Tick 0: establish. Jobs 1, 3, 4 running; job 2 reserved.
        to_start, reservation = plan(EASY_BACKFILL, example_queue(), pool, [], NOW, None)
        allocs = {j.id: a for j, a in to_start}
        assert reservation.head_id == 2

        # Tick 1: job 1 finishes, freeing 1 GPU. A brand-new 1-GPU job (id 6)
        # arrives that WOULD fit -- but the freed GPU must be held for job 2,
        # because the backfill budget was already spent by jobs 3 and 4.
        pool.release(allocs[1])
        queued = [job(2, 4), job(5, 2), job(6, 1)]
        result = plan(EASY_BACKFILL, queued, pool, [], NOW, reservation)
        assert started_ids(result) == set()  # nothing started; GPU held for job 2
        assert result[1].head_id == 2

        # Tick 2: jobs 3 and 4 finish too -> 4 GPUs free -> job 2 finally runs.
        pool.release(allocs[3])
        pool.release(allocs[4])
        to_start2, reservation2 = plan(EASY_BACKFILL, queued, pool, [], NOW, result[1])
        assert {j.id for j, _ in to_start2} == {2}
        assert reservation2.head_id == 5  # job 5 is the new blocked head

    def test_new_head_after_reserved_head_leaves(self):
        pool = gpu_pool(4)
        # Reserved head (id 2) is gone from the queue; a fitting job remains.
        stale = Reservation(head_id=2, budget=gpu_pool(0))
        result = plan(EASY_BACKFILL, [job(5, 2)], pool, [], NOW, stale)
        assert started_ids(result) == {5}


class TestEasyBackfillPerNodeBudget:
    """The frozen budget is per NUMA node, because that is the unit a job is
    placed in. A machine-wide total would let a backfill job spend capacity
    that lives on a different node from the one it lands on."""

    def numa_pool(self) -> ResourcePool:
        return ResourcePool(
            node_cpus={0: [0, 1, 2, 3], 1: list(range(4, 12))},
            gpu_ids=[],
            node_mem_gb={0: 32.0, 1: 64.0},
        )

    def test_budget_is_not_spent_on_a_node_that_freed_up_later(self):
        pool = self.numa_pool()
        # node 0: 3 of 4 cpus busy (1 idle). node 1: 4 of 8 busy (4 idle).
        node0_job = held([0, 1, 2], 0)
        pool.reserve(node0_job)
        pool.reserve(held([4, 5, 6, 7], 1))

        # Tick 0: the head needs 5 cpus on one node; neither has them.
        head = job(1, 0, cpu=5)
        to_start, reservation = plan(EASY_BACKFILL, [head], pool, [], NOW, None)
        assert to_start == []
        assert reservation.head_id == 1
        assert reservation.budget.free_cpus_by_node() == {0: 1, 1: 4}

        # Tick 1: node 0's job finishes, so node 0 now has 4 idle cpus that
        # must be held for the head. A 3-cpu job arrives behind it.
        pool.release(node0_job)
        later = job(2, 0, cpu=3)
        to_start, _ = plan(EASY_BACKFILL, [head, later], pool, [], NOW, reservation)

        # It runs -- node 1 had 4 cpus idle when the budget froze -- but on
        # node 1, not on the cpus node 0 freed afterwards. A machine-wide
        # budget would have said "5 cpus free, take 3" and cpu best-fit would
        # have put it on node 0, eating the head's freed capacity.
        assert {j.id for j, _ in to_start} == {2}
        assert to_start[0][1].numa_node == 1
        assert pool.free_cpus_by_node()[0] == 4

    def test_budget_refuses_a_job_no_single_node_can_hold(self):
        pool = self.numa_pool()
        pool.reserve(held([0, 1], 0))
        pool.reserve(held([4, 5, 6, 7, 8, 9], 1))
        # 2 idle on node 0 and 2 on node 1: 4 machine-wide, 2 usable by one job.
        head = job(1, 0, cpu=8)
        to_start, reservation = plan(EASY_BACKFILL, [head], pool, [], NOW, None)
        assert to_start == []
        assert reservation.budget.free_cpus_by_node() == {0: 2, 1: 2}
        wide = job(2, 0, cpu=4)
        to_start, _ = plan(EASY_BACKFILL, [head, wide], pool, [], NOW, reservation)
        assert to_start == []

    def test_a_budget_that_spans_nodes_can_still_backfill(self):
        # The snapshot places candidates with the real allocator, so a job
        # whose memory has to span nodes is judged like any other. Counters
        # beside the pool could not express this and had to refuse it.
        pool = self.numa_pool()  # node 0: 4 cpu/32 GiB, node 1: 8 cpu/64 GiB
        head = job(1, 0, cpu=9)  # wider than either node, so it blocks at once
        spanning = job(2, 0, cpu=1)
        spanning.max_mem_gb = 80.0  # 96 GiB machine-wide, but no node holds 80

        to_start, _ = plan(EASY_BACKFILL, [head, spanning], pool, [], NOW, None)

        assert {j.id for j, _ in to_start} == {2}
        alloc = to_start[0][1]
        assert alloc.spans_nodes
        assert alloc.mem_by_node == {1: 64.0, 0: 16.0}

    def test_an_exclusive_job_never_backfills(self):
        # The snapshot froze while the machine was busy, so it refuses an
        # exclusive candidate for the same reason the live pool would. A
        # cpu/memory counter had no notion of exclusivity at all and judged
        # such a job by a rule that did not describe it.
        pool = self.numa_pool()
        pool.reserve(held([0, 1, 2], 0))
        head = job(1, 0, cpu=9)
        greedy = job(2, 0, cpu=2)
        greedy.exclusive = True

        to_start, _ = plan(EASY_BACKFILL, [head, greedy], pool, [], NOW, None)

        assert to_start == []

    def test_memory_budget_is_held_per_node(self):
        pool = self.numa_pool()
        head = job(1, 0, cpu=4)
        head.max_mem_gb = 60.0  # only node 1 is big enough
        big = job(2, 0, cpu=1)
        big.max_mem_gb = 40.0  # also only node 1
        small = job(3, 0, cpu=1)
        small.max_mem_gb = 20.0

        # Head takes node 1 (node 0 cannot hold 60 GiB); nothing blocks yet.
        to_start, reservation = plan(EASY_BACKFILL, [head], pool, [], NOW, None)
        assert to_start[0][1].numa_node == 1
        assert reservation is None

        # Now node 1 has 4 GiB left. The 40 GiB job cannot go anywhere and
        # becomes the head; the 20 GiB job backfills onto node 0.
        to_start, reservation = plan(EASY_BACKFILL, [big, small], pool, [], NOW, None)
        assert reservation.head_id == 2
        assert {j.id for j, _ in to_start} == {3}
        assert to_start[0][1].numa_node == 0


class TestStrictBackfill:
    def _setup(self):
        # 6 GPUs. One running job holds 2 GPUs until deadline 1050.
        pool = gpu_pool(6)
        running = job(9, 2, max_time=100, state=RUNNING, start=950.0)
        running.gpus = [0, 1]
        pool.reserve(running.allocation())  # 4 GPUs free
        return pool, running

    def test_backfills_only_jobs_that_finish_before_the_reservation(self):
        pool, running = self._setup()
        # Head needs all 6 GPUs (blocked). A short 2-GPU job finishes well
        # before the head's reservation (1050); a long one does not.
        head = job(1, 6)
        short = job(2, 2, max_time=30)   # 1000 + 30 = 1030 <= 1050 -> allowed
        long = job(3, 2, max_time=500)   # 1000 + 500 = 1500 > 1050 -> refused
        result = plan(STRICT_BACKFILL, [head, short, long], pool, [running], NOW, None)
        assert started_ids(result) == {2}

    def test_easy_would_run_both_but_strict_protects_the_head(self):
        pool, running = self._setup()
        head = job(1, 6)
        short = job(2, 2, max_time=30)
        long = job(3, 2, max_time=500)
        queue = [head, short, long]

        # easy-backfill fills the 4 idle GPUs with both backfill jobs...
        easy = plan(EASY_BACKFILL, queue, pool.clone(), [running], NOW, None)
        assert started_ids(easy) == {2, 3}

        # ...strict-backfill refuses the long one because it would delay head.
        strict = plan(STRICT_BACKFILL, queue, pool.clone(), [running], NOW, None)
        assert started_ids(strict) == {2}
