import json
import time

from hpc_batch.jobs import DONE, QUEUED, RUNNING, Job


def make_job(**overrides) -> Job:
    defaults = dict(
        id=1,
        user="alice",
        uid=1000,
        gid=1000,
        argv=["echo", "hello world"],
        cwd="/home/alice",
        cpu=2,
        gpu_cores=1,
        max_mem_gb=8.0,
        max_time_s=3600,
        exclusive=False,
        submit_time=time.time(),
    )
    defaults.update(overrides)
    return Job(**defaults)


class TestJob:
    def test_roundtrip(self):
        job = make_job(state=RUNNING, pid=1234, cpus=[0, 1], gpus=[2],
                       numa_node=0, numa_nodes=[0], mem_by_node={0: 8.0})
        assert Job.from_dict(job.to_dict()) == job

    def test_from_dict_ignores_unknown_fields(self):
        data = make_job().to_dict()
        data["future_field"] = "whatever"
        assert Job.from_dict(data).id == 1

    def test_command_is_shell_quoted(self):
        assert make_job().command() == "echo 'hello world'"

    def test_uptime_only_when_running(self):
        job = make_job()
        now = time.time()
        assert job.uptime(now) is None  # queued
        job.state = RUNNING
        job.start_time = now - 10
        assert 9 < job.uptime(now) < 11
        job.state = DONE
        assert job.uptime(now) is None

    def test_state_file_roundtrip_keeps_memory_charge_per_node(self):
        # JSON object keys are strings. If they came back as strings the pool
        # would look for domain "0" instead of 0, find nothing, and quietly
        # never give the memory back.
        job = make_job(state=RUNNING, cpus=[0, 1], numa_node=0, numa_nodes=[0, 1],
                       mem_by_node={0: 32.0, 1: 16.0})
        revived = Job.from_dict(json.loads(json.dumps(job.to_dict())))
        assert revived.mem_by_node == {0: 32.0, 1: 16.0}
        assert revived == job

    def test_allocation_mirrors_held_resources(self):
        job = make_job(state=RUNNING, cpus=[0, 1], gpus=[2], numa_node=1,
                       numa_nodes=[1], mem_by_node={1: 8.0})
        alloc = job.allocation()
        assert alloc.cpus == [0, 1]
        assert alloc.numa_node == 1
        assert alloc.numa_nodes == [1]
        assert alloc.gpus == [2]
        assert alloc.mem_gb == job.max_mem_gb
        assert alloc.mem_by_node == {1: 8.0}
        assert alloc.exclusive == job.exclusive

    def test_older_state_is_normalised_on_load(self):
        # A job persisted before per-node charges existed still has to give
        # its memory back somewhere, or the node leaks capacity. Filling it in
        # at load (not in allocation()) means it is written back on the next
        # persist and every reader sees it, not just the scheduler.
        legacy = make_job(state=RUNNING, cpus=[0], numa_node=1).to_dict()
        del legacy["numa_nodes"], legacy["mem_by_node"]

        job = Job.from_dict(legacy)

        assert job.numa_nodes == [1]
        assert job.mem_by_node == {1: 8.0}
        assert job.allocation().mem_by_node == {1: 8.0}
        assert job.public_row(1000.0)["mem_spans_nodes"] is False
        # Normalised once: reloading the written-back form changes nothing.
        assert Job.from_dict(job.to_dict()) == job

    def test_queued_jobs_are_left_unplaced(self):
        # Placement fields are meaningless before a job starts, and inventing
        # them would charge the pool for a job that holds nothing.
        job = Job.from_dict(make_job(state=QUEUED).to_dict())
        assert job.numa_nodes == [] and job.mem_by_node == {}

    def test_request_carries_the_placement_flags(self):
        job = make_job(cpu=4, gpu_cores=2, numa_local=True, exclusive=True)
        req = job.request()
        assert (req.cpu, req.gpu_cores, req.mem_gb) == (4, 2, 8.0)
        assert req.exclusive and req.numa_local

    def test_deadline_running_vs_queued(self):
        now = 1000.0
        running = make_job(state=RUNNING, start_time=900.0, max_time_s=100)
        assert running.deadline(now) == 1000.0  # start + max_time
        queued = make_job(state=QUEUED, max_time_s=100)
        assert queued.deadline(now) == 1100.0  # measured from now

    def test_public_row_fields(self):
        row = make_job().public_row(time.time())
        assert set(row) == {
            "user", "id", "command", "start_time", "uptime_s", "max_time_s",
            "exclusive", "state", "exit_code", "reason", "output_dest",
            "output_error", "max_mem_gb", "mem_defaulted", "mem_spans_nodes",
        }
        assert row["state"] == QUEUED
        assert row["start_time"] is None  # queued: it has not started yet

    def test_public_row_reports_the_start_time(self):
        row = make_job(state=RUNNING, start_time=900.0).public_row(1000.0)
        assert row["start_time"] == 900.0

    def test_elapsed_survives_the_job_ending(self):
        now = 1000.0
        queued = make_job(state=QUEUED)
        assert queued.elapsed(now) is None

        running = make_job(state=RUNNING, start_time=990.0)
        assert running.elapsed(now) == 10.0  # counts up against now

        # uptime() goes None once the job is done, elapsed() stays final so
        # `dispatch list --finished` can still show how long it ran.
        done = make_job(state=DONE, start_time=900.0, end_time=925.0)
        assert done.uptime(now) is None
        assert done.elapsed(now) == 25.0

    def test_public_row_reports_exit_status_of_finished_job(self):
        row = make_job(state=DONE, start_time=900.0, end_time=925.0,
                       exit_code=3).public_row(1000.0)
        assert row["exit_code"] == 3
        assert row["reason"] is None
        assert row["uptime_s"] == 25.0

        killed = make_job(state=DONE, start_time=900.0, end_time=925.0,
                          exit_code=None, reason="timeout").public_row(1000.0)
        assert killed["reason"] == "timeout"
