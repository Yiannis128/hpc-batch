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
        job = make_job(state=RUNNING, pid=1234, cpus=[0, 1], gpus=[2], numa_node=0)
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

    def test_allocation_mirrors_held_resources(self):
        job = make_job(state=RUNNING, cpus=[0, 1], gpus=[2], numa_node=1)
        alloc = job.allocation()
        assert alloc.cpus == [0, 1]
        assert alloc.numa_node == 1
        assert alloc.gpus == [2]
        assert alloc.mem_gb == job.max_mem_gb
        assert alloc.exclusive == job.exclusive

    def test_deadline_running_vs_queued(self):
        now = 1000.0
        running = make_job(state=RUNNING, start_time=900.0, max_time_s=100)
        assert running.deadline(now) == 1000.0  # start + max_time
        queued = make_job(state=QUEUED, max_time_s=100)
        assert queued.deadline(now) == 1100.0  # measured from now

    def test_public_row_fields(self):
        row = make_job().public_row(time.time())
        assert set(row) == {
            "user", "id", "command", "uptime_s", "max_time_s", "exclusive", "state",
        }
        assert row["state"] == QUEUED
