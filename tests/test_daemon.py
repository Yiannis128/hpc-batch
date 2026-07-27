import os
from pathlib import Path

from hpc_batch.daemon import Config, Daemon, run_as_user
from hpc_batch.jobs import DONE, RUNNING, Job


def make_config(tmp_path: Path, **kw) -> Config:
    defaults = dict(
        max_lifetime=3600,
        list_is_public=False,
        admin_group="wheel",
        socket_path=tmp_path / "sock",
        state_dir=tmp_path / "state",
        dev_dir=tmp_path / "dev",
        use_cgroups=False,
        schedule="fifo-strict",
        keep_finished=50,
    )
    defaults.update(kw)
    return Config(**defaults)


def make_daemon(tmp_path: Path, **kw) -> Daemon:
    daemon = Daemon(make_config(tmp_path, **kw), [])
    (daemon.cfg.state_dir / "jobs").mkdir(parents=True, exist_ok=True)
    return daemon


def add_job(
    daemon: Daemon, job_id: int, *, state=DONE, uid=None, output=False
) -> Job:
    """Register a job with the daemon as if it had run. `output` creates the
    state-dir buffer, via the daemon's own accessor so a path change breaks
    the fixture instead of silently writing where nothing looks."""
    uid = os.getuid() if uid is None else uid
    job = Job(
        id=job_id, user=f"u{uid}", uid=uid, gid=os.getgid(),
        argv=["true"], cwd="/", cpu=1, gpu_cores=0, max_mem_gb=None,
        max_time_s=60, exclusive=False, state=state,
        start_time=0.0, end_time=1.0 if state == DONE else None,
    )
    daemon.jobs[job_id] = job
    daemon.job_dir(job_id).mkdir(parents=True, exist_ok=True)
    if output:
        daemon.output_path(job_id).write_bytes(b"captured output\n")
    return job


class TestRunAsUser:
    """Without root the credentials are left alone, but the fork, the error
    channel and the exit status still have to work: that is the machinery
    every user-controlled filesystem operation depends on."""

    def test_success_returns_none(self):
        assert run_as_user(os.getuid(), os.getgid(), "nobody", lambda: None) is None

    def test_failure_reports_the_exception(self):
        def boom():
            raise PermissionError("cannot write to /nope")

        msg = run_as_user(os.getuid(), os.getgid(), "nobody", boom)
        assert msg is not None
        assert "PermissionError" in msg
        assert "/nope" in msg

    def test_child_cannot_mutate_the_parent(self, tmp_path):
        # The work happens after a fork, so anything the callback computes is
        # lost. Callers must not rely on side effects, only on the verdict.
        seen = []
        run_as_user(os.getuid(), os.getgid(), "nobody", lambda: seen.append(1))
        assert seen == []


class TestOutputDestination:
    def test_directory_becomes_output_id_log(self, tmp_path):
        daemon = make_daemon(tmp_path)
        target = tmp_path / "results"
        target.mkdir()
        dest, err = daemon._resolve_output_dest(
            str(target), 7, os.getuid(), os.getgid(), "u"
        )
        assert err is None
        assert dest == str(target / "output.7.log")

    def test_non_directory_is_used_verbatim(self, tmp_path):
        daemon = make_daemon(tmp_path)
        dest, err = daemon._resolve_output_dest(
            str(tmp_path / "run1.log"), 7, os.getuid(), os.getgid(), "u"
        )
        assert err is None
        assert dest == str(tmp_path / "run1.log")

    def test_relative_path_is_rejected(self, tmp_path):
        daemon = make_daemon(tmp_path)
        dest, err = daemon._resolve_output_dest(
            "results", 7, os.getuid(), os.getgid(), "u"
        )
        assert dest is None
        assert "absolute" in err

    def test_missing_parent_is_rejected_at_submit_time(self, tmp_path):
        daemon = make_daemon(tmp_path)
        dest, err = daemon._resolve_output_dest(
            str(tmp_path / "gone" / "out.log"), 7, os.getuid(), os.getgid(), "u"
        )
        assert dest is None
        assert err is not None


class TestRetention:
    def test_keeps_the_most_recent_n_per_user(self, tmp_path):
        daemon = make_daemon(tmp_path, keep_finished=2)
        for job_id in (1, 2, 3, 4):
            add_job(daemon, job_id)

        daemon._trim_finished()

        assert sorted(daemon.jobs) == [3, 4]
        assert not daemon.job_dir(1).exists()
        assert daemon.job_dir(4).exists()

    def test_one_busy_user_does_not_evict_another(self, tmp_path):
        # The whole point of counting per user: alice submitting all evening
        # must not empty bob's list.
        daemon = make_daemon(tmp_path, keep_finished=2)
        for job_id in (1, 2, 3, 4, 5):
            add_job(daemon, job_id, uid=1001)
        add_job(daemon, 6, uid=1002)

        daemon._trim_finished()

        assert sorted(daemon.jobs) == [4, 5, 6]  # alice's last 2, bob's 1

    def test_running_jobs_are_never_trimmed(self, tmp_path):
        daemon = make_daemon(tmp_path, keep_finished=1)
        add_job(daemon, 1, state=RUNNING)
        add_job(daemon, 2, state=RUNNING)
        daemon._trim_finished()
        assert sorted(daemon.jobs) == [1, 2]

    def test_zero_keeps_nothing(self, tmp_path):
        # jobs[:-0] would be the whole list, so this guards the slice.
        daemon = make_daemon(tmp_path, keep_finished=0)
        add_job(daemon, 1)
        daemon._trim_finished()
        assert daemon.jobs == {}

    def test_retention_counts_jobs_not_surviving_output(self, tmp_path):
        # Retention is driven by tracked metadata, so a job whose buffer was
        # already discarded still occupies a slot and still evicts an older
        # one. Job 1 keeps its buffer, job 2 does not; the newer wins anyway.
        daemon = make_daemon(tmp_path, keep_finished=1)
        add_job(daemon, 1, output=True)
        add_job(daemon, 2, output=False)

        daemon._trim_finished()

        assert sorted(daemon.jobs) == [2]
        assert not daemon.job_dir(1).exists()


class TestRetireOutput:
    """The state-dir copy is a streaming buffer for `attach`, not storage:
    it is handed to the user and then dropped."""

    def test_delivered_then_the_buffer_is_dropped(self, tmp_path):
        daemon = make_daemon(tmp_path)
        job = add_job(daemon, 1, output=True)
        job.output_dest = str(tmp_path / "kept.log")

        daemon._retire_output(job)

        assert job.output_error is None
        assert (tmp_path / "kept.log").read_bytes() == b"captured output\n"
        assert not daemon.output_path(1).exists()

    def test_buffer_is_dropped_when_the_user_opted_out(self, tmp_path):
        daemon = make_daemon(tmp_path)
        job = add_job(daemon, 1, output=True)
        job.output_dest = None  # --no-output

        daemon._retire_output(job)

        assert not daemon.output_path(1).exists()

    def test_buffer_survives_a_failed_delivery(self, tmp_path):
        # Otherwise this would be the only copy, and we would be deleting the
        # user's results precisely because we could not hand them over.
        daemon = make_daemon(tmp_path)
        job = add_job(daemon, 1, output=True)
        job.output_dest = str(tmp_path / "gone" / "out.log")

        daemon._retire_output(job)

        assert job.output_error is not None
        assert daemon.output_path(1).read_bytes() == b"captured output\n"
