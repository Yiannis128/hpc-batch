import os
import time
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
        keep_finished=604800,
    )
    defaults.update(kw)
    return Config(**defaults)


def make_daemon(tmp_path: Path, **kw) -> Daemon:
    daemon = Daemon(make_config(tmp_path, **kw), [])
    (daemon.cfg.state_dir / "jobs").mkdir(parents=True, exist_ok=True)
    return daemon


def add_job(daemon: Daemon, job_id: int, *, state=DONE, end_time=None, size=0) -> Job:
    job = Job(
        id=job_id, user="u", uid=os.getuid(), gid=os.getgid(),
        argv=["true"], cwd="/", cpu=1, gpu_cores=0, max_mem_gb=None,
        max_time_s=60, exclusive=False, state=state,
        start_time=0.0, end_time=end_time,
    )
    daemon.jobs[job_id] = job
    d = daemon.job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    if size:
        (d / "output").write_bytes(b"x" * size)
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
    def test_finished_jobs_expire_by_age(self, tmp_path):
        daemon = make_daemon(tmp_path, keep_finished=100)
        now = time.time()
        fresh = add_job(daemon, 1, end_time=now - 10)
        stale = add_job(daemon, 2, end_time=now - 1000)

        daemon._trim_finished()

        assert 1 in daemon.jobs
        assert 2 not in daemon.jobs
        assert daemon.job_dir(1).exists()
        assert not daemon.job_dir(2).exists()

    def test_running_jobs_are_never_trimmed(self, tmp_path):
        daemon = make_daemon(tmp_path, keep_finished=1)
        add_job(daemon, 1, state=RUNNING, end_time=None)
        daemon._trim_finished()
        assert 1 in daemon.jobs

    def test_expiry_is_independent_of_whether_output_exists(self, tmp_path):
        # --finished is driven by tracked job metadata, not by leftover files,
        # so a job whose output was discarded is still listable.
        daemon = make_daemon(tmp_path, keep_finished=100)
        job = add_job(daemon, 1, end_time=time.time(), size=64)
        daemon.output_path(1).unlink()

        daemon._trim_finished()

        assert 1 in daemon.jobs
        assert daemon.jobs[1] is job


class TestDiscardOutput:
    """The state-dir copy is a streaming buffer for `attach`, not storage."""

    def test_discarded_after_successful_delivery(self, tmp_path):
        daemon = make_daemon(tmp_path)
        job = add_job(daemon, 1, end_time=time.time(), size=64)
        job.output_dest = str(tmp_path / "kept.log")
        job.output_error = None

        daemon._discard_output(job)

        assert not daemon.output_path(1).exists()

    def test_discarded_when_the_user_opted_out(self, tmp_path):
        daemon = make_daemon(tmp_path)
        job = add_job(daemon, 1, end_time=time.time(), size=64)
        job.output_dest = None  # --no-output

        daemon._discard_output(job)

        assert not daemon.output_path(1).exists()

    def test_kept_when_delivery_failed(self, tmp_path):
        # Otherwise this would be the only copy and we would be deleting the
        # user's results because we could not hand them over.
        daemon = make_daemon(tmp_path)
        job = add_job(daemon, 1, end_time=time.time(), size=64)
        job.output_dest = str(tmp_path / "gone" / "out.log")
        job.output_error = "FileNotFoundError: ..."

        daemon._discard_output(job)

        assert daemon.output_path(1).exists()
