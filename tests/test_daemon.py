import os
import pwd
from pathlib import Path

import pytest

from hpc_batch import daemon as daemon_mod
from hpc_batch.daemon import (
    MAX_ENV_BYTES,
    Config,
    Daemon,
    StartupError,
    _env_problem,
    run_as_user,
)
from hpc_batch.jobs import DONE, QUEUED, RUNNING, Job
from hpc_batch.resources import ResourcePool
from test_resources import held


def make_config(tmp_path: Path, **kw) -> Config:
    defaults = dict(
        max_lifetime=3600,
        list_is_public=False,
        admin_group="wheel",
        socket_path=tmp_path / "sock",
        state_dir=tmp_path / "state",
        dev_dir=tmp_path / "dev",
        use_cgroups=False,
        use_dev_dir=False,
        schedule="fifo-strict",
        keep_finished=50,
        reserve_cpu=2,
        reserve_mem=2.0,
        min_job_mem=2.0,
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


class TestStartupRefusals:
    """Isolation that was asked for and cannot be delivered stops the daemon.
    A warning gets read once, at install time, by someone who is not looking
    for it; the promise then quietly does not hold for months."""

    def test_a_missing_admin_group_is_fatal(self, tmp_path):
        daemon = make_daemon(tmp_path, admin_group="no-such-group-here")

        with pytest.raises(StartupError) as caught:
            daemon._resolve_admin_gid()
        assert "no-such-group-here" in str(caught.value)

    def test_it_names_a_group_that_does_exist(self, tmp_path, monkeypatch):
        # The usual cause is `wheel` on a Debian box, where the answer is
        # `sudo`. Saying which groups exist turns a puzzle into a one-word fix.
        monkeypatch.setattr(daemon_mod, "group_exists", lambda g: g == "sudo")
        daemon = make_daemon(tmp_path, admin_group="wheel-but-not-on-this-box")

        with pytest.raises(StartupError) as caught:
            daemon._resolve_admin_gid()
        assert "sudo" in str(caught.value)

    def test_an_unusable_dev_dir_is_fatal(self, tmp_path):
        blocked = tmp_path / "not-a-dir"
        blocked.write_text("")
        daemon = make_daemon(tmp_path, dev_dir=blocked, use_dev_dir=True)

        with pytest.raises(StartupError) as caught:
            daemon._setup_dirs()
        assert "--no-dev-dir" in str(caught.value)  # refusals name their opt-out

    def test_no_dev_dir_is_the_way_to_ask_for_it(self, tmp_path):
        blocked = tmp_path / "not-a-dir"
        blocked.write_text("")
        daemon = make_daemon(tmp_path, dev_dir=blocked, use_dev_dir=False)

        daemon._setup_dirs()  # does not raise

        assert daemon._dev_ok is False


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

    def test_a_job_killed_off_the_queue_delivers_nothing(self, tmp_path):
        # Reporting a delivery failure here left every cancelled job warning
        # about output that never existed, for the rest of its retention.
        daemon = make_daemon(tmp_path)
        job = add_job(daemon, 1, output=False)
        job.start_time = None
        job.output_dest = str(tmp_path / "kept.log")

        daemon._retire_output(job)

        assert job.output_error is None
        assert not (tmp_path / "kept.log").exists()

    def test_a_job_that_failed_to_start_still_delivers_its_reason(self, tmp_path):
        # Also never started, but _try_start wrote why into the buffer.
        daemon = make_daemon(tmp_path)
        job = add_job(daemon, 1, output=True)
        job.start_time = None
        job.output_dest = str(tmp_path / "kept.log")

        daemon._retire_output(job)

        assert job.output_error is None
        assert (tmp_path / "kept.log").read_bytes() == b"captured output\n"

    def test_buffer_survives_a_failed_delivery(self, tmp_path):
        # Otherwise this would be the only copy, and we would be deleting the
        # user's results precisely because we could not hand them over.
        daemon = make_daemon(tmp_path)
        job = add_job(daemon, 1, output=True)
        job.output_dest = str(tmp_path / "gone" / "out.log")

        daemon._retire_output(job)

        assert job.output_error is not None
        assert daemon.output_path(1).read_bytes() == b"captured output\n"


def pooled_daemon(tmp_path, **kw) -> Daemon:
    """A daemon with a known two-node machine instead of the real one."""
    daemon = make_daemon(tmp_path, **kw)
    daemon.pool = ResourcePool(
        node_cpus={0: [0, 1, 2, 3], 1: [4, 5, 6, 7]},
        gpu_ids=[],
        node_mem_gb={0: 32.0, 1: 32.0},
    )
    return daemon


class TestJobEnvironment:
    def env_for(self, tmp_path, env=None, job_id=1, gpus=(), gpu_ids=()) -> dict:
        daemon = make_daemon(tmp_path)
        daemon.pool = ResourcePool(
            node_cpus={0: [0, 1]}, gpu_ids=list(gpu_ids), node_mem_gb={0: 8.0}
        )
        job = add_job(daemon, job_id)
        job.env = env or {}
        return daemon._job_env(
            job, pwd.getpwuid(os.getuid()), held([0], gpus=gpus, mem=1.0)
        )

    def test_a_clean_environment_by_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SECCOM_A2_BATCH_SIZE", "3")

        env = self.env_for(tmp_path)

        assert env["PATH"] == "/usr/local/bin:/usr/bin:/bin"
        assert "SECCOM_A2_BATCH_SIZE" not in env  # the daemon's own env is not the job's

    def test_forwards_what_the_submitter_sent(self, tmp_path):
        env = self.env_for(tmp_path, {"SECCOM_A2_BATCH_SIZE": "3", "PATH": "/data/venv/bin"})

        assert env["SECCOM_A2_BATCH_SIZE"] == "3"
        assert env["PATH"] == "/data/venv/bin"  # theirs beats the default

    def test_a_forwarded_cuda_visible_devices_cannot_widen_the_allocation(self, tmp_path):
        # If a variable carried over from the submitting shell won, any user
        # could reach every card on the machine by exporting it before
        # submitting.
        env = self.env_for(
            tmp_path, {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}, gpus=[2], gpu_ids=[0, 1, 2, 3]
        )

        assert env["CUDA_VISIBLE_DEVICES"] == "2"

    def test_identity_and_job_id_are_not_overridable(self, tmp_path):
        env = self.env_for(
            tmp_path, {"HOME": "/root", "USER": "root", "HPC_BATCH_JOB_ID": "999"}, job_id=7
        )

        pw = pwd.getpwuid(os.getuid())
        assert env["HOME"] == pw.pw_dir
        assert env["USER"] == pw.pw_name
        assert env["HPC_BATCH_JOB_ID"] == "7"


class TestEnvProblem:
    def test_accepts_a_string_mapping(self):
        assert _env_problem({"A": "1"}) is None

    def test_rejects_non_string_values(self):
        # json.dumps would happily send numbers or nested objects, and execve
        # would reject them much later, when the job is already queued.
        assert _env_problem({"A": 1}) is not None

    def test_rejects_an_oversized_environment(self):
        assert "too large" in _env_problem({"BIG": "x" * (MAX_ENV_BYTES + 1)})


class TestForwardedEnvIsNotKept:
    """A forwarded environment is stored only so a queued job survives a
    daemon restart, so it lives no longer than the wait to start."""

    def test_dropped_once_the_job_has_been_exec_d(self, tmp_path):
        daemon = pooled_daemon(tmp_path)
        job = add_job(daemon, 1, state=QUEUED)
        job.env = {"HF_TOKEN": "secret"}

        daemon._start_job(job, held([0], mem=1.0))
        daemon._procs[1].wait()
        daemon._persist(force=True)

        assert job.env == {}
        assert "secret" not in daemon.state_file.read_text()

    def test_never_reaches_the_published_info_file(self, tmp_path):
        daemon = make_daemon(tmp_path)
        job = add_job(daemon, 1, state=QUEUED)
        job.env = {"HF_TOKEN": "secret"}

        daemon._write_info(job)

        assert "secret" not in (daemon.job_dir(1) / "info.json").read_text()


class TestStateFilePermissions:
    def test_state_file_is_root_only(self, tmp_path):
        # It holds every user's jobs, and with --env their secrets too.
        daemon = make_daemon(tmp_path)
        add_job(daemon, 1)

        daemon._persist(force=True)

        assert daemon.state_file.stat().st_mode & 0o777 == 0o600


class TestDefaultMemoryBudget:
    def test_scales_with_the_cores_asked_for(self, tmp_path):
        daemon = pooled_daemon(tmp_path)
        # 4 cpus / 32 GiB a node, so a core is worth 8 GiB.
        assert daemon._default_mem_gb(1, exclusive=False) == 8.0
        assert daemon._default_mem_gb(2, exclusive=False) == 16.0
        assert daemon._default_mem_gb(4, exclusive=False) == 32.0

    def test_floor_protects_small_requests(self, tmp_path):
        # A core-dense node: one core's share is 0.5 GiB, which nothing real
        # runs in. The admin floor is what keeps casual jobs working.
        daemon = make_daemon(tmp_path, min_job_mem=2.0)
        daemon.pool = ResourcePool(
            node_cpus={0: list(range(64))}, gpu_ids=[], node_mem_gb={0: 32.0}
        )
        assert daemon._default_mem_gb(1, exclusive=False) == 2.0

    def test_never_defaults_above_what_a_node_can_give(self, tmp_path):
        daemon = make_daemon(tmp_path, min_job_mem=999.0)
        daemon.pool = ResourcePool(
            node_cpus={0: [0, 1]}, gpu_ids=[], node_mem_gb={0: 8.0}
        )
        # An absurd floor must not produce a budget that can never be placed.
        assert daemon._default_mem_gb(1, exclusive=False) == 8.0

    def test_exclusive_gets_the_whole_machine(self, tmp_path):
        daemon = pooled_daemon(tmp_path)
        assert daemon._default_mem_gb(8, exclusive=True) == 64.0

    def test_none_when_memory_is_untracked(self, tmp_path):
        daemon = make_daemon(tmp_path)
        daemon.pool = ResourcePool(
            node_cpus={0: [0, 1]}, gpu_ids=[], node_mem_gb={0: 0.0}
        )
        assert daemon._default_mem_gb(1, exclusive=False) is None


class TestSubmit:
    def submit(self, daemon, **kw) -> dict:
        req = {"cmd": "new", "argv": ["true"], "cwd": "/"}
        req.update(kw)
        return daemon._submit(req, os.getuid())

    def test_unstated_budget_is_filled_in_and_reported(self, tmp_path):
        daemon = pooled_daemon(tmp_path)
        resp = self.submit(daemon, cpu=2)
        assert resp["ok"] and resp["mem_defaulted"] is True
        assert resp["max_mem_gb"] == 16.0
        # The job carries the number, so the scheduler can account for it.
        assert daemon.jobs[resp["id"]].max_mem_gb == 16.0

    def test_explicit_budget_is_kept_and_marked_as_such(self, tmp_path):
        daemon = pooled_daemon(tmp_path)
        resp = self.submit(daemon, cpu=2, max_mem_gb=4.0)
        assert resp["max_mem_gb"] == 4.0 and resp["mem_defaulted"] is False

    def test_exclusive_defaults_to_every_core(self, tmp_path):
        daemon = pooled_daemon(tmp_path)
        resp = self.submit(daemon, exclusive=True)
        assert resp["cpu"] == 8
        assert resp["max_mem_gb"] == 64.0

    def test_budget_may_span_nodes_by_default(self, tmp_path):
        daemon = pooled_daemon(tmp_path)
        assert self.submit(daemon, cpu=1, max_mem_gb=48.0)["ok"]

    def test_numa_local_rejects_a_budget_no_node_can_hold(self, tmp_path):
        daemon = pooled_daemon(tmp_path)
        resp = self.submit(daemon, cpu=1, max_mem_gb=48.0, numa_local=True)
        assert not resp["ok"]
        assert "--numa-local" in resp["error"] and "--exclusive" in resp["error"]

    def test_numa_local_queues_when_merely_unavailable(self, tmp_path):
        daemon = pooled_daemon(tmp_path)
        # Fits a node in principle, so it must queue rather than be refused.
        resp = self.submit(daemon, cpu=1, max_mem_gb=32.0, numa_local=True)
        assert resp["ok"]
        assert daemon.jobs[resp["id"]].numa_local is True


class TestList:
    def test_response_carries_the_clock_the_rows_were_built_against(self, tmp_path):
        daemon = make_daemon(tmp_path)
        add_job(daemon, 1, state=RUNNING)  # start_time=0.0, no end_time yet
        resp = daemon._h_list({}, os.getuid())
        row = resp["jobs"][0]
        # The client renders the START column from start_time against this
        # "now". Sending it means both come from the daemon's clock, so the
        # start time and the uptime beside it can never disagree.
        assert row["start_time"] == 0.0
        assert resp["now"] == row["start_time"] + row["uptime_s"]
