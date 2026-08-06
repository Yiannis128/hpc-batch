"""cgroup file-writing tests.

These drive CgroupManager against a temporary directory rather than a real
cgroup mount. What matters here is exactly which values land in which file:
`cpuset.mems` is what makes the scheduler's memory accounting binding, so a
mistake in it silently turns a hard guarantee into a hope.
"""

from pathlib import Path

import pytest

from hpc_batch import cgroup as cgroup_mod
from hpc_batch.cgroup import CgroupError, CgroupManager


def manager(tmp_path: Path) -> CgroupManager:
    cg = CgroupManager(enabled=True, root=tmp_path)
    cg.ready = True
    cg.controllers = {"cpuset", "memory"}
    return cg


def fake_cgroup_fs(tmp_path: Path, monkeypatch, controllers="cpuset memory") -> Path:
    """Stand in for /sys/fs/cgroup; returns the job subtree."""
    (tmp_path / "cgroup.controllers").write_text("cpuset memory pids\n")
    monkeypatch.setattr(cgroup_mod, "CGROUP_FS", tmp_path)
    root = tmp_path / "hpc-batch"
    root.mkdir()
    (root / "cgroup.controllers").write_text(controllers + "\n")
    return root


def job_cgroup(root: Path, job_id: int, pids: str = "") -> Path:
    # A job that ended is left as a bare directory: rmdir on real cgroupfs
    # ignores the interface files, but on a temporary one they would block it.
    path = root / f"job-{job_id}"
    path.mkdir()
    if pids:
        (path / "cgroup.procs").write_text(pids)
    return path


def read(path: Path, name: str) -> str:
    return (path / name).read_text()


class TestSetup:
    """The job subtree lives outside the daemon's service cgroup, and has to
    survive the daemon being restarted underneath running jobs. Isolation the
    admin asked for and cannot have is refused, never quietly downgraded."""

    def test_claims_the_job_subtree(self, tmp_path, monkeypatch):
        root = fake_cgroup_fs(tmp_path, monkeypatch)

        cg = CgroupManager()

        cg.setup()

        assert cg.controllers == {"cpuset", "memory"}
        assert cg.create(1, cpus=[0], mems=[0], mem_bytes=None) == root / "job-1"

    def test_claiming_an_existing_subtree_does_not_disturb_it(self, tmp_path, monkeypatch):
        # A restarted daemon claims a tree it is already running jobs in.
        root = fake_cgroup_fs(tmp_path, monkeypatch)
        live = job_cgroup(root, 7, pids="4242\n")
        (live / "cpuset.cpus").write_text("12-23")

        CgroupManager().setup()

        assert read(live, "cpuset.cpus") == "12-23"

    def test_a_withheld_controller_is_refused_not_worked_around(self, tmp_path, monkeypatch):
        # Without cpuset a job's memory is not confined to its node. Carrying
        # on would keep every promise except the one that matters, and nothing
        # in the daemon's behaviour would look wrong.
        fake_cgroup_fs(tmp_path, monkeypatch, controllers="memory")

        cg = CgroupManager()

        with pytest.raises(CgroupError) as caught:
            cg.setup()
        assert "cpuset" in str(caught.value)
        assert "Delegate=" in str(caught.value)  # names the usual cause
        assert cg.create(1, cpus=[0], mems=[0], mem_bytes=None) is None

    def test_without_cgroup_v2_it_refuses_to_start(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cgroup_mod, "CGROUP_FS", tmp_path)  # no cgroup.controllers

        with pytest.raises(CgroupError) as caught:
            CgroupManager().setup()
        assert "--no-cgroups" in str(caught.value)  # every refusal names the opt-out

    def test_no_cgroups_is_the_way_to_ask_for_the_fallback(self, tmp_path):
        cg = CgroupManager(enabled=False, root=tmp_path / "hpc-batch")

        cg.setup()  # does not raise

        assert cg.ready is False
        assert not (tmp_path / "hpc-batch").exists()
        assert cg.create(1, cpus=[0], mems=[0], mem_bytes=None) is None


class TestPrune:
    """Job cgroups outlive the daemon that made them, and nothing else reaps
    them now that the tree sits outside the service cgroup."""

    def test_reaps_the_cgroups_of_jobs_that_ended(self, tmp_path):
        dead = job_cgroup(tmp_path, 3)

        manager(tmp_path).prune()

        assert not dead.exists()

    def test_leaves_a_job_that_is_still_running_alone(self, tmp_path):
        live = job_cgroup(tmp_path, 7, pids="4242\n")

        manager(tmp_path).prune()

        assert live.exists()

    def test_does_nothing_until_a_subtree_has_been_claimed(self, tmp_path):
        dead = job_cgroup(tmp_path, 3)

        CgroupManager(root=tmp_path).prune()  # setup() never ran

        assert dead.exists()


class TestCreate:
    def test_pins_cpus_and_the_nodes_the_memory_was_charged_to(self, tmp_path):
        cg = manager(tmp_path)
        path = cg.create(7, cpus=[0, 1, 2], mems=[0], mem_bytes=1 << 30)
        assert path == tmp_path / "job-7"
        assert read(path, "cpuset.cpus") == "0,1,2"
        assert read(path, "cpuset.mems") == "0"
        assert read(path, "memory.max") == str(1 << 30)

    def test_writes_every_node_of_a_budget_that_spans(self, tmp_path):
        cg = manager(tmp_path)
        path = cg.create(8, cpus=[0], mems=[0, 1], mem_bytes=1 << 31)
        # The job may allocate on both nodes precisely because both were
        # charged for it; anything else and the books would be fiction.
        assert read(path, "cpuset.mems") == "0,1"

    def test_swap_is_always_off(self, tmp_path):
        cg = manager(tmp_path)
        path = cg.create(9, cpus=[0], mems=[0], mem_bytes=None)
        assert read(path, "memory.swap.max") == "0"

    def test_oom_group_is_set_even_without_a_limit(self, tmp_path):
        # A job left half-dead is worse than a clean kill whether or not it
        # named a budget, so this is not conditional on memory.max.
        cg = manager(tmp_path)
        path = cg.create(10, cpus=[0], mems=[0], mem_bytes=None)
        assert read(path, "memory.oom.group") == "1"
        assert not (path / "memory.max").exists()

    def test_returns_none_before_setup_has_claimed_a_subtree(self, tmp_path):
        cg = CgroupManager(enabled=True)
        assert cg.create(12, cpus=[0], mems=[0], mem_bytes=None) is None


class TestOomKilled:
    def test_reports_a_kill(self, tmp_path):
        path = tmp_path / "job-1"
        path.mkdir()
        (path / "memory.events").write_text("low 0\nhigh 0\nmax 12\noom 3\noom_kill 1\n")
        assert manager(tmp_path).oom_killed(path) is True

    def test_hitting_the_limit_without_a_kill_is_not_a_kill(self, tmp_path):
        # A job can bump against memory.max repeatedly and reclaim its way
        # out; only oom_kill means the kernel actually killed something.
        path = tmp_path / "job-2"
        path.mkdir()
        (path / "memory.events").write_text("max 900\noom 0\noom_kill 0\n")
        assert manager(tmp_path).oom_killed(path) is False

    def test_missing_or_unreadable_file_is_not_a_kill(self, tmp_path):
        path = tmp_path / "job-3"
        path.mkdir()
        assert manager(tmp_path).oom_killed(path) is False
        (path / "memory.events").write_text("oom_kill nonsense\n")
        assert manager(tmp_path).oom_killed(path) is False
