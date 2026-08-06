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


def manager(tmp_path: Path, controllers=("cpuset", "memory")) -> CgroupManager:
    cg = CgroupManager(enabled=True, root=tmp_path)
    cg.ready = True
    cg.controllers = set(controllers)
    return cg


def fake_cgroup_fs(tmp_path: Path, monkeypatch, controllers="cpuset memory pids") -> Path:
    """A stand-in for /sys/fs/cgroup whose root delegates `controllers`."""
    (tmp_path / "cgroup.controllers").write_text(controllers + "\n")
    monkeypatch.setattr(cgroup_mod, "CGROUP_FS", tmp_path)
    return tmp_path


def read(path: Path, name: str) -> str:
    return (path / name).read_text()


class TestSetup:
    """The job subtree lives outside the daemon's service cgroup, and has to
    survive the daemon being restarted underneath running jobs. Isolation the
    admin asked for and cannot have is refused, never quietly downgraded.
    These build the interface files the kernel would materialise on mkdir."""

    def test_claims_the_configured_subtree(self, tmp_path, monkeypatch):
        fs = fake_cgroup_fs(tmp_path, monkeypatch)
        root = fs / "hpc-batch"
        root.mkdir()
        (root / "cgroup.controllers").write_text("cpuset memory pids\n")

        cg = CgroupManager(root=root)

        cg.setup()

        assert cg.controllers == {"cpuset", "memory"}
        assert cg.create(1, cpus=[0], mems=[0], mem_bytes=None) == root / "job-1"

    def test_a_restart_leaves_live_job_cgroups_alone(self, tmp_path, monkeypatch):
        # The outage this guards against: a daemon coming back must adopt the
        # cgroups of jobs that outlived it, not disturb or recreate them.
        fs = fake_cgroup_fs(tmp_path, monkeypatch)
        root = fs / "hpc-batch"
        root.mkdir()
        (root / "cgroup.controllers").write_text("cpuset memory\n")
        live = root / "job-7"
        live.mkdir()
        (live / "cpuset.cpus").write_text("12-23")

        CgroupManager(root=root).setup()

        assert (live / "cpuset.cpus").read_text() == "12-23"

    def test_creates_nothing_outside_its_own_subtree(self, tmp_path, monkeypatch):
        # Putting jobs in the service cgroup is what made the unit
        # unrestartable once a job outlived it, so setup writes nowhere else.
        fs = fake_cgroup_fs(tmp_path, monkeypatch)
        service = fs / "system.slice" / "hpc-batch.service"
        service.mkdir(parents=True)
        (service / "cgroup.procs").write_text("1234\n")
        root = fs / "hpc-batch"
        root.mkdir()
        (root / "cgroup.controllers").write_text("cpuset memory\n")

        CgroupManager(root=root).setup()

        assert list(service.iterdir()) == [service / "cgroup.procs"]

    def test_a_withheld_controller_is_refused_not_worked_around(self, tmp_path, monkeypatch):
        # Without cpuset a job's memory is not confined to its node. Carrying
        # on would keep every promise except the one that matters, and nothing
        # in the daemon's behaviour would look wrong.
        fs = fake_cgroup_fs(tmp_path, monkeypatch)
        root = fs / "hpc-batch"
        root.mkdir()
        (root / "cgroup.controllers").write_text("memory\n")

        cg = CgroupManager(root=root)

        with pytest.raises(CgroupError) as caught:
            cg.setup()
        assert "cpuset" in str(caught.value)
        assert "Delegate=" in str(caught.value)  # names the usual cause
        assert cg.create(1, cpus=[0], mems=[0], mem_bytes=None) is None

    def test_without_cgroup_v2_it_refuses_to_start(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cgroup_mod, "CGROUP_FS", tmp_path)  # no cgroup.controllers

        with pytest.raises(CgroupError) as caught:
            CgroupManager(root=tmp_path / "hpc-batch").setup()
        assert "--no-cgroups" in str(caught.value)  # every refusal names the opt-out

    def test_no_cgroups_is_the_way_to_ask_for_the_fallback(self, tmp_path):
        cg = CgroupManager(enabled=False, root=tmp_path / "hpc-batch")

        cg.setup()  # does not raise

        assert cg.ready is False
        assert not (tmp_path / "hpc-batch").exists()
        assert cg.create(1, cpus=[0], mems=[0], mem_bytes=None) is None


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

    def test_no_cpuset_controller_means_no_cpuset_files(self, tmp_path):
        cg = manager(tmp_path, controllers=("memory",))
        path = cg.create(11, cpus=[0], mems=[0], mem_bytes=1 << 30)
        assert not (path / "cpuset.cpus").exists()
        assert read(path, "memory.max") == str(1 << 30)

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
