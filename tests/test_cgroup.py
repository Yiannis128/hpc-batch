"""cgroup file-writing tests.

These drive CgroupManager against a temporary directory rather than a real
cgroup mount. What matters here is exactly which values land in which file:
`cpuset.mems` is what makes the scheduler's memory accounting binding, so a
mistake in it silently turns a hard guarantee into a hope.
"""

from pathlib import Path

from hpc_batch import cgroup as cgroup_mod
from hpc_batch.cgroup import CgroupManager


def manager(tmp_path: Path, controllers=("cpuset", "memory")) -> CgroupManager:
    cg = CgroupManager(enabled=True, root=tmp_path)
    cg.ready = True
    cg.controllers = set(controllers)
    return cg


def fake_cgroup_fs(tmp_path: Path, monkeypatch, controllers="cpuset memory") -> Path:
    """Stand in for /sys/fs/cgroup, building the interface files the kernel
    would materialise. Returns the job subtree, which has `controllers` to
    hand out to its children."""
    (tmp_path / "cgroup.controllers").write_text("cpuset memory pids\n")
    monkeypatch.setattr(cgroup_mod, "CGROUP_FS", tmp_path)
    root = tmp_path / "hpc-batch"
    root.mkdir()
    (root / "cgroup.controllers").write_text(controllers + "\n")
    return root


def job_cgroup(root: Path, job_id: int, pids: str = "") -> Path:
    """A job cgroup left behind by a previous daemon; `pids` is its
    cgroup.procs. A job that ended is a bare directory: rmdir on real cgroupfs
    ignores the interface files, but here they would block it."""
    path = root / f"job-{job_id}"
    path.mkdir()
    if pids:
        (path / "cgroup.procs").write_text(pids)
    return path


def read(path: Path, name: str) -> str:
    return (path / name).read_text()


class TestSetup:
    """The job subtree lives outside the daemon's service cgroup, and has to
    survive the daemon being restarted underneath running jobs."""

    def test_claims_the_job_subtree(self, tmp_path, monkeypatch, caplog):
        root = fake_cgroup_fs(tmp_path, monkeypatch)

        cg = CgroupManager()

        assert cg.setup() is True
        assert cg.controllers == {"cpuset", "memory"}
        assert cg.create(1, cpus=[0], mems=[0], mem_bytes=None) == root / "job-1"
        assert "NO NUMA ISOLATION" not in caplog.text

    def test_a_restart_leaves_live_job_cgroups_alone(self, tmp_path, monkeypatch):
        # The outage this guards against: a daemon coming back must adopt the
        # cgroups of jobs that outlived it, not disturb or recreate them.
        root = fake_cgroup_fs(tmp_path, monkeypatch)
        live = job_cgroup(root, 7, pids="4242\n")
        (live / "cpuset.cpus").write_text("12-23")

        assert CgroupManager().setup() is True

        assert read(live, "cpuset.cpus") == "12-23"

    def test_a_restart_reaps_the_cgroups_of_jobs_that_ended(self, tmp_path, monkeypatch):
        # Nothing else does: the tree outlives the unit by design, and the
        # busy-cgroup retry list is in memory only.
        root = fake_cgroup_fs(tmp_path, monkeypatch)
        dead = job_cgroup(root, 3)

        assert CgroupManager().setup() is True

        assert not dead.exists()

    def test_creates_nothing_outside_its_own_subtree(self, tmp_path, monkeypatch):
        # Putting jobs in the service cgroup is what made the unit
        # unrestartable once a job outlived it, so setup writes nowhere else.
        fake_cgroup_fs(tmp_path, monkeypatch)
        service = tmp_path / "system.slice" / "hpc-batch.service"
        service.mkdir(parents=True)
        (service / "cgroup.procs").write_text("1234\n")

        CgroupManager().setup()

        assert list(service.iterdir()) == [service / "cgroup.procs"]

    def test_a_controller_the_kernel_root_withholds_is_skipped(self, tmp_path, monkeypatch, caplog):
        fake_cgroup_fs(tmp_path, monkeypatch, controllers="memory")

        cg = CgroupManager()

        assert cg.setup() is True
        assert cg.controllers == {"memory"}
        assert "NO NUMA ISOLATION" in caplog.text

    def test_without_cgroup_v2_jobs_fall_back_to_affinity(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cgroup_mod, "CGROUP_FS", tmp_path)  # no cgroup.controllers
        cg = CgroupManager()

        assert cg.setup() is False
        assert cg.create(1, cpus=[0], mems=[0], mem_bytes=None) is None

    def test_disabled_by_configuration(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cgroup_mod, "CGROUP_FS", tmp_path)
        cg = CgroupManager(enabled=False)

        assert cg.setup() is False
        assert not (tmp_path / "hpc-batch").exists()


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
