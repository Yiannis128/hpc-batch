"""cgroup file-writing tests.

These drive CgroupManager against a temporary directory rather than a real
cgroup mount. What matters here is exactly which values land in which file:
`cpuset.mems` is what makes the scheduler's memory accounting binding, so a
mistake in it silently turns a hard guarantee into a hope.
"""

from pathlib import Path

from hpc_batch.cgroup import CgroupManager


def manager(tmp_path: Path, controllers=("cpuset", "memory")) -> CgroupManager:
    cg = CgroupManager(enabled=True)
    cg.base = tmp_path
    cg.controllers = set(controllers)
    return cg


def read(path: Path, name: str) -> str:
    return (path / name).read_text()


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

    def test_returns_none_without_a_delegated_subtree(self, tmp_path):
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
