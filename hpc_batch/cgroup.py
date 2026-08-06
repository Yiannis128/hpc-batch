"""cgroup v2 management for job isolation.

Jobs go in `/sys/fs/cgroup/hpc-batch/job-<id>`, with cpuset (cpus pinned to
one NUMA node) and memory limits applied. That tree is deliberately outside
the daemon's service cgroup: jobs KillMode=process leaves behind would keep
that cgroup populated, and systemd could not recreate it on restart
(219/CGROUP) for as long as any of them lived.

Everything degrades gracefully: when cgroups are unavailable (not root,
no cgroup v2, missing controllers) the daemon falls back to
sched_setaffinity-only pinning and logs a warning.
"""

import logging
import os
from pathlib import Path

from .resources import format_id_list

log = logging.getLogger(__name__)

CGROUP_FS = Path("/sys/fs/cgroup")
_WANTED_CONTROLLERS = ("cpuset", "memory")


class CgroupManager:
    def __init__(self, enabled: bool = True, root: Path | None = None):
        self.enabled = enabled
        self.root = root or CGROUP_FS / "hpc-batch"
        self.ready = False
        self.controllers: set[str] = set()

    def setup(self) -> bool:
        """Create or reclaim the job subtree, leaving the cgroups of jobs that
        outlived the previous daemon alone. Returns True when cgroups are
        usable."""
        if not self.enabled:
            log.info("cgroups disabled by configuration")
            return False
        if not (CGROUP_FS / "cgroup.controllers").exists():
            log.warning("cgroup v2 not available; jobs will not be isolated")
            return False
        try:
            self.root.mkdir(exist_ok=True)
            available = set((self.root / "cgroup.controllers").read_text().split())
            for ctrl in _WANTED_CONTROLLERS:
                if ctrl not in available:
                    log.warning("cgroup controller %r not enabled for %s", ctrl, self.root)
                    continue
                try:
                    (self.root / "cgroup.subtree_control").write_text(f"+{ctrl}")
                    self.controllers.add(ctrl)
                except OSError as exc:
                    log.warning("could not enable cgroup controller %r: %s", ctrl, exc)
        except OSError as exc:
            log.warning("cgroup setup failed (%s); jobs will not be isolated", exc)
            return False
        self.ready = True
        self._prune()
        log.info(
            "cgroup subtree %s ready (controllers: %s)",
            self.root, ", ".join(sorted(self.controllers)) or "none",
        )
        if "cpuset" not in self.controllers:
            log.warning(
                "NO NUMA ISOLATION: jobs fall back to cpu-affinity pinning and "
                "their memory is not confined to a node. Check that the unit "
                "still has 'Delegate=cpuset memory pids'."
            )
        return True

    def _prune(self) -> None:
        """Remove job cgroups left behind by a previous daemon; nothing else
        reaps them outside the service cgroup, and the list of cgroups still
        busy at removal is in-memory only. A populated one is a job about to
        be adopted, and rmdir refuses it in any case, so a lost state file
        cannot turn every running job into an orphan."""
        removed = 0
        for path in self.root.glob("job-*"):
            try:
                pids = (path / "cgroup.procs").read_text().strip()
            except OSError:
                pids = ""  # let rmdir be the judge
            if pids:
                continue
            try:
                path.rmdir()
            except OSError:
                continue
            removed += 1
        if removed:
            log.info("removed %d job cgroup(s) left by a previous daemon", removed)

    def create(
        self,
        job_id: int,
        cpus: list[int],
        mems: list[int],
        mem_bytes: int | None,
    ) -> Path | None:
        """Create the cgroup for a job; the spawned pid is added by the caller.

        ``mems`` is the set of NUMA nodes the scheduler charged this job's
        memory to — usually just the node its cpus are on. Writing exactly
        those nodes is what makes the accounting binding: the job cannot
        allocate memory on a node nobody budgeted for it.
        """
        if not self.ready:
            return None
        path = self.root / f"job-{job_id}"
        path.mkdir(exist_ok=True)
        if "cpuset" in self.controllers:
            (path / "cpuset.cpus").write_text(format_id_list(cpus))
            (path / "cpuset.mems").write_text(format_id_list(mems))
        if "memory" in self.controllers:
            try:
                # Never let a job swap: swapping would wreck benchmark
                # timings. A job over its budget should OOM, not thrash.
                (path / "memory.swap.max").write_text("0")
            except OSError:
                pass  # kernel built without swap accounting
            try:
                # If one process OOMs, take the whole job down with it. Set
                # unconditionally: a job left half-dead is a worse outcome
                # than a clean kill whether or not it named a budget.
                (path / "memory.oom.group").write_text("1")
            except OSError:
                pass
            if mem_bytes:
                (path / "memory.max").write_text(str(mem_bytes))
        return path

    def oom_killed(self, path: Path) -> bool:
        """Did the kernel OOM-kill anything in this cgroup? Read before the
        cgroup is removed, so a job killed for exceeding its memory budget can
        say so instead of just reporting SIGKILL."""
        try:
            for line in (path / "memory.events").read_text().splitlines():
                key, _, value = line.partition(" ")
                if key == "oom_kill":
                    return int(value) > 0
        except (OSError, ValueError):
            pass
        return False

    def confine_current(self, cgroup: Path | None, cpus: list[int]) -> None:
        """Confine the calling process to its job's resources. Runs in the
        child between fork and exec: enter the job cgroup, or fall back to
        plain cpu-affinity pinning when cgroups are unavailable."""
        if cgroup is not None:
            with open(cgroup / "cgroup.procs", "w") as f:
                f.write(str(os.getpid()))
        else:
            os.sched_setaffinity(0, cpus)

    def kill(self, path: Path) -> None:
        """SIGKILL every process in the cgroup."""
        try:
            (path / "cgroup.kill").write_text("1")
        except OSError:
            pass

    def try_remove(self, path: Path) -> bool:
        """Kill stragglers and try to remove the job cgroup. Returns False
        while the cgroup is still busy; callers retry later rather than
        blocking on it."""
        if not path.exists():
            return True
        self.kill(path)
        try:
            path.rmdir()
            return True
        except OSError:
            return False
