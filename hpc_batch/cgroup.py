"""cgroup v2 management for job isolation.

Jobs go in `/sys/fs/cgroup/hpc-batch/job-<id>`, with cpuset (cpus pinned to
one NUMA node) and memory limits applied. That tree is the daemon's own,
deliberately outside its service cgroup.

Nesting jobs under the service is the obvious layout and Delegate= invites
it, but it makes the unit unrestartable. cgroup v2 forbids processes in a
cgroup that has controllers enabled for its children, so once KillMode=
process leaves jobs behind on stop, systemd cannot put a new daemon into
the service cgroup and the start fails with 219/CGROUP -- for as long as
any job lives. A separate tree is untouched by anything systemd does to
the unit, which is what lets a restarted daemon re-adopt running jobs.

Isolation is all-or-nothing on purpose. If cgroups were asked for and
cannot be delivered -- no cgroup v2, no root, a controller the root never
enabled -- setup raises rather than falling back. Jobs would still run
without them, and nothing would look wrong, which is exactly the problem:
the memory-locality guarantee this tool exists to make would be quietly
gone. `--no-cgroups` is how you ask for the fallback on purpose.
"""

import logging
import os
from pathlib import Path

from .resources import format_id_list

log = logging.getLogger(__name__)

CGROUP_FS = Path("/sys/fs/cgroup")
DEFAULT_ROOT = CGROUP_FS / "hpc-batch"
_WANTED_CONTROLLERS = ("cpuset", "memory")

_NO_CGROUPS = "pass --no-cgroups to run without isolation (development only)"


class CgroupError(Exception):
    """cgroups were asked for and cannot be provided."""


class CgroupManager:
    def __init__(self, enabled: bool = True, root: Path = DEFAULT_ROOT):
        self.enabled = enabled
        self.root = root
        self.ready = False
        self.controllers: set[str] = set()

    def setup(self) -> None:
        """Create and claim the job subtree, or raise CgroupError saying why
        it could not be.

        Idempotent over an existing tree, because that is the restart case:
        the job cgroups of everything still running are sitting in it, and a
        daemon that has just come back has to adopt them, not disturb them.
        """
        if not self.enabled:
            log.warning("cgroups disabled (--no-cgroups): jobs get cpu-affinity "
                        "pinning only, and no memory limit is enforced")
            return
        if not (CGROUP_FS / "cgroup.controllers").exists():
            raise CgroupError(
                f"cgroup v2 is not available at {CGROUP_FS}. This daemon needs a "
                f"unified cgroup hierarchy; {_NO_CGROUPS}"
            )
        try:
            self.root.mkdir(exist_ok=True)
        except OSError as exc:
            raise CgroupError(
                f"cannot create the job cgroup {self.root} ({exc}). The daemon "
                f"must run as root; {_NO_CGROUPS}"
            ) from exc

        available = set((self.root / "cgroup.controllers").read_text().split())
        missing = [c for c in _WANTED_CONTROLLERS if c not in available]
        if missing:
            # The controllers reach us from the kernel root, and systemd only
            # enables one there when a unit asks. Delegate= in our unit is the
            # ask, which is why deleting it takes cpuset away from a tree that
            # is not even inside the unit.
            raise CgroupError(
                f"the {', '.join(missing)} controller(s) are not available in "
                f"{self.root}, so jobs cannot be confined to a NUMA node. Check "
                f"that the unit still has 'Delegate=cpuset memory pids'; "
                f"{_NO_CGROUPS}"
            )
        for ctrl in _WANTED_CONTROLLERS:
            try:
                (self.root / "cgroup.subtree_control").write_text(f"+{ctrl}")
            except OSError as exc:
                raise CgroupError(
                    f"could not enable the {ctrl} controller in {self.root} ({exc})"
                ) from exc
            self.controllers.add(ctrl)

        self.ready = True
        log.info(
            "cgroup subtree %s ready (controllers: %s)",
            self.root, ", ".join(sorted(self.controllers)),
        )

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
