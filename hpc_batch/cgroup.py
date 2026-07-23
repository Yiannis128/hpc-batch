"""cgroup v2 management for job isolation.

The daemon runs as a systemd service with Delegate=, so it owns its own
cgroup subtree. To create child cgroups we first move ourselves into a
`supervisor` leaf (cgroup v2 forbids processes in inner nodes), enable the
controllers we need on the service cgroup, then place every job in its own
`job-<id>` child with cpuset (cpus pinned to one NUMA node) and memory
limits applied.

Everything degrades gracefully: when cgroups are unavailable (not root,
no cgroup v2, missing controllers) the daemon falls back to
sched_setaffinity-only pinning and logs a warning.
"""

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

CGROUP_FS = Path("/sys/fs/cgroup")
_WANTED_CONTROLLERS = ("cpuset", "memory")


def _own_cgroup() -> Path | None:
    """Absolute path of the cgroup this process lives in (v2 only)."""
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                rel = line[3:].strip("/")
                return CGROUP_FS / rel if rel else CGROUP_FS
    except OSError:
        pass
    return None


class CgroupManager:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.base: Path | None = None
        self.controllers: set[str] = set()

    def setup(self) -> bool:
        """Claim our delegated subtree. Returns True when cgroups are usable."""
        if not self.enabled:
            log.info("cgroups disabled by configuration")
            return False
        own = _own_cgroup()
        if own is None or not (CGROUP_FS / "cgroup.controllers").exists():
            log.warning("cgroup v2 not available; jobs will not be isolated")
            return False
        # After a re-exec we are already inside the supervisor leaf.
        base = own.parent if own.name == "supervisor" else own
        if base == CGROUP_FS:
            log.warning("refusing to manage the cgroup root; jobs will not be isolated")
            return False
        try:
            supervisor = base / "supervisor"
            supervisor.mkdir(exist_ok=True)
            # Move every process (normally just us) out of the inner node.
            procs = (base / "cgroup.procs").read_text().split()
            for pid in procs:
                (supervisor / "cgroup.procs").write_text(pid)
            available = set((base / "cgroup.controllers").read_text().split())
            for ctrl in _WANTED_CONTROLLERS:
                if ctrl not in available:
                    log.warning("cgroup controller %r not delegated to us", ctrl)
                    continue
                try:
                    (base / "cgroup.subtree_control").write_text(f"+{ctrl}")
                    self.controllers.add(ctrl)
                except OSError as exc:
                    log.warning("could not enable cgroup controller %r: %s", ctrl, exc)
        except OSError as exc:
            log.warning("cgroup setup failed (%s); jobs will not be isolated", exc)
            return False
        self.base = base
        log.info("cgroup subtree %s ready (controllers: %s)", base, ", ".join(sorted(self.controllers)) or "none")
        return True

    def create(
        self,
        job_id: int,
        cpus: list[int],
        numa_node: int,
        mem_bytes: int | None,
    ) -> Path | None:
        """Create the cgroup for a job; the spawned pid is added by the caller."""
        if self.base is None:
            return None
        path = self.base / f"job-{job_id}"
        path.mkdir(exist_ok=True)
        if "cpuset" in self.controllers:
            (path / "cpuset.cpus").write_text(",".join(str(c) for c in cpus))
            # Confine memory allocation to the same NUMA node as the cpus.
            (path / "cpuset.mems").write_text(str(numa_node))
        if "memory" in self.controllers:
            try:
                # Never let a job swap: swapping would wreck benchmark
                # timings. A job over its budget should OOM, not thrash.
                (path / "memory.swap.max").write_text("0")
            except OSError:
                pass  # kernel built without swap accounting
            if mem_bytes:
                (path / "memory.max").write_text(str(mem_bytes))
                try:
                    # If one process OOMs, take the whole job down with it.
                    (path / "memory.oom.group").write_text("1")
                except OSError:
                    pass
        return path

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
