"""hpc-batchd: the hpc-batch dispatch daemon.

Runs as a systemd service (root), accepts job submissions from the
`dispatch` client over a unix socket authenticated with SO_PEERCRED, and
starts them in FIFO order (subject to the configured scheduling policy, see
`scheduling.py`) inside per-job cgroups whose cpus are pinned to a single
NUMA node.

Hot reload: `systemctl reload hpc-batch` sends SIGHUP; the daemon persists
its state and re-execs itself in place. Running jobs keep their pids (they
stay children of the daemon across exec) and are re-adopted on startup, so
a reload never kills jobs.
"""

import argparse
import asyncio
import contextlib
import grp
import json
import logging
import math
import os
import pwd
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path

from . import __version__
from .cgroup import CgroupError, CgroupManager
from .jobs import DONE, QUEUED, RUNNING, Job
from .protocol import (
    DEFAULT_SOCKET,
    ERROR,
    KILLED,
    MAX_LINE,
    OOM,
    TIMEOUT,
    err,
    read_json,
    send_json,
)
from .resources import (
    Allocation,
    GpuTopology,
    Request,
    ResourcePool,
    apply_reserve,
    discover_gpu_topology,
    discover_gpus,
    discover_node_memory_gb,
    discover_numa_nodes,
    format_id_list,
    total_memory_gb,
)
from .scheduling import FIFO_STRICT, MODES, Reservation, plan
from .util import ADMIN_GROUPS, duration_arg, format_duration, format_gb, group_exists

log = logging.getLogger("hpc-batchd")

TICK_S = 1.0
KILL_GRACE_S = 10.0
ATTACH_POLL_S = 0.3
# A queued job's --env is held in the state file until it starts, so it has to
# be bounded. Kept under MAX_LINE so an oversized one is rejected with a reason
# instead of killing the connection.
MAX_ENV_BYTES = 256 * 1024
# EX_CONFIG. Paired with RestartPreventExitStatus= in the unit: a daemon that
# is misconfigured will be just as misconfigured two seconds later, and a
# restart loop buries the one log line that says what is wrong.
EX_CONFIG = 78


class StartupError(Exception):
    """A configuration problem that must be fixed before the daemon can run."""


@dataclass
class Config:
    max_lifetime: int
    list_is_public: bool
    admin_group: str
    socket_path: Path
    state_dir: Path
    dev_dir: Path
    use_cgroups: bool
    use_dev_dir: bool
    schedule: str
    keep_finished: int
    reserve_cpu: int
    reserve_mem: float
    min_job_mem: float


def drop_privileges(uid: int, gid: int, user: str) -> None:
    """Become `user`, irreversibly. Call only between fork and exec/work.

    The order is the whole point: supplementary groups have to be set while
    we still have the privilege to set them, and setgid has to precede
    setuid for the same reason. Getting it wrong leaves the process holding
    credentials it should have shed, so both the job-spawn path and the
    output-delivery path share this one definition.
    """
    if os.getuid() != 0:
        return  # development mode: nothing to drop
    os.initgroups(user, gid)
    os.setgid(gid)
    os.setuid(uid)


def run_as_user(uid: int, gid: int, user: str, fn) -> str | None:
    """Run ``fn`` in a forked child holding the given user's credentials.

    Every filesystem operation on a path the *user* chose goes through here.
    The daemon is root, so doing that work in-process would let a user borrow
    root's privileges to reach a file they could not touch themselves (the
    classic case being a symlink planted at the destination). Dropping to
    their uid first means ordinary permission checks apply.

    Returns None on success, or a message describing the failure.
    """
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        os.close(read_fd)
        msg = ""
        try:
            drop_privileges(uid, gid, user)
            fn()
        except BaseException as exc:  # noqa: BLE001 - reported to the parent
            msg = f"{type(exc).__name__}: {exc}"[:400]
        finally:
            with contextlib.suppress(OSError):
                os.write(write_fd, msg.encode())
                os.close(write_fd)
            os._exit(1 if msg else 0)

    os.close(write_fd)
    chunks = []
    with contextlib.suppress(OSError):
        while chunk := os.read(read_fd, 4096):
            chunks.append(chunk)
    os.close(read_fd)
    # waitpid on our own pid only; it never reaps a job's Popen child.
    _, status = os.waitpid(pid, 0)
    if os.waitstatus_to_exitcode(status) == 0:
        return None
    return b"".join(chunks).decode(errors="replace") or "failed"


def _tidy_gb(gb: float) -> float:
    """Round a derived budget to 2 decimals, always downwards.

    Rounding to nearest would let a budget derived from the whole machine
    land a hair above it, and the job would then be rejected by the very
    limit it was computed from.
    """
    return math.floor(gb * 100) / 100


def _env_problem(env) -> str | None:
    """Why a submitted --env payload is unusable, or None if it is fine."""
    if not isinstance(env, dict):
        return "env must be a mapping of strings to strings"
    size = 0
    for k, v in env.items():
        if not (isinstance(k, str) and isinstance(v, str)):
            return "env must be a mapping of strings to strings"
        size += len(k) + len(v) + 2
    if size > MAX_ENV_BYTES:
        return f"environment is too large ({size} bytes, limit {MAX_ENV_BYTES})"
    return None


def peer_creds(sock: socket.socket) -> tuple[int, int, int]:
    """(pid, uid, gid) of the unix-socket peer, from SO_PEERCRED."""
    data = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
    pid, uid, gid = struct.unpack("3i", data)
    return pid, uid, gid


def proc_starttime(pid: int) -> int | None:
    """starttime field of /proc/<pid>/stat; used to guard against pid reuse."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return int(stat.rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


class Daemon:
    def __init__(self, cfg: Config, saved_args: list[str]):
        self.cfg = cfg
        self.saved_args = saved_args
        self.jobs: dict[int, Job] = {}
        self.next_id = 1
        self.pool: ResourcePool | None = None
        self.cgroups = CgroupManager(enabled=cfg.use_cgroups)
        self.admin_gid: int | None = None
        self.is_root = os.getuid() == 0
        self._procs: dict[int, subprocess.Popen] = {}
        self._reserved: set[int] = set()  # job ids currently holding pool resources
        self._reservation: Reservation | None = None  # easy-backfill head budget
        self._doomed_cgroups: list[Path] = []  # busy at removal; retried each tick
        self._dirty = False
        self._stop = False
        self._reload = False
        self._dev_ok = False

    # -- paths -----------------------------------------------------------

    def job_dir(self, job_id: int) -> Path:
        return self.cfg.state_dir / "jobs" / str(job_id)

    def output_path(self, job_id: int) -> Path:
        return self.job_dir(job_id) / "output"

    def dev_link(self, job_id: int) -> Path:
        return self.cfg.dev_dir / "jobs" / str(job_id)

    @property
    def state_file(self) -> Path:
        return self.cfg.state_dir / "state.json"

    # -- permissions -----------------------------------------------------

    def is_admin(self, uid: int) -> bool:
        if uid == 0:
            return True
        if self.admin_gid is None:
            return False
        try:
            pw = pwd.getpwuid(uid)
        except KeyError:
            return False
        return self.admin_gid in os.getgrouplist(pw.pw_name, pw.pw_gid)

    # -- lifecycle -------------------------------------------------------

    async def run(self) -> None:
        os.umask(0o022)
        os.chdir("/")
        self._setup_dirs()
        try:
            self.cgroups.setup()
        except CgroupError as exc:
            raise StartupError(str(exc)) from exc
        self._setup_pool()
        self._resolve_admin_gid()
        self._load_state()
        # Only now: _load_state finalizes the jobs that died while we were
        # away, and finalizing reads memory.events out of their cgroups.
        self.cgroups.prune()
        # Apply the retention limit to what we just loaded, so lowering
        # --keep-finished takes effect on the next start rather than only
        # once another job happens to finish.
        self._trim_finished()
        self._rebuild_dev_links()

        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGHUP, self._request_reload)
        loop.add_signal_handler(signal.SIGTERM, self._request_stop)
        loop.add_signal_handler(signal.SIGINT, self._request_stop)

        self.cfg.socket_path.unlink(missing_ok=True)
        server = await asyncio.start_unix_server(
            self._client, path=str(self.cfg.socket_path), limit=MAX_LINE
        )
        os.chmod(self.cfg.socket_path, 0o666)
        log.info(
            "hpc-batchd %s ready on %s (schedule=%s max_lifetime=%s admin_group=%s list_is_public=%s)",
            __version__,
            self.cfg.socket_path,
            self.cfg.schedule,
            format_duration(self.cfg.max_lifetime),
            self.cfg.admin_group,
            self.cfg.list_is_public,
        )

        self._schedule()
        self._persist(force=True)
        try:
            while not (self._stop or self._reload):
                await asyncio.sleep(TICK_S)
                self._tick()
        finally:
            server.close()
            with contextlib.suppress(Exception):
                await server.wait_closed()
            self._persist(force=True)

        running = sum(1 for j in self.jobs.values() if j.state == RUNNING)
        if self._reload:
            log.info("reloading: re-exec with %d running job(s) preserved", running)
            self._reexec()
        else:
            log.info("shutting down; leaving %d running job(s) untouched", running)

    def _request_reload(self) -> None:
        self._reload = True

    def _request_stop(self) -> None:
        self._stop = True

    def _reexec(self) -> None:
        """Replace this process with a fresh daemon. Children (jobs) survive."""
        # Make sure the new interpreter can import us even though our cwd is
        # "/" (matters when running from a checkout rather than an install).
        pkg_parent = str(Path(__file__).resolve().parent.parent)
        env = dict(os.environ)
        paths = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
        if pkg_parent not in paths:
            env["PYTHONPATH"] = os.pathsep.join([pkg_parent, *paths])
        args = [sys.executable, "-m", "hpc_batch.daemon", *self.saved_args]
        os.execve(sys.executable, args, env)

    # -- setup -----------------------------------------------------------

    def _setup_dirs(self) -> None:
        (self.cfg.state_dir / "jobs").mkdir(parents=True, exist_ok=True)
        self.cfg.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.cfg.use_dev_dir:
            return
        try:
            (self.cfg.dev_dir / "jobs").mkdir(parents=True, exist_ok=True)
            self._dev_ok = True
        except OSError as exc:
            raise StartupError(
                f"cannot create {self.cfg.dev_dir} ({exc}); pass --no-dev-dir "
                f"to run without the job inspection entries"
            ) from exc

    def _setup_pool(self) -> None:
        nodes = discover_numa_nodes()
        gpus = discover_gpus()
        topology = discover_gpu_topology() if gpus else GpuTopology()
        node_mem = discover_node_memory_gb(nodes)
        # Memory is only confined to a node when we can actually write
        # cpuset.mems. Under --no-cgroups a job can allocate from any node, so
        # track one machine-wide pool rather than enforcing a split that is
        # not real. Runs after cgroups.setup(), which has already refused to
        # start if cgroups were wanted and cpuset was missing.
        confined = bool(node_mem) and self.cgroups.ready
        if not node_mem:
            node_mem = {next(iter(nodes), 0): total_memory_gb()}
        nodes, node_mem = apply_reserve(
            nodes, node_mem, self.cfg.reserve_cpu, self.cfg.reserve_mem
        )
        self.pool = ResourcePool(
            node_cpus=nodes, gpu_ids=gpus, node_mem_gb=node_mem, mem_confined=confined,
            gpu_topology=topology,
        )
        if len(gpus) > 1 and not topology:
            log.warning(
                "gpu topology unavailable (nvidia-smi topo -m); multi-gpu jobs will "
                "get the lowest free indices, which can straddle the interconnect"
            )
        log.info(
            "resources available to jobs: %d cpus over %d NUMA node(s), %d gpu(s), "
            "%.0f GiB memory (%s); reserved for the system: %d cpu, %g GiB",
            self.pool.total_cpus(),
            len(nodes),
            len(gpus),
            self.pool.usable_mem_gb(),
            "per-node" if self.pool.mem_confined else "machine-wide",
            self.cfg.reserve_cpu,
            self.cfg.reserve_mem,
        )

    def _resolve_admin_gid(self) -> None:
        try:
            self.admin_gid = grp.getgrnam(self.cfg.admin_group).gr_gid
        except KeyError:
            # "wheel" not existing on Debian is the usual cause.
            existing = sorted(g for g in ADMIN_GROUPS if group_exists(g))
            raise StartupError(
                f"--admin-group {self.cfg.admin_group!r} does not exist on this "
                f"system" + (f"; try one of: {', '.join(existing)}" if existing else "")
            ) from None

    # -- state persistence ----------------------------------------------

    def _load_state(self) -> None:
        if not self.state_file.exists():
            return
        try:
            data = json.loads(self.state_file.read_text())
            self.next_id = int(data.get("next_id", 1))
            jobs = [Job.from_dict(j) for j in data.get("jobs", [])]
        except (OSError, ValueError, TypeError, KeyError) as exc:
            log.error("corrupt state file %s (%s); starting fresh", self.state_file, exc)
            with contextlib.suppress(OSError):
                self.state_file.rename(self.state_file.with_suffix(".corrupt"))
            return
        for job in jobs:
            self.jobs[job.id] = job
            if job.state != RUNNING:
                continue
            alive, exit_code = self._probe(job)
            if alive:
                self.pool.reserve(job.allocation())
                self._reserved.add(job.id)
                log.info("adopted running job %d (pid %d, user %s)", job.id, job.pid, job.user)
            else:
                log.info("job %d died while the daemon was away", job.id)
                self._finalize(job, exit_code)
        queued = sum(1 for j in self.jobs.values() if j.state == QUEUED)
        if queued:
            log.info("restored %d queued job(s)", queued)

    def _persist(self, force: bool = False) -> None:
        if not (self._dirty or force):
            return
        data = {
            "next_id": self.next_id,
            "jobs": [j.to_dict() for j in sorted(self.jobs.values(), key=lambda j: j.id)],
        }
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        # Holds every user's jobs, and with --env their secrets: not 0644.
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.state_file)
        self._dirty = False

    # -- /dev + info files ----------------------------------------------

    def _own_job_path(self, path: Path, job: Job, mode: int) -> None:
        """Canonical ownership for every job artifact: owned by the user,
        group-readable by admins (falling back to the user's group)."""
        if self.is_root:
            group = self.admin_gid if self.admin_gid is not None else job.gid
            os.chown(path, job.uid, group)
        os.chmod(path, mode)

    def _job_changed(self, job: Job) -> None:
        """Every job mutation funnels through here: refresh the on-disk
        info.json and schedule a state.json write."""
        self._write_info(job)
        self._dirty = True

    def _make_job_dir(self, job: Job) -> None:
        d = self.job_dir(job.id)
        d.mkdir(parents=True, exist_ok=True)
        self._own_job_path(d, job, 0o750)
        self._dev_add(job)

    def _write_info(self, job: Job) -> None:
        info = self.job_dir(job.id) / "info.json"
        data = job.to_dict()
        data.pop("env")  # readable by admins; only the root-only state file replays it
        try:
            info.write_text(json.dumps(data, indent=1) + "\n")
            self._own_job_path(info, job, 0o640)
        except OSError as exc:
            log.warning("could not write info for job %d: %s", job.id, exc)

    def _dev_add(self, job: Job) -> None:
        """Best-effort: /dev entries mirror state, so failures never block a job."""
        if not self._dev_ok:
            return
        with contextlib.suppress(OSError):
            link = self.dev_link(job.id)
            link.unlink(missing_ok=True)
            link.symlink_to(self.job_dir(job.id))

    def _dev_remove(self, job_id: int) -> None:
        if not self._dev_ok:
            return
        with contextlib.suppress(OSError):
            self.dev_link(job_id).unlink(missing_ok=True)

    def _rebuild_dev_links(self) -> None:
        """Make /dev/hpc-batch/jobs mirror the queued+running jobs exactly."""
        if not self._dev_ok:
            return
        dev_jobs = self.cfg.dev_dir / "jobs"
        active = {str(j.id) for j in self.jobs.values() if j.state != DONE}
        with contextlib.suppress(OSError):
            for entry in dev_jobs.iterdir():
                if entry.name not in active:
                    with contextlib.suppress(OSError):
                        entry.unlink()
        for job in self.jobs.values():
            if job.state != DONE:
                self._make_job_dir(job)

    # -- job lifecycle ---------------------------------------------------

    def _submit(self, req: dict, uid: int) -> dict:
        argv = req.get("argv")
        if not (isinstance(argv, list) and argv and all(isinstance(a, str) for a in argv)):
            return err("no command given")
        try:
            pw = pwd.getpwuid(uid)
        except KeyError:
            return err(f"unknown uid {uid}")
        if not self.is_root and uid != os.getuid():
            return err("daemon is not running as root; it can only run your own jobs")
        try:
            cpu_raw = req.get("cpu")
            gpu_cores = int(req.get("gpu_cores") or 0)
            mem_raw = req.get("max_mem_gb")
            mem_gb = float(mem_raw) if mem_raw is not None else None
            time_raw = req.get("max_time_s")
            requested = int(time_raw) if time_raw is not None else None
            exclusive = bool(req.get("exclusive"))
            numa_local_mem = bool(req.get("numa_local_mem"))
            numa_local_gpu = bool(req.get("numa_local_gpu"))
            min_gpu_link = req.get("min_gpu_link") or None
            # An exclusive job owns the machine, so "unspecified" means all
            # of it rather than a single core.
            default_cpu = self.pool.total_cpus() if exclusive else 1
            cpu = int(cpu_raw) if cpu_raw is not None else default_cpu
        except (TypeError, ValueError):
            return err("malformed request")
        if cpu < 1 or gpu_cores < 0 or (mem_gb is not None and mem_gb <= 0) or (
            requested is not None and requested < 1
        ):
            return err("resource requests must be positive")

        # An unstated budget still has to be a number: an unbounded job would
        # be invisible to the scheduler and could starve everything sharing
        # its node. Derive one from the share of the machine being asked for.
        mem_defaulted = mem_gb is None
        if mem_defaulted:
            mem_gb = self._default_mem_gb(cpu, exclusive)

        problem = self.pool.validate(
            Request(cpu, gpu_cores, mem_gb, exclusive, numa_local_mem,
                    numa_local_gpu, min_gpu_link)
        )
        if problem:
            return err(problem)
        # Users may claim less time than the admin ceiling, never more.
        max_time = min(requested, self.cfg.max_lifetime) if requested else self.cfg.max_lifetime
        cwd = req.get("cwd") if isinstance(req.get("cwd"), str) else pw.pw_dir

        # Where the user's durable copy of the output goes. Absent or null
        # means they opted out with --no-output.
        output_dest = None
        raw_output = req.get("output")
        if isinstance(raw_output, str) and raw_output:
            output_dest, problem = self._resolve_output_dest(
                raw_output, self.next_id, uid, pw.pw_gid, pw.pw_name
            )
            if problem:
                return err(f"cannot save output there: {problem}")

        env = req.get("env") or {}
        problem = _env_problem(env)
        if problem:
            return err(problem)

        job = Job(
            id=self.next_id,
            user=pw.pw_name,
            uid=uid,
            gid=pw.pw_gid,
            argv=argv,
            cwd=cwd,
            cpu=cpu,
            gpu_cores=gpu_cores,
            max_mem_gb=mem_gb,
            max_time_s=max_time,
            exclusive=exclusive,
            numa_local_mem=numa_local_mem,
            numa_local_gpu=numa_local_gpu,
            min_gpu_link=min_gpu_link,
            env=env,
            mem_defaulted=mem_defaulted,
            submit_time=time.time(),
            output_dest=output_dest,
        )
        self.next_id += 1
        self.jobs[job.id] = job
        self._make_job_dir(job)
        self._job_changed(job)
        log.info(
            "job %d submitted by %s: %s (cpu=%d gpu=%d mem=%s%s max-time=%s%s%s)",
            job.id, job.user, job.command(), cpu, gpu_cores,
            format_gb(mem_gb),
            " default" if mem_defaulted else "",
            format_duration(max_time),
            " exclusive" if exclusive else "",
            "".join(
                f" numa-local-{what}"
                for what, on in (("mem", numa_local_mem), ("gpu", numa_local_gpu))
                if on
            ),
        )
        self._schedule()
        self._persist()
        return {
            "ok": True,
            "id": job.id,
            "state": job.state,
            "max_time_s": max_time,
            "cpu": cpu,
            "max_mem_gb": mem_gb,
            "mem_defaulted": mem_defaulted,
            "output_dest": job.output_dest,
        }

    def _default_mem_gb(self, cpu: int, exclusive: bool) -> float | None:
        """The memory budget for a job that did not name one.

        An exclusive job has the machine, so it gets all of it. Everyone else
        gets the share of a node their cores represent, floored so that a
        small `--cpu` on a core-dense machine is not left with a couple of
        gigabytes. Users who need a different number pass `--max-mem`.
        """
        if not self.pool.tracks_memory:
            return None
        if exclusive:
            return _tidy_gb(self.pool.usable_mem_gb())
        share = max(self.cfg.min_job_mem, self.pool.proportional_share(cpu))
        # Never default above what a job could actually be given.
        return _tidy_gb(min(share, self.pool.largest_node_mem_gb()))

    def _jobs_in_state(self, state: str) -> list[Job]:
        return [j for j in self.jobs.values() if j.state == state]

    def _queued_fifo(self) -> list[Job]:
        return sorted(self._jobs_in_state(QUEUED), key=lambda j: j.id)

    def _schedule(self) -> None:
        """Start queued jobs according to the configured scheduling policy.
        The policy reserves resources in the pool for each job it picks; we
        just spawn them."""
        queued = self._queued_fifo()
        running = self._jobs_in_state(RUNNING)
        to_start, self._reservation = plan(
            self.cfg.schedule, queued, self.pool, running, time.time(),
            self._reservation,
        )
        for job, alloc in to_start:
            self._try_start(job, alloc)

    def _try_start(self, job: Job, alloc: Allocation) -> None:
        try:
            self._start_job(job, alloc)
        except Exception as exc:
            log.exception("failed to start job %d", job.id)
            self.pool.release(alloc)
            self._write_output_line(job, f"hpc-batch: failed to start job: {exc}")
            job.reason = ERROR
            self._finalize(job, None)

    def _start_job(self, job: Job, alloc: Allocation) -> None:
        pw = pwd.getpwuid(job.uid)
        out_path = self.output_path(job.id)
        mem_bytes = int(alloc.mem_gb * (1 << 30)) if alloc.mem_gb else None
        cg = self.cgroups.create(job.id, alloc.cpus, alloc.mem_nodes(), mem_bytes)

        out_fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o640)
        devnull = os.open(os.devnull, os.O_RDONLY)
        self._own_job_path(out_path, job, 0o640)

        def preexec() -> None:
            # Runs in the child, still as root, just before exec.
            self.cgroups.confine_current(cg, alloc.cpus)
            drop_privileges(job.uid, pw.pw_gid, pw.pw_name)
            os.chdir(job.cwd)  # as the target user, so permissions apply

        try:
            proc = subprocess.Popen(
                job.argv,
                stdin=devnull,
                stdout=out_fd,
                stderr=out_fd,
                env=self._job_env(job, pw, alloc),
                start_new_session=True,
                preexec_fn=preexec,
            )
        finally:
            os.close(out_fd)
            os.close(devnull)

        self._procs[job.id] = proc
        self._reserved.add(job.id)
        job.pid = proc.pid
        job.proc_start = proc_starttime(proc.pid)
        job.state = RUNNING
        job.start_time = time.time()
        job.cpus = list(alloc.cpus)
        job.numa_node = alloc.numa_node
        job.numa_nodes = list(alloc.numa_nodes)
        job.mem_by_node = dict(alloc.mem_by_node)
        job.gpus = list(alloc.gpus)
        # Recorded now rather than derived on demand: `dispatch list` has no
        # topology, and a reload could hand it a different one.
        job.gpu_link = self.pool.gpu_topology.worst_link(alloc.gpus)
        job.cgroup = str(cg) if cg is not None else None
        # Only a queued job needs it; a re-adopted one is never re-spawned.
        # Keeping it would persist the submitter's secrets for the job's whole
        # retained life, and rewrite them on every state change.
        job.env = {}
        self._job_changed(job)
        log.info(
            "job %d started (pid %d, node %s, cpus %s%s, mem nodes %s%s)",
            job.id, job.pid,
            format_id_list(alloc.numa_nodes),
            format_id_list(alloc.cpus),
            self._gpu_note(alloc),
            format_id_list(alloc.mem_nodes()),
            " (spans nodes: remote memory is slower)" if alloc.spans_nodes else "",
        )

    def _gpu_note(self, alloc: Allocation) -> str:
        """The gpus a job got, and the link class pacing them when known."""
        if not alloc.gpus:
            return ""
        link = self.pool.gpu_topology.worst_link(alloc.gpus)
        return f", gpus {format_id_list(alloc.gpus)}" + (f" over {link}" if link else "")

    def _job_env(self, job: Job, pw: pwd.struct_passwd, alloc: Allocation) -> dict[str, str]:
        defaults = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LANG": os.environ.get("LANG", "C.UTF-8"),
        }
        ours = {
            "HOME": pw.pw_dir,
            "USER": pw.pw_name,
            "LOGNAME": pw.pw_name,
            "SHELL": pw.pw_shell or "/bin/sh",
            "HPC_BATCH_JOB_ID": str(job.id),
        }
        if self.pool.gpu_ids:
            # A forwarded one must never win: this is the whole of a job's gpu
            # isolation. Empty string for 0-gpu jobs, which must see no gpu.
            ours["CUDA_VISIBLE_DEVICES"] = format_id_list(alloc.gpus)
        return defaults | job.env | ours

    def _write_output_line(self, job: Job, text: str) -> None:
        with contextlib.suppress(OSError):
            with open(self.output_path(job.id), "a") as f:
                f.write(text + "\n")

    def _probe(self, job: Job) -> tuple[bool, int | None]:
        """Is the job's process still alive? Returns (alive, exit_code)."""
        proc = self._procs.get(job.id)
        if proc is not None:
            rc = proc.poll()
            return rc is None, rc
        if job.pid is None:
            return False, None
        try:
            pid, status = os.waitpid(job.pid, os.WNOHANG)
            if pid == 0:
                return True, None
            return False, os.waitstatus_to_exitcode(status)
        except ChildProcessError:
            pass
        # Not our child (daemon was fully restarted): fall back to /proc,
        # comparing starttime so a recycled pid is not mistaken for the job.
        start = proc_starttime(job.pid)
        if start is not None and start == job.proc_start:
            return True, None
        return False, None

    def _request_kill(self, job: Job, reason: str) -> None:
        if job.state != RUNNING:
            return
        job.reason = reason
        job.term_time = time.time()
        log.info("sending SIGTERM to job %d (%s)", job.id, reason)
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(job.pid, signal.SIGTERM)
        self._job_changed(job)

    def _hard_kill(self, job: Job) -> None:
        log.info("escalating to SIGKILL for job %d", job.id)
        if job.cgroup:
            self.cgroups.kill(Path(job.cgroup))
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(job.pid, signal.SIGKILL)

    # -- durable output ---------------------------------------------------

    def _resolve_output_dest(
        self, raw: str, job_id: int, uid: int, gid: int, user: str
    ) -> tuple[str | None, str | None]:
        """Decide where this job's durable output copy goes and check now
        that the user can write there. Returns (destination, error).

        Validating at submit time is the point: a job that runs for 20 hours
        and only then discovers its output directory is unwritable has lost
        the user's work for no reason.
        """
        path = Path(raw)
        if not path.is_absolute():
            return None, "output path must be absolute"
        # An existing directory means "write output.<id>.log in here";
        # anything else is taken as the exact filename to write.
        dest = path / f"output.{job_id}.log" if path.is_dir() else path
        parent = dest.parent

        def check() -> None:
            if not parent.is_dir():
                raise NotADirectoryError(f"{parent} is not a directory")
            if not os.access(parent, os.W_OK | os.X_OK):
                raise PermissionError(f"cannot write to {parent}")

        error = run_as_user(uid, gid, user, check)
        return (None, error) if error else (str(dest), None)

    def _retire_output(self, job: Job) -> None:
        """Hand the job's output to the user, then drop our copy.

        The state-dir file is a buffer so `dispatch attach` has something to
        stream while the job runs, not storage. Once the job ends we copy it
        to the user's chosen destination and unlink ours, so the only lasting
        copy is theirs. Clients still attached hold an open fd and finish
        their stream normally, by ordinary unlink semantics.

        A failed copy (directory deleted mid-run, quota exhausted) is
        recorded on the job rather than raised, since the job itself already
        succeeded or failed on its own terms, and the buffer stays put
        because it is then the only copy left.
        """
        src = self.output_path(job.id)
        # Not a start_time test: a job that failed in _try_start also never
        # started, but its buffer holds the reason and still has to reach the
        # user.
        if job.output_dest and src.exists():
            dest = Path(job.output_dest)

            def copy() -> None:
                # O_NOFOLLOW even though we have already dropped privileges:
                # a symlink appearing at the destination between validation
                # and now should fail loudly, not be followed.
                fd = os.open(
                    dest, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o640
                )
                with open(fd, "wb") as out, open(src, "rb") as inp:
                    shutil.copyfileobj(inp, out)

            job.output_error = run_as_user(job.uid, job.gid, job.user, copy)
            if job.output_error:
                log.warning(
                    "job %d: could not save output to %s: %s; keeping %s",
                    job.id, dest, job.output_error, src,
                )
                return
            log.info("job %d: output saved to %s", job.id, dest)

        with contextlib.suppress(OSError):
            src.unlink(missing_ok=True)

    def _finalize(self, job: Job, exit_code: int | None) -> None:
        job.state = DONE
        job.end_time = time.time()
        job.exit_code = exit_code
        self._procs.pop(job.id, None)
        if job.id in self._reserved:
            # Only jobs that actually hold pool resources release them, so a
            # finalize can never double-release or free what was never taken.
            self._reserved.discard(job.id)
            self.pool.release(job.allocation())
        if job.cgroup:
            cg = Path(job.cgroup)
            # Read the OOM counter before the cgroup goes away: without it a
            # job killed for exceeding its budget just looks like SIGKILL, and
            # the user has no way to tell that memory was the problem.
            if job.reason is None and self.cgroups.oom_killed(cg):
                job.reason = OOM
            if not self.cgroups.try_remove(cg):
                self._doomed_cgroups.append(cg)
            job.cgroup = None
        self._dev_remove(job.id)
        self._retire_output(job)
        self._job_changed(job)
        log.info(
            "job %d finished (exit=%s%s)",
            job.id, exit_code, f", reason={job.reason}" if job.reason else "",
        )
        self._trim_finished()

    def _forget(self, job: Job) -> None:
        self.jobs.pop(job.id, None)
        shutil.rmtree(self.job_dir(job.id), ignore_errors=True)
        self._dirty = True

    def _trim_finished(self) -> None:
        """Keep each user's most recent --keep-finished finished jobs.

        Per user, not globally: a global count lets one busy account evict
        everyone else's history, so your job list would empty out because
        somebody else was busy.

        Only metadata is normally at stake. A job's output goes to the user's
        own directory and the state-dir buffer is dropped when the job ends,
        so trimming costs one small info.json. The exception is a job whose
        delivery failed, where the buffer is the only copy left; those are
        trimmed like any other once they fall out of the window, so the
        eviction is logged with the path being lost.
        """
        keep = max(0, self.cfg.keep_finished)
        per_user: dict[int, list[Job]] = {}
        for job in self.jobs.values():
            if job.state == DONE:
                per_user.setdefault(job.uid, []).append(job)

        for jobs in per_user.values():
            jobs.sort(key=lambda j: j.id)  # id is submission order
            for job in (jobs[:-keep] if keep else jobs):
                if job.output_dest and job.output_error:
                    log.warning(
                        "job %d (%s): discarding undelivered output at %s; "
                        "past the %d finished-job limit",
                        job.id, job.user, self.output_path(job.id), keep,
                    )
                self._forget(job)

    # -- periodic work ---------------------------------------------------

    def _tick(self) -> None:
        now = time.time()
        for job in self._jobs_in_state(RUNNING):
            alive, exit_code = self._probe(job)
            if not alive:
                self._finalize(job, exit_code)
                continue
            uptime = job.uptime(now) or 0.0
            if uptime > job.max_time_s and job.term_time is None:
                self._write_output_line(
                    job,
                    f"hpc-batch: job exceeded its time limit ({format_duration(job.max_time_s)}); killing",
                )
                self._request_kill(job, TIMEOUT)
            elif job.term_time is not None and now - job.term_time > KILL_GRACE_S:
                self._hard_kill(job)
                job.term_time = now  # re-arm so we retry rather than busy-kill
        self._doomed_cgroups = [
            p for p in self._doomed_cgroups if not self.cgroups.try_remove(p)
        ]
        self._schedule()
        self._persist()

    # -- request handling ------------------------------------------------

    async def _client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            sock = writer.get_extra_info("socket")
            _, uid, _ = peer_creds(sock)
            req = await read_json(reader)
            if req is None:
                return
            cmd = req.get("cmd")
            if cmd == "attach":
                await self._h_attach(req, uid, writer)
            elif cmd == "new":
                await send_json(writer, self._submit(req, uid))
            elif cmd == "list":
                await send_json(writer, self._h_list(req, uid))
            elif cmd == "kill":
                await send_json(writer, self._h_kill(req, uid))
            else:
                await send_json(writer, err(f"unknown command {cmd!r}"))
        except (ConnectionResetError, BrokenPipeError):
            pass
        except Exception:
            log.exception("error handling client request")
            with contextlib.suppress(Exception):
                await send_json(writer, err("internal daemon error"))
        finally:
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()

    def _h_list(self, req: dict, uid: int) -> dict:
        show_all = bool(req.get("all"))
        show_finished = bool(req.get("finished"))
        if show_all and not (self.cfg.list_is_public or self.is_admin(uid)):
            return err(f"'list --all' requires membership of group {self.cfg.admin_group!r}")
        now = time.time()
        rows = [
            job.public_row(now)
            for job in sorted(self.jobs.values(), key=lambda j: j.id)
            if (show_finished or job.state != DONE) and (show_all or job.uid == uid)
        ]
        # Send the clock the rows were built against: uptime_s is already
        # relative to it, and the client needs it to place a job's absolute
        # start_time against "today" without consulting a second clock.
        return {"ok": True, "jobs": rows, "now": now}

    def _find_job(self, req: dict, uid: int) -> tuple[Job | None, dict | None]:
        try:
            job_id = int(req.get("id"))
        except (TypeError, ValueError):
            return None, err("invalid job id")
        job = self.jobs.get(job_id)
        if job is None:
            return None, err(f"no such job {job_id}")
        if job.uid != uid and not self.is_admin(uid):
            return None, err(f"job {job_id} belongs to {job.user}")
        return job, None

    def _h_kill(self, req: dict, uid: int) -> dict:
        job, problem = self._find_job(req, uid)
        if problem:
            return problem
        if job.state == DONE:
            return err(f"job {job.id} already finished")
        if job.state == QUEUED:
            job.reason = KILLED
            self._finalize(job, None)
            self._persist()
            return {"ok": True, "state": "removed"}
        self._request_kill(job, KILLED)
        self._persist()
        return {"ok": True, "state": "killing"}

    async def _h_attach(self, req: dict, uid: int, writer: asyncio.StreamWriter) -> None:
        job, problem = self._find_job(req, uid)
        if problem:
            await send_json(writer, problem)
            return
        await send_json(
            writer,
            {"ok": True, "state": job.state, "output_dest": job.output_dest},
        )
        path = self.output_path(job.id)
        f = None
        try:
            while True:
                # Snapshot before draining: if the job was already done, one
                # final drain below is guaranteed complete and we can return
                # without another poll cycle.
                done = job.state == DONE
                if f is None and path.exists():
                    f = open(path, "rb")
                if f is not None:
                    while chunk := f.read(65536):
                        writer.write(chunk)
                        await writer.drain()
                if done:
                    return
                await asyncio.sleep(ATTACH_POLL_S)
        finally:
            if f is not None:
                f.close()


# -- entry point ---------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Option dests match the Config field names so main() can build the
    Config mechanically — adding a knob means one option plus one field."""
    parser = argparse.ArgumentParser(
        prog="hpc-batchd",
        description="hpc-batch dispatch daemon (normally run from systemd)",
    )
    parser.add_argument(
        "--max-lifetime", type=duration_arg, default=86400, metavar="DURATION",
        help="kill any job running longer than this (default: 1d)",
    )
    parser.add_argument(
        "--list-is-public", action="store_true",
        help="allow non-admins to run 'dispatch list --all'",
    )
    parser.add_argument(
        "--admin-group", default="wheel", metavar="GROUP",
        help="members of this group are admins (default: wheel)",
    )
    parser.add_argument(
        "--socket", dest="socket_path", type=Path, default=Path(DEFAULT_SOCKET),
        metavar="PATH",
        help=f"unix socket to listen on (default: {DEFAULT_SOCKET})",
    )
    parser.add_argument(
        "--state-dir", type=Path, default=Path("/var/lib/hpc-batch"), metavar="DIR",
        help="where job state and output live (default: /var/lib/hpc-batch)",
    )
    parser.add_argument(
        "--dev-dir", type=Path, default=Path("/dev/hpc-batch"), metavar="DIR",
        help="where job inspection entries appear (default: /dev/hpc-batch)",
    )
    parser.add_argument(
        "--keep-finished", type=int, default=50, metavar="N",
        help="how many finished jobs to remember per user, for 'dispatch list "
             "--finished' (default: 50). Metadata only; job output is "
             "delivered to the user's own directory and never kept here.",
    )
    parser.add_argument(
        "--reserve-cpu", type=int, default=2, metavar="N",
        help="cpu cores held back for the OS and this daemon, never offered "
             "to jobs (default: 2). Taken from the lowest-numbered NUMA node "
             "so the other nodes keep their full width.",
    )
    parser.add_argument(
        "--reserve-mem", type=float, default=2.0, metavar="GB",
        help="memory in GiB held back for the OS and this daemon (default: 2). "
             "Taken from the same node as --reserve-cpu, and never more than "
             "half of any one node.",
    )
    parser.add_argument(
        "--min-job-mem", type=float, default=2.0, metavar="GB",
        help="floor for an automatically assigned memory budget (default: 2). "
             "Jobs that did not pass --max-mem get the share of a node their "
             "cores represent, but never less than this.",
    )
    parser.add_argument(
        "--no-cgroups", dest="use_cgroups", action="store_false",
        help="run without cgroups: cpu-affinity pinning only and no enforced "
             "memory limit. Development mode; without it the daemon refuses to "
             "start when cgroups are unavailable rather than silently "
             "dropping the isolation it promises",
    )
    parser.add_argument(
        "--no-dev-dir", dest="use_dev_dir", action="store_false",
        help="do not create the per-job inspection entries under --dev-dir",
    )
    parser.add_argument(
        "--schedule", choices=MODES, default=FIFO_STRICT, metavar="POLICY",
        help="scheduling policy: fifo-strict (default, head-of-line blocking), "
             "easy-backfill (fill idle resources then reserve for the head), or "
             "strict-backfill (backfill using each job's max-time so the head "
             "is never delayed)",
    )
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="logging verbosity (default: INFO)")
    parser.add_argument("--version", action="version", version=f"hpc-batchd {__version__}")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    saved_args = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(saved_args)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    cfg = Config(**{f.name: getattr(args, f.name) for f in fields(Config)})
    daemon = Daemon(cfg, saved_args)
    try:
        asyncio.run(daemon.run())
    except StartupError as exc:
        log.error("%s", exc)
        sys.exit(EX_CONFIG)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
