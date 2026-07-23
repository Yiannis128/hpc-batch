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
from .cgroup import CgroupManager
from .jobs import DONE, QUEUED, RUNNING, Job
from .protocol import DEFAULT_SOCKET, MAX_LINE, err, read_json, send_json
from .resources import (
    Allocation,
    ResourcePool,
    discover_gpus,
    discover_numa_nodes,
    total_memory_gb,
)
from .scheduling import FIFO_STRICT, MODES, Reservation, plan
from .util import duration_arg, format_duration

log = logging.getLogger("hpc-batchd")

TICK_S = 1.0
KILL_GRACE_S = 10.0
FINISHED_KEEP = 50
ATTACH_POLL_S = 0.3


@dataclass
class Config:
    max_lifetime: int
    list_is_public: bool
    admin_group: str
    socket_path: Path
    state_dir: Path
    dev_dir: Path
    use_cgroups: bool
    schedule: str


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
        self.cgroups.setup()
        self._setup_pool()
        self._resolve_admin_gid()
        self._load_state()
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
        try:
            (self.cfg.dev_dir / "jobs").mkdir(parents=True, exist_ok=True)
            self._dev_ok = True
        except OSError as exc:
            log.warning("cannot create %s (%s); /dev job entries disabled", self.cfg.dev_dir, exc)

    def _setup_pool(self) -> None:
        nodes = discover_numa_nodes()
        gpus = discover_gpus()
        mem = total_memory_gb()
        self.pool = ResourcePool(node_cpus=nodes, gpu_ids=gpus, total_mem_gb=mem)
        log.info(
            "resources: %d cpus over %d NUMA node(s), %d gpu(s), %.0f GiB memory",
            sum(len(c) for c in nodes.values()),
            len(nodes),
            len(gpus),
            mem,
        )

    def _resolve_admin_gid(self) -> None:
        try:
            self.admin_gid = grp.getgrnam(self.cfg.admin_group).gr_gid
        except KeyError:
            log.warning("admin group %r does not exist", self.cfg.admin_group)
            self.admin_gid = None

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
        try:
            info.write_text(json.dumps(job.to_dict(), indent=1) + "\n")
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
            cpu = int(req.get("cpu") or 1)
            gpu_cores = int(req.get("gpu_cores") or 0)
            mem_raw = req.get("max_mem_gb")
            mem_gb = float(mem_raw) if mem_raw is not None else None
            time_raw = req.get("max_time_s")
            requested = int(time_raw) if time_raw is not None else None
            exclusive = bool(req.get("exclusive"))
        except (TypeError, ValueError):
            return err("malformed request")
        if cpu < 1 or gpu_cores < 0 or (mem_gb is not None and mem_gb <= 0) or (
            requested is not None and requested < 1
        ):
            return err("resource requests must be positive")
        problem = self.pool.validate(cpu, gpu_cores, mem_gb)
        if problem:
            return err(problem)
        # Users may claim less time than the admin ceiling, never more.
        max_time = min(requested, self.cfg.max_lifetime) if requested else self.cfg.max_lifetime
        cwd = req.get("cwd") if isinstance(req.get("cwd"), str) else pw.pw_dir

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
            submit_time=time.time(),
        )
        self.next_id += 1
        self.jobs[job.id] = job
        self._make_job_dir(job)
        self._job_changed(job)
        log.info(
            "job %d submitted by %s: %s (cpu=%d gpu=%d mem=%s max-time=%s%s)",
            job.id, job.user, job.command(), cpu, gpu_cores,
            f"{mem_gb:g}G" if mem_gb else "-", format_duration(max_time),
            " exclusive" if exclusive else "",
        )
        self._schedule()
        self._persist()
        return {"ok": True, "id": job.id, "state": job.state, "max_time_s": max_time}

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
            job.reason = "error"
            self._finalize(job, None)

    def _start_job(self, job: Job, alloc: Allocation) -> None:
        pw = pwd.getpwuid(job.uid)
        out_path = self.output_path(job.id)
        mem_bytes = int(alloc.mem_gb * (1 << 30)) if alloc.mem_gb else None
        cg = self.cgroups.create(job.id, alloc.cpus, alloc.numa_node, mem_bytes)

        out_fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o640)
        devnull = os.open(os.devnull, os.O_RDONLY)
        self._own_job_path(out_path, job, 0o640)

        def preexec() -> None:
            # Runs in the child, still as root, just before exec.
            self.cgroups.confine_current(cg, alloc.cpus)
            if self.is_root:
                os.initgroups(pw.pw_name, pw.pw_gid)
                os.setgid(pw.pw_gid)
                os.setuid(job.uid)
            os.chdir(job.cwd)  # as the target user, so permissions apply

        env = {
            "HOME": pw.pw_dir,
            "USER": pw.pw_name,
            "LOGNAME": pw.pw_name,
            "SHELL": pw.pw_shell or "/bin/sh",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "HPC_BATCH_JOB_ID": str(job.id),
        }
        if self.pool.gpu_ids:
            # Empty string for 0-gpu jobs: they must not see any gpu.
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in alloc.gpus)

        try:
            proc = subprocess.Popen(
                job.argv,
                stdin=devnull,
                stdout=out_fd,
                stderr=out_fd,
                env=env,
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
        job.gpus = list(alloc.gpus)
        job.cgroup = str(cg) if cg is not None else None
        self._job_changed(job)
        log.info(
            "job %d started (pid %d, node %d, cpus %s%s)",
            job.id, job.pid, alloc.numa_node,
            ",".join(str(c) for c in alloc.cpus),
            f", gpus {','.join(str(g) for g in alloc.gpus)}" if alloc.gpus else "",
        )

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
            if not self.cgroups.try_remove(cg):
                self._doomed_cgroups.append(cg)
            job.cgroup = None
        self._dev_remove(job.id)
        self._job_changed(job)
        log.info(
            "job %d finished (exit=%s%s)",
            job.id, exit_code, f", reason={job.reason}" if job.reason else "",
        )
        self._trim_finished()

    def _trim_finished(self) -> None:
        done = sorted(
            (j for j in self.jobs.values() if j.state == DONE), key=lambda j: j.id
        )
        for job in done[:-FINISHED_KEEP]:
            del self.jobs[job.id]
            shutil.rmtree(self.job_dir(job.id), ignore_errors=True)

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
                self._request_kill(job, "timeout")
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
        if show_all and not (self.cfg.list_is_public or self.is_admin(uid)):
            return err(f"'list --all' requires membership of group {self.cfg.admin_group!r}")
        now = time.time()
        rows = [
            job.public_row(now)
            for job in sorted(self.jobs.values(), key=lambda j: j.id)
            if job.state != DONE and (show_all or job.uid == uid)
        ]
        return {"ok": True, "jobs": rows}

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
            job.reason = "killed"
            self._finalize(job, None)
            self._persist()
            return {"ok": True, "state": "removed"}
        self._request_kill(job, "killed")
        self._persist()
        return {"ok": True, "state": "killing"}

    async def _h_attach(self, req: dict, uid: int, writer: asyncio.StreamWriter) -> None:
        job, problem = self._find_job(req, uid)
        if problem:
            await send_json(writer, problem)
            return
        await send_json(writer, {"ok": True, "state": job.state})
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
        "--no-cgroups", dest="use_cgroups", action="store_false",
        help="do not use cgroups (development mode; falls back to cpu affinity)",
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
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
