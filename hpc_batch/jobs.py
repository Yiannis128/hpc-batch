"""The Job model shared by the daemon's queue, state file and job info files."""

import shlex
from dataclasses import asdict, dataclass, field, fields

from .protocol import DONE, QUEUED, RUNNING
from .resources import Allocation, Request, charged_nodes


@dataclass
class Job:
    id: int
    user: str
    uid: int
    gid: int
    argv: list[str]
    cwd: str
    cpu: int
    gpu_cores: int
    max_mem_gb: float | None
    max_time_s: int
    exclusive: bool
    numa_local_mem: bool = False
    numa_local_gpu: bool = False
    min_gpu_link: str | None = None  # worst gpu link the job will accept
    env: dict[str, str] = field(default_factory=dict)  # --env, dropped at start; {} = clean
    mem_defaulted: bool = False  # budget assigned by us, not asked for
    state: str = QUEUED
    submit_time: float = 0.0
    start_time: float | None = None
    end_time: float | None = None
    pid: int | None = None
    proc_start: int | None = None  # /proc/<pid>/stat starttime, guards pid reuse
    exit_code: int | None = None
    reason: str | None = None  # None, or one of protocol's KILLED/TIMEOUT/ERROR
    cpus: list[int] = field(default_factory=list)
    numa_node: int | None = None  # home node: where this job's cpus are
    numa_nodes: list[int] = field(default_factory=list)  # every node its cpus span
    mem_by_node: dict[int, float] = field(default_factory=dict)  # exact charge
    gpus: list[int] = field(default_factory=list)
    gpu_link: str | None = None  # link class it actually got
    cgroup: str | None = None
    term_time: float | None = None  # when SIGTERM was sent, for escalation
    output_dest: str | None = None  # user's durable copy; None = don't save one
    output_error: str | None = None  # why delivering output_dest failed

    def command(self) -> str:
        return shlex.join(self.argv)

    def elapsed(self, now: float) -> float | None:
        """Wall time spent running: counting up for a running job, final for a
        finished one. Survives the job ending, so `dispatch list --finished`
        can show how long a job actually took."""
        if self.start_time is None:
            return None
        end = self.end_time if self.end_time is not None else now
        return end - self.start_time

    def uptime(self, now: float) -> float | None:
        """`elapsed` restricted to a running job. The time-limit check works
        off this, so that a job which is over but not yet reaped can never
        look like it is still accruing runtime."""
        return self.elapsed(now) if self.state == RUNNING else None

    def deadline(self, now: float) -> float:
        """When this job's time limit expires. For a job that has not started
        yet, measured from now (its start is bounded below by now)."""
        start = self.start_time if self.start_time is not None else now
        return start + self.max_time_s

    def request(self) -> Request:
        """What this job is asking the pool for."""
        return Request(
            cpu=self.cpu,
            gpu_cores=self.gpu_cores,
            mem_gb=self.max_mem_gb,
            exclusive=self.exclusive,
            numa_local_mem=self.numa_local_mem,
            numa_local_gpu=self.numa_local_gpu,
            min_gpu_link=self.min_gpu_link,
        )

    def allocation(self) -> Allocation:
        """The resources this (running) job holds, as one pool token."""
        return Allocation(
            cpus=list(self.cpus),
            numa_nodes=list(self.numa_nodes),
            gpus=list(self.gpus),
            mem_gb=self.max_mem_gb,
            mem_by_node=dict(self.mem_by_node),
            exclusive=self.exclusive,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        # JSON object keys are always strings, but the pool keys memory
        # charges by node id. Without this a reloaded job would release its
        # memory into domains that do not exist, leaking capacity.
        charge = kwargs.get("mem_by_node")
        if isinstance(charge, dict):
            kwargs["mem_by_node"] = {int(k): float(v) for k, v in charge.items()}
        job = cls(**kwargs)
        job._backfill_placement()
        return job

    def _backfill_placement(self) -> None:
        """Fill in placement fields absent from an older state file.

        Done here rather than in `allocation()` so it happens once and is
        written back on the next persist, instead of being re-derived on
        every scheduling tick and left invisible to `public_row`. A job with
        no recorded node is charged to node 0: over-charging one node is
        safe, charging nothing would let it be oversubscribed.
        """
        if self.state != RUNNING:
            return
        home = self.numa_node if self.numa_node is not None else 0
        if not self.numa_nodes:
            self.numa_node = home
            self.numa_nodes = [home]
        if not self.mem_by_node and self.max_mem_gb:
            self.mem_by_node = {home: self.max_mem_gb}

    def public_row(self, now: float) -> dict:
        """The fields exposed by `dispatch list`."""
        return {
            "user": self.user,
            "id": self.id,
            "command": self.command(),
            "start_time": self.start_time,
            "uptime_s": self.elapsed(now),
            "max_time_s": self.max_time_s,
            "exclusive": self.exclusive,
            "max_mem_gb": self.max_mem_gb,
            "mem_defaulted": self.mem_defaulted,
            # Remote memory is slower, so a job that had to spread its budget
            # says so rather than quietly producing worse timings. Same for the
            # link its gpus ended up talking over.
            "mem_spans_nodes": len(charged_nodes(self.mem_by_node)) > 1,
            "gpus": len(self.gpus),
            "gpu_link": self.gpu_link,
            "state": self.state,
            "exit_code": self.exit_code,
            "reason": self.reason,
            "output_dest": self.output_dest,
            "output_error": self.output_error,
        }
