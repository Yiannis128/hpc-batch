"""The Job model shared by the daemon's queue, state file and job info files."""

import shlex
from dataclasses import asdict, dataclass, field, fields

from .protocol import DONE, QUEUED, RUNNING
from .resources import Allocation


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
    state: str = QUEUED
    submit_time: float = 0.0
    start_time: float | None = None
    end_time: float | None = None
    pid: int | None = None
    proc_start: int | None = None  # /proc/<pid>/stat starttime, guards pid reuse
    exit_code: int | None = None
    reason: str | None = None  # None, or one of protocol's KILLED/TIMEOUT/ERROR
    cpus: list[int] = field(default_factory=list)
    numa_node: int | None = None
    gpus: list[int] = field(default_factory=list)
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

    def allocation(self) -> Allocation:
        """The resources this (running) job holds, as one pool token."""
        return Allocation(
            cpus=list(self.cpus),
            numa_node=self.numa_node or 0,
            gpus=list(self.gpus),
            mem_gb=self.max_mem_gb,
            exclusive=self.exclusive,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def public_row(self, now: float) -> dict:
        """The fields exposed by `dispatch list`."""
        return {
            "user": self.user,
            "id": self.id,
            "command": self.command(),
            "uptime_s": self.elapsed(now),
            "max_time_s": self.max_time_s,
            "exclusive": self.exclusive,
            "state": self.state,
            "exit_code": self.exit_code,
            "reason": self.reason,
            "output_dest": self.output_dest,
            "output_error": self.output_error,
        }
