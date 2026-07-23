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
    reason: str | None = None  # None | "killed" | "timeout" | "error"
    cpus: list[int] = field(default_factory=list)
    numa_node: int | None = None
    gpus: list[int] = field(default_factory=list)
    cgroup: str | None = None
    term_time: float | None = None  # when SIGTERM was sent, for escalation

    def command(self) -> str:
        return shlex.join(self.argv)

    def uptime(self, now: float) -> float | None:
        if self.state == RUNNING and self.start_time is not None:
            return now - self.start_time
        return None

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
            "uptime_s": self.uptime(now),
            "max_time_s": self.max_time_s,
            "exclusive": self.exclusive,
            "state": self.state,
        }
