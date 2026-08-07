"""What `dispatch job` can read or change on a job, and who may do it.

`FIELDS` is the whole contract: each entry names a field, the actions it
accepts, whether an action needs admin, and how a value is parsed, applied
and rendered. The client builds its argument parser out of this table and
the daemon dispatches through it, so a new knob is one entry here rather
than matching edits to the CLI, the wire format and the permission check.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .jobs import Job
from .util import duration_arg, format_duration


class Verb:
    """Action names. A class rather than module-level constants so `match`
    can use them: a bare name in a `case` captures instead of comparing."""

    GET = "get"
    SET = "set"
    ADD = "add"


class ModError(Exception):
    """A modification the daemon refuses; the message reaches the user."""


@dataclass(frozen=True)
class Action:
    """One verb on one field. `admin_only` has no default: an action that did
    not state its gate would inherit whatever sat above it in the table."""

    name: str
    help: str
    admin_only: bool
    mutates: bool = True  # also decides whether the verb takes a value


@dataclass(frozen=True)
class Field:
    name: str
    help: str
    metavar: str
    value_help: str
    parse: Callable[[str], Any]  # client side: argv text -> wire value
    read: Callable[[Job], Any]
    apply: Callable[[Job, str, Any], None]  # mutates; raises ModError
    render: Callable[[Any], str]
    actions: dict[str, Action]


def _apply_max_time(job: Job, action: str, value: Any) -> None:
    if job.term_time is not None:
        # SIGTERM is already out and the next tick only escalates it to
        # SIGKILL, so a larger limit cannot call the job back.
        raise ModError(f"job {job.id} is already being killed")
    try:
        # The client parsed "2d" for us, but it is the client.
        seconds = int(value)
    except (TypeError, ValueError, OverflowError):
        # OverflowError because JSON's 1e400 decodes to float("inf").
        raise ModError("max-time must be a whole number of seconds") from None
    if seconds < 1:
        raise ModError("max-time must be positive; use 'set' to shorten a job")
    match action:
        case Verb.SET:
            job.max_time_s = seconds
        case Verb.ADD:
            job.max_time_s += seconds
        case _:
            raise ModError(f"cannot {action} max-time")


MAX_TIME = Field(
    name="max-time",
    help="the job's time limit",
    metavar="DURATION",
    value_help="a duration, e.g. 30m, 2h or 1d",
    parse=duration_arg,
    read=lambda job: job.max_time_s,
    apply=_apply_max_time,
    render=format_duration,
    actions={
        Verb.GET: Action(Verb.GET, "print the current time limit",
                         admin_only=False, mutates=False),
        Verb.SET: Action(Verb.SET, "replace the time limit", admin_only=True),
        Verb.ADD: Action(Verb.ADD, "extend the time limit by this much", admin_only=True),
    },
)

FIELDS: dict[str, Field] = {MAX_TIME.name: MAX_TIME}
