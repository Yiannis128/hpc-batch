"""Client/daemon wire protocol: newline-delimited JSON over a unix socket.

Every request is a single JSON object on one line. Every response starts with
a single JSON object on one line; for `attach` the response line is followed
by a raw byte stream of the job's output until the connection closes.

The framing lives here, the transports do not: keep this module free of
asyncio, or every `dispatch` pays for an event loop it never runs.
"""

import json
import os

from .util import DEFAULT_SOCKET

# Generous cap on a single protocol line (command lines can be long).
MAX_LINE = 1 << 20

# Job states; part of the wire contract (list/attach responses).
QUEUED = "queued"
RUNNING = "running"
DONE = "done"

# Why a job stopped, when it did not exit on its own. Also wire contract:
# `dispatch list --finished` prints these verbatim in its EXIT column.
KILLED = "killed"
TIMEOUT = "timeout"
ERROR = "error"
OOM = "oom"


def encode(obj: dict) -> bytes:
    """One protocol frame: a JSON object on a single line."""
    return json.dumps(obj).encode() + b"\n"


def decode(line: bytes) -> dict | None:
    """One frame back, or None if the line is not one. Both transports come
    here, so neither has to remember that a bare JSON value has no `.get`."""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def err(message: str) -> dict:
    """The protocol's error-response shape."""
    return {"ok": False, "error": message}


def socket_path() -> str:
    """Socket path used by the client; override with $HPC_BATCH_SOCKET."""
    return os.environ.get("HPC_BATCH_SOCKET", DEFAULT_SOCKET)
