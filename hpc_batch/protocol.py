"""Client/daemon wire protocol: newline-delimited JSON over a unix socket.

Every request is a single JSON object on one line. Every response starts with
a single JSON object on one line; for `attach` the response line is followed
by a raw byte stream of the job's output until the connection closes.
"""

import asyncio
import json
import os

DEFAULT_SOCKET = "/run/hpc-batch/hpc-batch.sock"

# Generous cap on a single protocol line (command lines can be long).
MAX_LINE = 1 << 20

# Job states; part of the wire contract (list/attach responses).
QUEUED = "queued"
RUNNING = "running"
DONE = "done"


def encode(obj: dict) -> bytes:
    """One protocol frame: a JSON object on a single line."""
    return json.dumps(obj).encode() + b"\n"


def err(message: str) -> dict:
    """The protocol's error-response shape."""
    return {"ok": False, "error": message}


def socket_path() -> str:
    """Socket path used by the client; override with $HPC_BATCH_SOCKET."""
    return os.environ.get("HPC_BATCH_SOCKET", DEFAULT_SOCKET)


async def read_json(reader: asyncio.StreamReader) -> dict | None:
    """Read one JSON line from an asyncio stream; None on EOF or garbage."""
    try:
        line = await reader.readline()
    except (asyncio.LimitOverrunError, ValueError):
        return None
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


async def send_json(writer: asyncio.StreamWriter, obj: dict) -> None:
    writer.write(encode(obj))
    await writer.drain()
