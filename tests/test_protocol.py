"""The wire vocabulary is shared; the transports are not."""

import subprocess
import sys

from hpc_batch.protocol import DONE, QUEUED, RUNNING, encode, err

CLIENT_SIDE = ["hpc_batch.protocol", "hpc_batch.jobs", "hpc_batch.client", "hpc_batch.install"]


def imports_asyncio(*modules: str) -> bool:
    """Whether importing these modules drags asyncio in, asked in a fresh
    interpreter because pytest has already imported it here."""
    code = f"import sys, {', '.join(modules)}; print('asyncio' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    answer = out.stdout.strip()
    assert answer in ("True", "False"), out.stdout
    return answer == "True"


def test_the_client_side_never_pays_for_asyncio():
    # Only the daemon runs an event loop. read_json/send_json live there for
    # this reason: moving them back into protocol.py puts asyncio, and ~20ms,
    # into the startup of every `dispatch` the user runs.
    assert not imports_asyncio(*CLIENT_SIDE), f"one of {CLIENT_SIDE} now imports asyncio"


def test_the_daemon_still_gets_it():
    # Control: without it, a probe that answered False for the wrong reason
    # would let the assertion above pass vacuously.
    assert imports_asyncio("hpc_batch.daemon")


def test_a_frame_is_one_json_line():
    frame = encode({"cmd": "list"})
    assert frame.endswith(b"\n") and frame.count(b"\n") == 1


def test_the_states_are_the_strings_state_json_holds():
    # Renaming one silently breaks re-adoption across a daemon restart.
    assert (QUEUED, RUNNING, DONE) == ("queued", "running", "done")


def test_err_shape():
    assert err("nope") == {"ok": False, "error": "nope"}
