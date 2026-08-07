"""The wire vocabulary is shared; the transports are not."""

import subprocess
import sys

from hpc_batch.protocol import DONE, QUEUED, RUNNING, decode, encode

CLIENT_SIDE = ["hpc_batch.protocol", "hpc_batch.jobs", "hpc_batch.client", "hpc_batch.install"]

_ASKED: tuple[bool, bool] | None = None


def who_imports_asyncio() -> tuple[bool, bool]:
    """(client side, daemon), asked in one fresh interpreter because pytest
    has already imported asyncio here, and spawning one costs ~70ms."""
    global _ASKED
    if _ASKED is None:
        code = (
            f"import sys, {', '.join(CLIENT_SIDE)}\n"
            "print('asyncio' in sys.modules)\n"
            "import hpc_batch.daemon\n"
            "print('asyncio' in sys.modules)"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        client, daemon = out.stdout.split()
        _ASKED = (client == "True", daemon == "True")
    return _ASKED


def test_the_client_side_never_pays_for_asyncio():
    # ~20ms of import time on every dispatch the user runs.
    client, _ = who_imports_asyncio()
    assert not client, f"one of {CLIENT_SIDE} now imports asyncio"


def test_the_daemon_still_gets_it():
    # Control: without it, a probe that answered False for the wrong reason
    # would let the assertion above pass vacuously.
    assert who_imports_asyncio()[1]


def test_a_frame_survives_the_round_trip():
    assert decode(encode({"cmd": "list"})) == {"cmd": "list"}


def test_a_bare_json_value_is_not_a_frame():
    # Both transports hand the result straight to .get, which a list has not.
    assert decode(b"[1, 2, 3]\n") is None
    assert decode(b"null\n") is None
    assert decode(b"{ truncated\n") is None


def test_the_states_are_the_strings_state_json_holds():
    # Renaming one silently breaks re-adoption across a daemon restart.
    assert (QUEUED, RUNNING, DONE) == ("queued", "running", "done")
