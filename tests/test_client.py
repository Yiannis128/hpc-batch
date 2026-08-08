"""What the client puts on the wire. The daemon's half of the environment
chain is in test_daemon.py; this is the half that composes what it receives."""

import pytest

from hpc_batch import client


@pytest.fixture
def sent(monkeypatch) -> list[dict]:
    """The requests cmd_new would have sent, with the socket taken out."""
    captured: list[dict] = []

    def fake_request(req: dict) -> dict:
        captured.append(req)
        return {"id": 1, "state": "queued", "cpu": 1, "max_time_s": 60}

    monkeypatch.setattr(client, "_request", fake_request)
    return captured


class TestJobEnvironment:
    def test_an_assignment_beats_the_same_name_in_the_sweep(self, monkeypatch, sent):
        # job.env reaches the daemon as one layer, so this precedence exists
        # nowhere else: the word was typed for this job, the sweep is whatever
        # the submitting shell happened to be holding.
        monkeypatch.setenv("MODEL", "from-the-shell")
        client.main(["new", "--no-output", "--env", "--", "MODEL=for-this-job", "/bin/true"])
        assert sent[0]["env"]["MODEL"] == "for-this-job"

    def test_the_sweep_still_carries_everything_else(self, monkeypatch, sent):
        monkeypatch.setenv("UNRELATED", "kept")
        client.main(["new", "--no-output", "--env", "--", "MODEL=x", "/bin/true"])
        assert sent[0]["env"]["UNRELATED"] == "kept"

    def test_without_env_only_the_assignments_are_sent(self, monkeypatch, sent):
        # A clean environment is the default; --env is what opts out of it.
        monkeypatch.setenv("MODEL", "from-the-shell")
        client.main(["new", "--no-output", "--", "N=1", "/bin/true"])
        assert sent[0]["env"] == {"N": "1"}
