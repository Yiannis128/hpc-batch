import pytest

from hpc_batch.client import build_parser
from hpc_batch.modify import FIELDS, MAX_TIME, ModError, Verb
from test_jobs import make_job


class TestPermissionGates:
    """Every action carries its own gate. A field is not a permission unit:
    reading a limit and rewriting it are not the same privilege."""

    def test_reading_is_open_to_the_owner(self):
        assert MAX_TIME.actions[Verb.GET].admin_only is False

    @pytest.mark.parametrize("verb", [Verb.SET, Verb.ADD])
    def test_changing_a_limit_needs_admin(self, verb):
        assert MAX_TIME.actions[verb].admin_only is True


class TestTableIsComplete:
    def test_every_mutating_action_is_implemented(self):
        # The table says which verbs exist and each field's `apply` says what
        # they do. Adding a verb to one and not the other would leave the CLI
        # advertising an action that refuses itself at runtime, so walk the
        # table and make each verb prove it is wired up.
        # The call is the assertion: `apply` raises on a verb it does not
        # handle. Feeding back what `read` returns keeps this working for a
        # field whose values are not durations.
        for field in FIELDS.values():
            for action in field.actions.values():
                if action.mutates:
                    job = make_job()
                    field.apply(job, action.name, field.read(job))


class TestMaxTime:
    def test_get_reads_the_current_limit(self):
        assert MAX_TIME.read(make_job(max_time_s=7200)) == 7200

    def test_set_replaces_it(self):
        job = make_job(max_time_s=3600)
        MAX_TIME.apply(job, Verb.SET, 7200)
        assert job.max_time_s == 7200

    def test_add_extends_it(self):
        job = make_job(max_time_s=3600)
        MAX_TIME.apply(job, Verb.ADD, 172800)
        assert job.max_time_s == 176400

    def test_set_may_shorten_a_job(self):
        # The way to cut a job short: `add` cannot take a negative, because
        # argparse would read a leading "-" as an option.
        job = make_job(max_time_s=86400)
        MAX_TIME.apply(job, Verb.SET, 60)
        assert job.max_time_s == 60

    @pytest.mark.parametrize("value", [0, -1, None, "2d", 1.5e400])
    def test_rejects_a_value_that_is_not_a_positive_count_of_seconds(self, value):
        # "2d" included on purpose: the client parses durations, so anything
        # still in that form reached the daemon from somewhere else.
        job = make_job()
        with pytest.raises(ModError):
            MAX_TIME.apply(job, Verb.SET, value)
        assert job.max_time_s == 3600

    def test_refuses_a_job_that_is_already_being_killed(self):
        # SIGTERM is out and the tick only escalates from there, so raising
        # the limit would report a reprieve the job is not getting.
        job = make_job(term_time=100.0)
        with pytest.raises(ModError, match="already being killed"):
            MAX_TIME.apply(job, Verb.ADD, 3600)
        assert job.max_time_s == 3600

    def test_an_unknown_action_changes_nothing(self):
        job = make_job()
        with pytest.raises(ModError):
            MAX_TIME.apply(job, "double", 3600)
        assert job.max_time_s == 3600

    def test_durations_round_trip_through_the_cli(self):
        assert MAX_TIME.parse("2d") == 172800
        assert MAX_TIME.render(172800) == "2d"


class TestParser:
    """The `job` parser is generated from FIELDS, so the CLI can never offer
    an action the daemon does not implement, or miss one it does."""

    def test_accepts_the_documented_form(self):
        args = build_parser().parse_args(["job", "7", "max-time", "add", "2d"])
        assert (args.id, args.field, args.action, args.value) == (7, "max-time", "add", 172800)

    def test_get_takes_no_value(self):
        args = build_parser().parse_args(["job", "7", "max-time", "get"])
        assert (args.field, args.action) == ("max-time", "get")
        assert not hasattr(args, "value")

    @pytest.mark.parametrize(
        "argv",
        [
            ["job", "7", "max-time"],            # no action
            ["job", "7", "max-time", "add"],     # add without a value
            ["job", "7", "max-time", "double"],  # not an action
            ["job", "7", "max-mem", "get"],      # not a field
            ["job", "max-time", "get"],          # no job id
        ],
    )
    def test_rejects_an_incomplete_or_unknown_form(self, argv):
        with pytest.raises(SystemExit):
            build_parser().parse_args(argv)

    def test_the_help_says_which_actions_need_admin(self, capsys, monkeypatch):
        # Rendered from the same flag the daemon enforces, so the help cannot
        # advertise a permission that is not the one applied.
        monkeypatch.setenv("COLUMNS", "200")  # one action per line, unwrapped
        with pytest.raises(SystemExit):
            build_parser().parse_args(["job", "7", "max-time", "--help"])

        rendered = {
            line.split()[0]: line
            for line in capsys.readouterr().out.splitlines()
            if line.startswith("    ")
        }
        assert "admins only" not in rendered[Verb.GET]
        assert all("admins only" in rendered[verb] for verb in (Verb.SET, Verb.ADD))
