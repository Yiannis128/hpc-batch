import time

import pytest

from hpc_batch.util import (
    format_duration,
    format_gb,
    format_table,
    format_time,
    parse_duration,
    split_assignments,
)


class TestParseDuration:
    def test_plain_seconds(self):
        assert parse_duration("3600") == 3600

    def test_units(self):
        assert parse_duration("45s") == 45
        assert parse_duration("15m") == 900
        assert parse_duration("2h") == 7200
        assert parse_duration("1d") == 86400

    def test_compound(self):
        assert parse_duration("1h30m") == 5400
        assert parse_duration("1d2h3m4s") == 86400 + 7200 + 180 + 4

    def test_whitespace_and_case(self):
        assert parse_duration(" 2H ") == 7200

    @pytest.mark.parametrize("bad", ["", "abc", "1x", "h1", "1h30", "-5", "1.5h"])
    def test_invalid(self, bad):
        with pytest.raises(ValueError):
            parse_duration(bad)


class TestFormatDuration:
    def test_none(self):
        assert format_duration(None) == "-"

    def test_zero(self):
        assert format_duration(0) == "0s"

    def test_compound(self):
        assert format_duration(5400) == "1h30m"
        assert format_duration(86400 + 61) == "1d1m1s"

    def test_negative_clamped(self):
        assert format_duration(-3) == "0s"

    def test_roundtrip(self):
        assert parse_duration(format_duration(123456)) == 123456


class TestFormatTable:
    def test_alignment(self):
        out = format_table(["A", "LONG"], [["xxx", "y"], ["z", "wwwww"]])
        lines = out.splitlines()
        assert lines[0] == "A    LONG"
        assert lines[1] == "xxx  y"
        assert lines[2] == "z    wwwww"


def _local(year, month, day, hour, minute, second=0):
    """Epoch seconds for a local wall-clock time, so the expected rendering is
    the same whatever timezone the tests run in.

    The wall clock skips an hour where daylight saving starts, and a time
    inside that gap does not exist locally: mktime quietly returns the
    shifted one instead, so a test would assert against a clock reading it
    never asked for. Reading the timestamp back catches that here, naming
    the cause, rather than as a puzzling one-hour diff in the assertion.

    An ambiguous time (the hour daylight saving repeats) needs no guard: both
    candidates read back as the wall clock that was asked for, which is all
    these tests render.
    """
    wanted = (year, month, day, hour, minute, second)
    ts = time.mktime((*wanted, 0, 0, -1))
    back = time.localtime(ts)
    got = (back.tm_year, back.tm_mon, back.tm_mday,
           back.tm_hour, back.tm_min, back.tm_sec)
    if got != wanted:
        raise ValueError(
            f"{wanted} is not a real local time in {time.tzname}: it falls in "
            f"a daylight-saving gap and moved to {got}. Pick a date away from "
            "a daylight-saving change."
        )
    return ts


class TestFormatTime:
    def test_not_started_is_a_dash(self):
        assert format_time(None, _local(2026, 7, 27, 9, 0)) == "-"

    def test_today_shows_the_clock_time(self):
        now = _local(2026, 7, 27, 18, 30)
        assert format_time(_local(2026, 7, 27, 14, 3, 12), now) == "14:03:12"

    def test_earlier_day_shows_the_date(self):
        now = _local(2026, 7, 27, 18, 30)
        assert format_time(_local(2026, 7, 26, 14, 3, 12), now) == "2026-07-26 14:03"

    def test_same_day_of_year_in_another_year_is_not_today(self):
        now = _local(2026, 7, 27, 18, 30)
        assert format_time(_local(2025, 7, 27, 14, 3), now) == "2025-07-27 14:03"

    def test_the_year_tells_old_jobs_apart(self):
        # Finished jobs are kept by count, not by age, so a listing can hold
        # two jobs a year apart. Without the year they would read the same.
        now = _local(2026, 7, 27, 18, 30)
        last_year = format_time(_local(2025, 6, 22, 17, 54), now)
        this_year = format_time(_local(2026, 6, 22, 17, 54), now)
        assert last_year != this_year


class TestLocalHelper:
    """The guard in `_local` protects every test above, so it is worth
    knowing it fires rather than trusting that it would."""

    def test_rejects_a_wall_clock_time_that_does_not_exist(self, monkeypatch):
        monkeypatch.setenv("TZ", "America/New_York")
        time.tzset()
        try:
            # 2026-03-08: the clocks go 01:59 -> 03:00, so 02:30 never happens.
            with pytest.raises(ValueError, match="daylight-saving gap"):
                _local(2026, 3, 8, 2, 30)
            # An hour either side of the gap is a real time and is accepted.
            assert _local(2026, 3, 8, 1, 30) < _local(2026, 3, 8, 3, 30)
        finally:
            monkeypatch.undo()
            time.tzset()


class TestFormatGb:
    def test_compact(self):
        assert format_gb(16.0) == "16G"
        assert format_gb(4.18) == "4.18G"

    def test_absent_is_a_dash(self):
        assert format_gb(None) == "-"

    def test_zero_is_not_absent(self):
        # A budget of 0 is a real (if useless) limit; rendering it as "-"
        # would claim the job has no limit at all.
        assert format_gb(0.0) == "0G"


class TestSplitAssignments:
    def test_leading_assignments_are_environment(self):
        env, rest = split_assignments(["FOO=bar", "N=1", "/bin/echo", "hi"])
        assert env == {"FOO": "bar", "N": "1"}
        assert rest == ["/bin/echo", "hi"]

    def test_stops_at_the_command(self):
        # Only leading words are environment: an argument that happens to look
        # like an assignment belongs to the job.
        env, rest = split_assignments(["/bin/make", "CC=gcc"])
        assert env == {}
        assert rest == ["/bin/make", "CC=gcc"]

    def test_paths_are_never_assignments(self):
        for word in ["./x=y", "/usr/bin/x=y", "1FOO=bar", "-o=v"]:
            assert split_assignments([word]) == ({}, [word])

    def test_empty_value(self):
        assert split_assignments(["FOO=", "/bin/true"]) == ({"FOO": ""}, ["/bin/true"])

    def test_value_may_contain_equals(self):
        env, rest = split_assignments(["EXPR=a=b", "/bin/true"])
        assert env == {"EXPR": "a=b"}
        assert rest == ["/bin/true"]

    def test_assignments_only_leaves_no_command(self):
        # The caller reports "no command given" from this, so the split must
        # not silently treat the last assignment as the program.
        assert split_assignments(["FOO=bar"]) == ({"FOO": "bar"}, [])
