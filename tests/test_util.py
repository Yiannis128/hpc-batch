import time

import pytest

from hpc_batch.util import (
    format_duration,
    format_gb,
    format_table,
    format_time,
    parse_duration,
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
    the same whatever timezone the tests run in."""
    return time.mktime((year, month, day, hour, minute, second, 0, 0, -1))


class TestFormatTime:
    def test_not_started_is_a_dash(self):
        assert format_time(None, _local(2026, 7, 27, 9, 0)) == "-"

    def test_today_shows_the_clock_time(self):
        now = _local(2026, 7, 27, 18, 30)
        assert format_time(_local(2026, 7, 27, 14, 3, 12), now) == "14:03:12"

    def test_earlier_day_shows_the_date(self):
        now = _local(2026, 7, 27, 18, 30)
        assert format_time(_local(2026, 7, 26, 14, 3, 12), now) == "07-26 14:03"

    def test_same_day_of_year_in_another_year_is_not_today(self):
        now = _local(2026, 7, 27, 18, 30)
        assert format_time(_local(2025, 7, 27, 14, 3), now) == "07-27 14:03"


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
