import pytest

from hpc_batch.util import format_duration, format_table, parse_duration


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
