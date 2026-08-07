"""Installer tests.

Only the pure parts: what goes into the unit, which admin group is chosen,
what lands in /etc/profile.d. The steps with side effects (venv, systemctl)
are left to the machine, but the decisions they act on are all here.
"""

from pathlib import Path

import pytest

from hpc_batch.install import (
    DEFAULT_BIN_DIR,
    InstallError,
    bin_dirs_to_clean,
    detect_admin_group,
    login_path_provides,
    profile_snippet,
    read_record,
    render_unit,
    settle,
    unit_template,
    write_record,
)
from hpc_batch.util import ADMIN_GROUPS


class TestDetectAdminGroup:
    def test_prefers_the_first_that_exists(self):
        # Order matters: a machine with both should get the same answer every
        # time, or two installs of the same version disagree about who is an
        # admin.
        assert detect_admin_group(exists=lambda g: g in ("sudo", "adm")) == "sudo"

    def test_finds_the_debian_name_when_wheel_is_absent(self):
        assert detect_admin_group(exists=lambda g: g == "sudo") == "sudo"

    def test_refuses_rather_than_guessing(self):
        with pytest.raises(InstallError) as caught:
            detect_admin_group(exists=lambda g: False)
        assert "--admin-group" in str(caught.value)


class TestRenderUnit:
    def test_fills_in_the_daemon_path_and_group(self):
        unit = render_unit(unit_template(), Path("/opt/bin/hpc-batchd"), "sudo")
        assert "ExecStart=/opt/bin/hpc-batchd" in unit
        assert "--admin-group sudo" in unit

    def test_leaves_systemd_specifiers_alone(self):
        # $MAINPID is systemd's, not ours; substituting it would break reload.
        unit = render_unit(unit_template(), Path("/opt/bin/hpc-batchd"), "wheel")
        assert "ExecReload=/bin/kill -HUP $MAINPID" in unit

    def test_an_unfilled_placeholder_is_caught(self):
        with pytest.raises(InstallError) as caught:
            render_unit("ExecStart=@HPC_BATCHD@ --thing @NEW_ONE@", Path("/x"), "wheel")
        assert "@NEW_ONE@" in str(caught.value)

    def test_the_shipped_unit_still_carries_what_the_daemon_needs(self):
        # Each of these is a bug we have already had once: without Delegate=
        # the cgroup root never enables cpuset, without KillMode=process a
        # restart kills every running job, and without the exit-status guard a
        # misconfigured daemon restart-loops over its own error message.
        unit = render_unit(unit_template(), Path("/opt/bin/hpc-batchd"), "wheel")
        assert "Delegate=cpuset memory pids" in unit
        assert "KillMode=process" in unit
        assert "RestartPreventExitStatus=78" in unit


class TestProfileSnippet:
    def test_puts_the_bin_dir_on_path(self):
        assert str(DEFAULT_BIN_DIR) in profile_snippet(DEFAULT_BIN_DIR)
        assert "export PATH" in profile_snippet(DEFAULT_BIN_DIR)

    def test_is_idempotent_when_sourced_twice(self, tmp_path):
        # Login shells source profile.d more than once often enough that an
        # unguarded append grows PATH without bound.
        snippet = tmp_path / "hpc-batch.sh"
        snippet.write_text(profile_snippet(Path("/opt/bin")))
        script = tmp_path / "run.sh"
        script.write_text(
            f'PATH=/usr/bin\n. {snippet}\n. {snippet}\n. {snippet}\necho "$PATH"\n'
        )
        import subprocess

        out = subprocess.run(
            ["sh", str(script)], capture_output=True, text=True, check=True
        ).stdout.strip()
        assert out.count("/opt/bin") == 1


def etc_with(tmp_path: Path, **files: str) -> Path:
    """An /etc holding the given login-shell config. Keys are paths under it,
    with `_` for `/` so they read as keyword arguments."""
    etc = tmp_path / "etc"
    (etc / "profile.d").mkdir(parents=True)
    for name, text in files.items():
        (etc / name.replace("_", "/")).write_text(text)
    return etc


class TestLoginPathProvides:
    def test_finds_an_unguarded_profile_d_script(self, tmp_path):
        etc = etc_with(tmp_path, **{"profile.d/opt-bin.sh": 'export PATH="$PATH:/opt/bin"\n'})
        assert login_path_provides(Path("/opt/bin"), etc) == etc / "profile.d/opt-bin.sh"

    def test_finds_the_distro_default_in_etc_environment(self, tmp_path):
        etc = etc_with(tmp_path, environment='PATH="/usr/local/bin:/usr/bin:/bin"\n')
        assert login_path_provides(Path("/usr/local/bin"), etc) == etc / "environment"

    def test_says_nothing_when_the_dir_is_absent(self, tmp_path):
        etc = etc_with(tmp_path, environment='PATH="/usr/local/bin:/usr/bin"\n')
        assert login_path_provides(Path("/opt/bin"), etc) is None

    def test_a_longer_name_starting_the_same_way_is_not_a_match(self, tmp_path):
        # /opt/binaries must not answer for /opt/bin, or the entry points
        # never reach anyone's PATH and the installer says it was fine.
        etc = etc_with(tmp_path, environment='PATH="/usr/bin:/opt/binaries"\n')
        assert login_path_provides(Path("/opt/bin"), etc) is None

    def test_a_mention_outside_a_path_assignment_is_not_a_match(self, tmp_path):
        etc = etc_with(tmp_path, **{"profile.d/notes.sh": "# /opt/bin holds criu\n"})
        assert login_path_provides(Path("/opt/bin"), etc) is None

    def test_our_own_snippet_does_not_answer_for_itself(self, tmp_path):
        # Otherwise the second install reads what the first one wrote and
        # concludes it has nothing to do.
        ours = tmp_path / "etc" / "profile.d" / "hpc-batch.sh"
        etc = etc_with(tmp_path, **{"profile.d/hpc-batch.sh": profile_snippet(Path("/opt/bin"))})
        assert login_path_provides(Path("/opt/bin"), etc, skip=ours) is None
        assert login_path_provides(Path("/opt/bin"), etc) == ours

    def test_a_guarded_snippet_from_someone_else_does_answer(self, tmp_path):
        etc = etc_with(tmp_path, **{"profile.d/local.sh": profile_snippet(Path("/opt/bin"))})
        assert login_path_provides(Path("/opt/bin"), etc) == etc / "profile.d/local.sh"


class TestRecord:
    def test_round_trips(self, tmp_path):
        write_record(tmp_path, Path("/usr/local/bin"), "sudo")
        record = read_record(tmp_path)
        assert record["bin_dir"] == "/usr/local/bin"
        assert record["admin_group"] == "sudo"

    def test_a_fresh_prefix_has_nothing_to_disagree_with(self, tmp_path):
        assert read_record(tmp_path) == {}

    def test_a_damaged_record_reads_as_absent(self, tmp_path):
        # Refusing here would block the reinstall that repairs it.
        (tmp_path / "install.json").write_text("{not json")
        assert read_record(tmp_path) == {}


class TestSettle:
    def test_an_absent_flag_takes_what_the_last_install_used(self):
        # The bug this exists for: a bare upgrade linking into the default
        # bin dir and leaving the first install's symlinks unmanaged.
        record = {"bin_dir": "/usr/local/bin"}
        assert settle(record, "bin_dir", None) == "/usr/local/bin"

    def test_the_same_value_passed_again_is_not_a_disagreement(self):
        record = {"bin_dir": "/usr/local/bin"}
        assert settle(record, "bin_dir", Path("/usr/local/bin")) == "/usr/local/bin"

    def test_a_different_value_refuses_and_names_both(self):
        record = {"bin_dir": "/usr/local/bin"}
        with pytest.raises(InstallError) as caught:
            settle(record, "bin_dir", Path("/opt/bin"))
        message = str(caught.value)
        assert "--bin-dir" in message
        assert "/usr/local/bin" in message and "/opt/bin" in message
        assert "--uninstall" in message

    def test_the_refusal_names_the_flag_the_key_belongs_to(self):
        # Derived rather than passed, so a third option cannot be wired up
        # with a key and a flag that disagree.
        with pytest.raises(InstallError) as caught:
            settle({"admin_group": "sudo"}, "admin_group", "wheel")
        assert "--admin-group" in str(caught.value)

    def test_nothing_remembered_leaves_the_caller_its_default(self):
        assert settle({}, "bin_dir", None) is None

    def test_a_group_that_appeared_later_does_not_take_over(self):
        # detect_admin_group() would now answer wheel on a box that has since
        # grown one, quietly moving admin control off sudo.
        assert settle({"admin_group": "sudo"}, "admin_group", None) == "sudo"


class TestBinDirsToClean:
    def test_the_record_wins_over_the_default(self):
        # Cleaning /opt/bin because that is the default would leave the real
        # entry points dangling into a prefix uninstall just deleted.
        record = {"bin_dir": "/usr/local/bin"}
        assert bin_dirs_to_clean(record, None) == {Path("/usr/local/bin")}

    def test_an_explicit_dir_adds_to_the_recorded_one(self):
        record = {"bin_dir": "/usr/local/bin"}
        assert bin_dirs_to_clean(record, Path("/opt/bin")) == {
            Path("/usr/local/bin"),
            Path("/opt/bin"),
        }

    def test_no_record_falls_back_to_the_default(self):
        assert bin_dirs_to_clean({}, None) == {DEFAULT_BIN_DIR}


def test_admin_group_candidates_cover_the_common_distros():
    assert "wheel" in ADMIN_GROUPS  # Fedora, RHEL, Arch
    assert "sudo" in ADMIN_GROUPS  # Debian, Ubuntu
