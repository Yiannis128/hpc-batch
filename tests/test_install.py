"""Installer tests.

Only the pure parts: what goes into the unit, which admin group is chosen,
what lands in /etc/profile.d. The steps with side effects (venv, systemctl)
are left to the machine, but the decisions they act on are all here.
"""

from pathlib import Path

import pytest

from hpc_batch.install import (
    ADMIN_GROUPS,
    DEFAULT_BIN_DIR,
    InstallError,
    detect_admin_group,
    profile_snippet,
    render_unit,
    unit_template,
)


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


def test_admin_group_candidates_cover_the_common_distros():
    assert "wheel" in ADMIN_GROUPS  # Fedora, RHEL, Arch
    assert "sudo" in ADMIN_GROUPS  # Debian, Ubuntu
