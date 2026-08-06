"""System-wide installer: `hpc-batch-install`.

`pip install hpc-batch` leaves two entry points in whichever environment ran
pip, which is not a working system. This puts the code in a venv nobody else
writes to, puts the client on every user's PATH, and writes a unit pointing
at the right binary with an admin group that exists on this distro.

It is also the upgrade path, so every step is idempotent: rerunning it
reinstalls the package and reloads the daemon rather than restarting it,
which is what keeps running jobs alive.

Refusals follow the daemon's rule -- if something was asked for and cannot
be provided, say so and stop, rather than installing something that looks
finished and is not.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__
from .util import ADMIN_GROUPS, group_exists

DEFAULT_PREFIX = Path("/opt/hpc-batch")
DEFAULT_BIN_DIR = Path("/opt/bin")
UNIT_PATH = Path("/etc/systemd/system/hpc-batch.service")
PROFILE_PATH = Path("/etc/profile.d/hpc-batch.sh")
ENTRY_POINTS = ("dispatch", "hpc-batchd", "hpc-batch-install")


class InstallError(Exception):
    """Something the installer needs is missing or refused."""


def detect_admin_group(exists=group_exists) -> str:
    """First of the usual admin groups that exists here.

    Debian calls it `sudo`, Fedora and Arch call it `wheel`. Hard-coding
    either means a silently admin-less daemon on the other half of the world.
    """
    for name in ADMIN_GROUPS:
        if exists(name):
            return name
    raise InstallError(
        f"none of {', '.join(ADMIN_GROUPS)} exists on this system; pass "
        f"--admin-group with a group whose members should administer jobs"
    )


def render_unit(template: str, hpc_batchd: Path, admin_group: str) -> str:
    """Fill the shipped unit in. Placeholders are @-delimited rather than
    $-delimited because the unit itself contains $MAINPID."""
    filled = template.replace("@HPC_BATCHD@", str(hpc_batchd))
    filled = filled.replace("@ADMIN_GROUP@", admin_group)
    leftover = re.findall(r"@[A-Z_]+@", filled)
    if leftover:
        raise InstallError(f"unit template has unfilled placeholders: {leftover}")
    return filled


def profile_snippet(bin_dir: Path) -> str:
    """PATH entry for login shells. Guarded so that re-sourcing a profile,
    or a user who already has the directory, does not accumulate copies."""
    return (
        "# Added by hpc-batch-install: puts `dispatch` on every user's PATH.\n"
        f'case ":$PATH:" in\n'
        f'  *":{bin_dir}:"*) ;;\n'
        f'  *) PATH="$PATH:{bin_dir}" ;;\n'
        f"esac\n"
        "export PATH\n"
    )


def unit_template() -> str:
    path = Path(__file__).with_name("data") / "hpc-batch.service"
    try:
        return path.read_text()
    except OSError as exc:
        raise InstallError(f"packaged unit file is missing ({exc})") from exc


def _run(cmd: list[str], what: str) -> None:
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise InstallError(f"{cmd[0]} not found; needed to {what}") from exc
    except subprocess.CalledProcessError as exc:
        raise InstallError(f"failed to {what}: {' '.join(cmd)} exited {exc.returncode}") from exc


def _check_preconditions() -> None:
    if os.geteuid() != 0:
        raise InstallError("must run as root (try: sudo hpc-batch-install)")
    if not Path("/run/systemd/system").is_dir():
        raise InstallError("systemd is not running; hpc-batch is a systemd service")
    if shutil.which("systemctl") is None:
        raise InstallError("systemctl not found on PATH")


def _make_venv(prefix: Path) -> Path:
    pip = prefix / "bin" / "pip"
    if not pip.exists():
        # Debian splits venv out of the stdlib package, so this is a real and
        # confusing failure rather than a theoretical one.
        try:
            subprocess.run([sys.executable, "-m", "venv", str(prefix)], check=True)
        except subprocess.CalledProcessError as exc:
            raise InstallError(
                f"could not create a virtualenv at {prefix}; on Debian/Ubuntu "
                f"install python3-venv"
            ) from exc
    if not pip.exists():
        raise InstallError(f"{pip} still missing after creating the virtualenv")
    return pip


def install(args: argparse.Namespace) -> None:
    _check_preconditions()
    admin_group = args.admin_group or detect_admin_group()

    if not args.already_installed:
        pip = _make_venv(args.prefix)
        _run([str(pip), "install", "--upgrade", args.spec], f"install {args.spec}")

    args.bin_dir.mkdir(parents=True, exist_ok=True)
    for name in ENTRY_POINTS:
        target = args.prefix / "bin" / name
        if not target.exists():
            raise InstallError(f"{target} was not installed by {args.spec}")
        link = args.bin_dir / name
        link.unlink(missing_ok=True)
        link.symlink_to(target)
    print(f"installed {', '.join(ENTRY_POINTS)} in {args.bin_dir}")

    PROFILE_PATH.write_text(profile_snippet(args.bin_dir))
    PROFILE_PATH.chmod(0o644)
    print(f"wrote {PROFILE_PATH} (login shells pick it up; this one will not)")

    unit = render_unit(unit_template(), args.bin_dir / "hpc-batchd", admin_group)
    UNIT_PATH.write_text(unit)
    UNIT_PATH.chmod(0o644)
    print(f"wrote {UNIT_PATH} (admin group: {admin_group})")

    _run(["systemctl", "daemon-reload"], "reload the systemd manager")
    if args.no_start:
        print("skipping start (--no-start); enable it with: systemctl enable --now hpc-batch")
        return

    active = subprocess.run(
        ["systemctl", "is-active", "--quiet", "hpc-batch"], check=False
    ).returncode == 0
    if active:
        # Reload rather than restart: it re-execs the same pid, so running
        # jobs stay its children and their exit codes are still collected.
        _run(["systemctl", "reload", "hpc-batch"], "reload the daemon")
        print("reloaded the running daemon; jobs were not interrupted")
    else:
        _run(["systemctl", "enable", "--now", "hpc-batch"], "enable the daemon")
        print("started hpc-batch")
    print("verify with: systemctl status hpc-batch && dispatch list")


def uninstall(args: argparse.Namespace) -> None:
    _check_preconditions()
    subprocess.run(["systemctl", "disable", "--now", "hpc-batch"], check=False)
    for path in (UNIT_PATH, PROFILE_PATH):
        path.unlink(missing_ok=True)
        print(f"removed {path}")
    for name in ENTRY_POINTS:
        (args.bin_dir / name).unlink(missing_ok=True)
    shutil.rmtree(args.prefix, ignore_errors=True)
    _run(["systemctl", "daemon-reload"], "reload the systemd manager")
    print(f"removed {args.prefix} and the {args.bin_dir} entry points")
    print("left /var/lib/hpc-batch alone: it holds job history and queued jobs")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hpc-batch-install",
        description="Install hpc-batch system-wide: a private virtualenv, the "
                    "client on every user's PATH, and the systemd unit. Rerun "
                    "it to upgrade; running jobs are not interrupted.",
    )
    parser.add_argument("--prefix", type=Path, default=DEFAULT_PREFIX, metavar="DIR",
                        help=f"virtualenv to install into (default: {DEFAULT_PREFIX})")
    parser.add_argument("--bin-dir", type=Path, default=DEFAULT_BIN_DIR, metavar="DIR",
                        help=f"where the entry points are linked, and what is "
                             f"added to PATH (default: {DEFAULT_BIN_DIR})")
    parser.add_argument("--admin-group", default=None, metavar="GROUP",
                        help=f"group whose members administer all jobs "
                             f"(default: the first of {', '.join(ADMIN_GROUPS)} "
                             f"that exists here)")
    parser.add_argument("--spec", default="hpc-batch", metavar="SPEC",
                        help="what to install, as pip would take it: a version "
                             "spec like 'hpc-batch==0.2.0', or a path to a "
                             "checkout (default: hpc-batch, from PyPI)")
    parser.add_argument("--no-start", action="store_true",
                        help="install everything but leave the service stopped")
    parser.add_argument("--already-installed", action="store_true",
                        help="skip the venv and pip step because the caller has "
                             "just done it (install.sh, which has to install the "
                             "package before this command exists to run)")
    parser.add_argument("--uninstall", action="store_true",
                        help="remove the unit, the entry points and the "
                             "virtualenv, keeping job state")
    parser.add_argument("--version", action="version", version=f"hpc-batch {__version__}")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        uninstall(args) if args.uninstall else install(args)
    except InstallError as exc:
        print(f"hpc-batch-install: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
