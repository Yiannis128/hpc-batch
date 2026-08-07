"""System-wide installer: `hpc-batch-install`.

`pip install hpc-batch` leaves two entry points in whichever environment ran
pip, which is not a working system. This puts the code in a venv nobody else
writes to, puts the client on every user's PATH, and writes a unit pointing
at the right binary with an admin group that exists on this distro.

It is also the upgrade path, so every step is idempotent: rerunning it
reinstalls the package and reloads the daemon rather than restarting it,
which is what keeps running jobs alive. What it chose is recorded in
`install.json` inside the prefix, so a bare upgrade repeats the first
install's layout instead of falling back to the defaults and leaving the
old entry points behind unmanaged.

Refusals follow the daemon's rule -- if something was asked for and cannot
be provided, say so and stop, rather than installing something that looks
finished and is not.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

from . import __version__
from .cgroup import CgroupManager
from .util import (
    ADMIN_GROUPS,
    DEFAULT_DEV_DIR,
    DEFAULT_SOCKET,
    DEFAULT_STATE_DIR,
    group_exists,
)

DEFAULT_PREFIX = Path("/opt/hpc-batch")
DEFAULT_BIN_DIR = Path("/opt/bin")
UNIT_PATH = Path("/etc/systemd/system/hpc-batch.service")
PROFILE_PATH = Path("/etc/profile.d/hpc-batch.sh")
ETC = Path("/etc")
RECORD_NAME = "install.json"
ENTRY_POINTS = ("dispatch", "hpc-batchd", "hpc-batch-install")

_PATH_ASSIGNMENT = re.compile(r"\bPATH\s*=")
_DIR_FLAG = re.compile(r"--(state-dir|dev-dir|socket)[=\s]+(\S+)")


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


def login_path_provides(
    bin_dir: Path, etc: Path = ETC, skip: Path | None = None
) -> Path | None:
    """Which login-shell config already puts `bin_dir` on PATH, if any.

    Reading the config rather than `$PATH` is the whole point: the installer
    runs under sudo, so its own environment is root's, and the question is
    what some other user's login shell will end up with. A false positive
    leaves nobody with `dispatch` on their PATH, so the caller prints the
    file this believed.
    """
    entry = re.compile(r"(?<![\w/])" + re.escape(str(bin_dir).rstrip("/")) + r"/?(?![\w/])")
    candidates = [etc / "environment", etc / "profile"]
    candidates += sorted((etc / "profile.d").glob("*.sh"))
    for path in candidates:
        if skip is not None and path == skip:
            continue
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        if any(_PATH_ASSIGNMENT.search(line) and entry.search(line) for line in lines):
            return path
    return None


def read_record(prefix: Path) -> dict:
    """What the last install of this prefix chose, or {} for a fresh one.

    Unreadable or corrupt reads as absent. The record exists to catch a
    contradiction, and there is none to catch when we cannot tell what was
    chosen; refusing on a damaged file would only block the reinstall that
    fixes it.
    """
    try:
        record = json.loads((prefix / RECORD_NAME).read_text())
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def write_record(prefix: Path, bin_dir: Path, admin_group: str) -> None:
    (prefix / RECORD_NAME).write_text(
        json.dumps({"bin_dir": str(bin_dir), "admin_group": admin_group}, indent=2) + "\n"
    )


def settle(record: dict, key: str, given: str | Path | None) -> str | None:
    """Reconcile one option against the last install of this prefix.

    Not passing the flag means "whatever last time did", so a bare upgrade
    cannot quietly move the entry points and leave the old ones behind, and
    an admin group that only appeared on the box later cannot silently take
    over. Passing something else is a refusal rather than a migration: the
    old install is still on disk and nothing here would move it. The flag
    name is derived from the key so a refusal cannot name the wrong option.
    """
    flag = "--" + key.replace("_", "-")
    remembered = record.get(key)
    if given is None:
        return remembered
    if remembered is not None and str(remembered) != str(given):
        raise InstallError(
            f"{flag} {given} disagrees with the {remembered} this prefix was "
            f"installed with; rerun without {flag} to keep {remembered}, or "
            f"'hpc-batch-install --uninstall' first and install again"
        )
    return str(given)


def bin_dirs_to_clean(record: dict, given: Path | None) -> set[Path]:
    """Everywhere uninstall should look for entry points.

    A union rather than a settlement: cleanup wants every place a symlink
    might be, and refusing a --bin-dir that disagrees with the record would
    leave behind exactly the split install it was passed to sweep up.
    """
    dirs = {Path(record["bin_dir"])} if record.get("bin_dir") else set()
    if given:
        dirs.add(given)
    return dirs or {DEFAULT_BIN_DIR}


class DaemonPaths(NamedTuple):
    state_dir: Path
    dev_dir: Path
    socket: Path


def daemon_paths(exec_start: str) -> DaemonPaths:
    """Where the daemon actually keeps its data.

    Read off its command line rather than assumed, because --state-dir and
    --dev-dir are exactly the sort of thing an admin moves: a purge that
    deletes the defaults would report the data gone while leaving it on disk.
    """
    found = {m.group(1): Path(m.group(2)) for m in _DIR_FLAG.finditer(exec_start)}
    return DaemonPaths(
        state_dir=found.get("state-dir", DEFAULT_STATE_DIR),
        dev_dir=found.get("dev-dir", DEFAULT_DEV_DIR),
        socket=found.get("socket", Path(DEFAULT_SOCKET)),
    )


def daemon_exec_start() -> str:
    """The daemon's command line as systemd will run it, drop-ins included.

    Ask systemd rather than read the unit: `systemctl edit` is the documented
    way to change a flag here, and the file we wrote knows nothing about it.
    Only valid before the unit is removed.
    """
    shown = subprocess.run(
        ["systemctl", "show", "hpc-batch", "--property=ExecStart", "--value"],
        capture_output=True, text=True, check=False,
    )
    return shown.stdout if shown.returncode == 0 else ""


def safe_to_remove(path: Path) -> bool:
    """Guard on paths that arrive from a hand-edited unit file: /var/lib/x is
    a data directory, /var is somebody's whole machine."""
    return path.is_absolute() and len(path.parts) > 2


def _rmdir_if_empty(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        pass


def _remove_data(path: Path, what: str) -> None:
    if not path.exists():
        return
    if not safe_to_remove(path):
        print(f"warning: refusing to remove {path} ({what}): too close to /")
        return
    shutil.rmtree(path, ignore_errors=True)
    # rmtree swallowed whatever went wrong, so look rather than announce.
    if path.exists():
        print(f"warning: could not fully remove {path} ({what})")
    else:
        print(f"removed {path} ({what})")


def purge(paths: DaemonPaths) -> None:
    """Remove everything the daemon owns at runtime: running jobs, the state
    directory, the inspection entries and the socket."""
    cgroups = CgroupManager()
    busy = cgroups.destroy()
    if busy:
        listed = ", ".join(p.name for p in busy)
        print(f"warning: could not remove {cgroups.root}: {listed} still busy")
    elif cgroups.root.exists():
        print(f"warning: {cgroups.root} is still there")

    _remove_data(paths.state_dir, "job state and output")
    _remove_data(paths.dev_dir, "job inspection entries")

    paths.socket.unlink(missing_ok=True)
    _rmdir_if_empty(paths.socket.parent)


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
    record = read_record(args.prefix)
    bin_dir = Path(settle(record, "bin_dir", args.bin_dir) or DEFAULT_BIN_DIR)
    admin_group = settle(record, "admin_group", args.admin_group) or detect_admin_group()

    if not args.already_installed:
        pip = _make_venv(args.prefix)
        _run([str(pip), "install", "--upgrade", args.spec], f"install {args.spec}")

    bin_dir.mkdir(parents=True, exist_ok=True)
    for name in ENTRY_POINTS:
        target = args.prefix / "bin" / name
        if not target.exists():
            raise InstallError(f"{target} was not installed by {args.spec}")
        link = bin_dir / name
        link.unlink(missing_ok=True)
        link.symlink_to(target)
    print(f"installed {', '.join(ENTRY_POINTS)} in {bin_dir}")

    provider = login_path_provides(bin_dir, skip=PROFILE_PATH)
    if provider is None:
        PROFILE_PATH.write_text(profile_snippet(bin_dir))
        PROFILE_PATH.chmod(0o644)
        print(f"wrote {PROFILE_PATH} (login shells pick it up; this one will not)")
    else:
        verb = "removed" if PROFILE_PATH.exists() else "skipped"
        PROFILE_PATH.unlink(missing_ok=True)
        print(f"{verb} {PROFILE_PATH}: {provider} already puts {bin_dir} on PATH")

    unit = render_unit(unit_template(), bin_dir / "hpc-batchd", admin_group)
    UNIT_PATH.write_text(unit)
    UNIT_PATH.chmod(0o644)
    print(f"wrote {UNIT_PATH} (admin group: {admin_group})")

    write_record(args.prefix, bin_dir, admin_group)

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
    bin_dirs = bin_dirs_to_clean(read_record(args.prefix), args.bin_dir)
    # While the unit is still there to be asked; --purge deletes what it says.
    paths = daemon_paths(daemon_exec_start())

    subprocess.run(["systemctl", "disable", "--now", "hpc-batch"], check=False)
    for path in (UNIT_PATH, PROFILE_PATH):
        path.unlink(missing_ok=True)
        print(f"removed {path}")
    for bin_dir in sorted(bin_dirs):
        for name in ENTRY_POINTS:
            (bin_dir / name).unlink(missing_ok=True)
        _rmdir_if_empty(bin_dir)
    shutil.rmtree(args.prefix, ignore_errors=True)
    _run(["systemctl", "daemon-reload"], "reload the systemd manager")
    listed = ", ".join(str(d) for d in sorted(bin_dirs))
    print(f"removed {args.prefix} and the entry points in {listed}")

    if args.purge:
        purge(paths)
        print("purged: no job history, queued jobs or undelivered output left")
    else:
        print(f"left {paths.state_dir} alone: it holds job history and "
              f"queued jobs, and jobs still running were not killed")
        print("pass --purge to remove those too")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hpc-batch-install",
        description="Install hpc-batch system-wide: a private virtualenv, the "
                    "client on every user's PATH, and the systemd unit. Rerun "
                    "it to upgrade; running jobs are not interrupted.",
    )
    parser.add_argument("--prefix", type=Path, default=DEFAULT_PREFIX, metavar="DIR",
                        help=f"virtualenv to install into (default: {DEFAULT_PREFIX})")
    parser.add_argument("--bin-dir", type=Path, default=None, metavar="DIR",
                        help=f"where the entry points are linked, and what is "
                             f"added to PATH (default: whatever the last "
                             f"install used, else {DEFAULT_BIN_DIR})")
    parser.add_argument("--admin-group", default=None, metavar="GROUP",
                        help=f"group whose members administer all jobs "
                             f"(default: whatever the last install used, else "
                             f"the first of {', '.join(ADMIN_GROUPS)} that "
                             f"exists here)")
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
    parser.add_argument("--purge", action="store_true",
                        help="uninstall, and also kill any job still running "
                             "and delete the daemon's data: the state "
                             "directory (queued jobs, history, output not yet "
                             "delivered), the inspection entries, the socket "
                             "and the job cgroups")
    parser.add_argument("--version", action="version", version=f"hpc-batch {__version__}")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        uninstall(args) if args.uninstall or args.purge else install(args)
    except InstallError as exc:
        print(f"hpc-batch-install: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
