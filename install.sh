#!/bin/sh
# Bootstrap hpc-batch on a fresh machine.
#
#   curl -fsSL https://raw.githubusercontent.com/Yiannis128/hpc-batch/master/install.sh | sudo sh
#
# All this does is create the virtualenv and pull the package into it; the
# real work is hpc-batch-install, which ships inside the package. Once that
# exists, upgrades never come back here:
#
#   sudo /opt/bin/hpc-batch-install
#
# Environment: PREFIX (default /opt/hpc-batch), SPEC (default hpc-batch, but
# a path to a checkout works too). Everything else is an hpc-batch-install
# flag and can be passed straight through to this script.
set -eu

PREFIX=${PREFIX:-/opt/hpc-batch}
SPEC=${SPEC:-hpc-batch}

if [ "$(id -u)" -ne 0 ]; then
    echo "install.sh: must run as root (try: sudo sh install.sh)" >&2
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "install.sh: python3 not found" >&2
    exit 1
fi

if ! python3 -m venv "$PREFIX" 2>/dev/null; then
    echo "install.sh: could not create a virtualenv at $PREFIX." >&2
    echo "            On Debian/Ubuntu: apt install python3-venv" >&2
    exit 1
fi

"$PREFIX/bin/pip" install --quiet --upgrade "$SPEC"

# Hand over. It links the entry points into /opt/bin, adds that to PATH via
# /etc/profile.d, writes the unit with an admin group that exists here, and
# starts the daemon.
exec "$PREFIX/bin/hpc-batch-install" --prefix "$PREFIX" --spec "$SPEC" "$@"
