#!/bin/sh
# Remove hpc-batch from a machine, job data included.
#
#   curl -fsSL https://raw.githubusercontent.com/Yiannis128/hpc-batch/master/uninstall.sh | sudo sh
#
# This kills anything still running and deletes the state directory: queued
# jobs, history, and output not yet delivered. To keep all that, run the
# installer's own uninstall instead:
#
#   sudo /opt/bin/hpc-batch-install --uninstall
#
# Like install.sh, this is a bootstrap: the real work is hpc-batch-install
# --purge, which ships inside the package and knows where this install put
# things. Environment: PREFIX (default /opt/hpc-batch), BIN_DIR (only used if
# the package is too broken to run). Everything else is passed through.
set -eu

PREFIX=${PREFIX:-/opt/hpc-batch}

if [ "$(id -u)" -ne 0 ]; then
    echo "uninstall.sh: must run as root (try: sudo sh uninstall.sh)" >&2
    exit 1
fi

if [ -x "$PREFIX/bin/hpc-batch-install" ]; then
    exec "$PREFIX/bin/hpc-batch-install" --prefix "$PREFIX" --purge "$@"
fi

# Only the console script is gone, so run the same code through the interpreter
# rather than reaching for the sweep below.
if [ -x "$PREFIX/bin/python3" ]; then
    exec "$PREFIX/bin/python3" -m hpc_batch.install --prefix "$PREFIX" --purge "$@"
fi

# Fallback: the venv is gone or broken, which is a state people uninstall from.
# Everything below restates what install.py owns, so it sweeps the defaults and
# nothing else -- a --state-dir moved in a drop-in is left behind, because only
# the packaged uninstaller reads the unit.
echo "uninstall.sh: cannot run $PREFIX/bin/hpc-batch-install; sweeping the defaults" >&2
BIN_DIR=${BIN_DIR:-$(sed -n 's/.*"bin_dir": *"\([^"]*\)".*/\1/p' "$PREFIX/install.json" 2>/dev/null)}
BIN_DIR=${BIN_DIR:-/opt/bin}

# PREFIX is an environment variable and the next rm is recursive.
case "$PREFIX" in
    */*/*) ;;
    *) echo "uninstall.sh: refusing to remove $PREFIX: too close to /" >&2; exit 1 ;;
esac

systemctl disable --now hpc-batch 2>/dev/null || :
rm -f /etc/systemd/system/hpc-batch.service /etc/profile.d/hpc-batch.sh
rm -f "$BIN_DIR"/dispatch "$BIN_DIR"/hpc-batchd "$BIN_DIR"/hpc-batch-install
rmdir "$BIN_DIR" 2>/dev/null || :
systemctl daemon-reload 2>/dev/null || :

for cg in /sys/fs/cgroup/hpc-batch/job-*; do
    [ -d "$cg" ] || continue
    echo 1 > "$cg/cgroup.kill" 2>/dev/null || :
    rmdir "$cg" 2>/dev/null || :
done
rmdir /sys/fs/cgroup/hpc-batch 2>/dev/null || :

rm -rf "$PREFIX" /var/lib/hpc-batch /dev/hpc-batch
rm -f /run/hpc-batch/hpc-batch.sock
rmdir /run/hpc-batch 2>/dev/null || :

echo "removed hpc-batch, its entry points in $BIN_DIR, and /var/lib/hpc-batch"
