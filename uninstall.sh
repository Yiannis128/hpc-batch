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

# Fallback: the venv is gone or broken, which is a state people uninstall from.
# Defaults only -- the packaged uninstaller is the one that reads the unit for
# a moved --state-dir and install.json for a moved --bin-dir.
echo "uninstall.sh: $PREFIX/bin/hpc-batch-install is missing; sweeping the defaults" >&2
BIN_DIR=${BIN_DIR:-$(sed -n 's/.*"bin_dir": *"\([^"]*\)".*/\1/p' "$PREFIX/install.json" 2>/dev/null)}
BIN_DIR=${BIN_DIR:-/opt/bin}

systemctl disable --now hpc-batch 2>/dev/null || :
rm -f /etc/systemd/system/hpc-batch.service /etc/profile.d/hpc-batch.sh
for name in dispatch hpc-batchd hpc-batch-install; do
    rm -f "$BIN_DIR/$name"
done
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
