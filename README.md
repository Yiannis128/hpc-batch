# hpc-batch

A single-node batch job system for shared HPC/benchmark machines. A root
daemon (`hpc-batchd`, run from systemd) accepts job submissions from users
via the `dispatch` CLI, queues them FIFO, and runs each one in its own
cgroup with the CPUs pinned to a single NUMA node — so memory stays local
and benchmark timings are stable.

## How it works

- **FIFO queue with pluggable scheduling** (`--schedule`, see below): jobs
  are always considered in submission order; the policy decides whether a
  later job may fill resources a blocked job cannot use.
- **cgroups v2**: each job runs in its own cgroup under the daemon's
  delegated subtree with `cpuset.cpus`, `cpuset.mems` (same NUMA node as
  the allocated CPUs) and `memory.max` applied. Swap is disabled for every
  job (`memory.swap.max=0`, reinforced by `MemorySwapMax=0` in the unit):
  `--max-mem` is a hard RAM budget, and a job that exceeds it is
  OOM-killed as a whole group instead of thrashing in swap.
- **Every job has a memory budget, whether or not it asks for one.** An
  unbounded job would be invisible to the scheduler and could starve
  everything sharing its node, so a job that omits `--max-mem` gets the
  share of a NUMA node its cores represent — 4 of 32 cores on a 256 GiB
  node is 32 GiB — floored at `--min-job-mem`. An `--exclusive` job gets the
  whole machine. `dispatch new` prints the budget it was given.
- **Memory is budgeted per NUMA node**, not machine-wide, because
  `cpuset.mems` confines a job to the nodes it was charged for. Normally
  that is one node and the job gets purely local memory. A budget that fits
  no single node is spread across nodes instead of waiting; the extra nodes
  go into `cpuset.mems`, so the job can never allocate where nobody
  budgeted for it. Access to the remote part is slower, so `dispatch list`
  marks those jobs with `+` and `--numa-local` refuses to spread at all,
  waiting for a node that fits the whole budget. When the cpuset controller
  is unavailable (undelegated, or `--no-cgroups`) a job really can allocate
  anywhere, and the daemon tracks one machine-wide pool instead; the
  startup log line says which mode is in force.
- **GPUs**: `--gpu-cores N` allocates N of the GPUs enumerated by
  `nvidia-smi -L`; the job sees them via `CUDA_VISIBLE_DEVICES` (jobs that
  requested no GPUs get an empty `CUDA_VISIBLE_DEVICES`).
- **/dev/hpc-batch/jobs/**: every queued/running job appears as
  `/dev/hpc-batch/jobs/<id>` (a symlink to its state directory) containing
  `info.json` (metadata) and `output` (combined stdout/stderr). Entries are
  owned by the submitting user with the admin group as group owner.
- **Output you keep**: while a job runs its combined stdout/stderr is
  buffered in the state directory so `dispatch attach` has something to
  stream. When it finishes that buffer is copied to `output.<id>.log` in the
  directory you submitted from and then dropped, so the daemon never stores
  your results: the only lasting copy is yours, on your own storage.
  `--output` picks a different directory or filename, `--no-output` discards
  the buffer instead of copying it (attach still works while the job runs).
  The destination is checked for writability at submit time, so a bad path
  is rejected immediately rather than after the job has run, and the copy is
  written as *you*, never as root. If the copy fails the buffer is kept so
  nothing is lost, and the error is reported by `dispatch list --finished`.
- **Authentication**: the daemon identifies clients by `SO_PEERCRED` on the
  unix socket, so users cannot impersonate each other. Jobs are executed
  under the submitting user's uid/gid with a clean environment.
- **Hot reload**: `systemctl reload hpc-batch` makes the daemon persist its
  state and re-exec itself in place. Running jobs are *not* killed; they are
  re-adopted by the new daemon (pid-reuse is guarded by comparing
  `/proc/<pid>/stat` start times). The same applies to `systemctl restart`
  thanks to `KillMode=process` + `Delegate=` in the unit.

## Install

Needs Python >= 3.10, cgroups v2, and root (systemd) for the daemon. The
whole sequence, copy/paste:

```sh
# 1. Source, and build a wheel as a normal user (not root, so no
#    root-owned build artifacts end up in the checkout).
git clone https://github.com/Yiannis128/hpc-batch.git ~/projects/hpc-batch
cd ~/projects/hpc-batch
rm -rf dist
python3 -m venv /tmp/hb-build
/tmp/hb-build/bin/pip wheel -q --no-deps -w dist .

# 2. Install into a dedicated venv and expose both entry points.
sudo python3 -m venv /opt/hpc-batch
sudo /opt/hpc-batch/bin/pip install -q dist/hpc_batch-*.whl
sudo ln -sf /opt/hpc-batch/bin/dispatch   /usr/local/bin/dispatch
sudo ln -sf /opt/hpc-batch/bin/hpc-batchd /usr/local/bin/hpc-batchd

# 3. Install the unit. --admin-group must name a group that exists:
#    "sudo" on Debian/Ubuntu, "wheel" on Fedora/RHEL (the shipped default).
sudo cp systemd/hpc-batch.service /etc/systemd/system/
sudo sed -i 's/--admin-group wheel/--admin-group sudo/' /etc/systemd/system/hpc-batch.service
sudo systemctl daemon-reload
sudo systemctl enable --now hpc-batch

# 4. Verify.
systemctl status hpc-batch --no-pager
dispatch new -- echo hello
dispatch list --finished
```

`/opt/hpc-batch` is world-readable, so every user on the machine can run
`dispatch`. Do **not** install with `sudo pipx`: it puts the entry points in
`/root/.local/bin` (mode 700), where `hpc-batchd` still works but no other
user can run the client.

Adjust the admin parameters on the `ExecStart=` line to taste (see
[Admin configuration](#admin-configuration)); they are all daemon arguments.

## Update

```sh
cd ~/projects/hpc-batch
git pull
rm -rf dist
python3 -m venv /tmp/hb-build
/tmp/hb-build/bin/pip wheel -q --no-deps -w dist .
sudo /opt/hpc-batch/bin/pip install -q --force-reinstall --no-deps dist/hpc_batch-*.whl
sudo systemctl reload hpc-batch
```

Use `reload`, not `restart`: reload makes the daemon persist its state and
re-exec in place, so running jobs are re-adopted instead of killed. If the
unit file itself changed, `sudo cp` it again and `systemctl daemon-reload`
before reloading (re-applying the `--admin-group` edit from step 3).

## Admin configuration

All admin parameters are arguments to `hpc-batchd`, configured in the
systemd unit's `ExecStart=` line:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--max-lifetime DURATION` | `1d` | Jobs running longer than this are killed. Also the upper bound and default for `--max-time`. |
| `--reserve-cpu N` | `2` | Cores held back for the OS and the daemon, never offered to jobs. Taken from the lowest-numbered NUMA node so the other nodes keep their full width. This is headroom, not a fence: nothing pins the OS to them. |
| `--reserve-mem GB` | `2` | Memory held back for the OS and the daemon, off the same node as `--reserve-cpu`, and never more than half of any one node. |
| `--min-job-mem GB` | `2` | Floor for an automatically assigned budget, so a single-core job on a core-dense machine is not left with a sliver. Ignored when the user passes `--max-mem`. |
| `--keep-finished N` | `50` | Finished jobs remembered **per user** for `dispatch list --finished`. Metadata only; output is never kept here. |
| `--list-is-public` | off | Allow non-admins to use `dispatch list --all`. |
| `--admin-group GROUP` | `wheel` | Members can list, attach to and kill any user's jobs. |
| `--socket PATH` | `/run/hpc-batch/hpc-batch.sock` | Unix socket the daemon listens on. |
| `--state-dir DIR` | `/var/lib/hpc-batch` | Job state, metadata and output. |
| `--dev-dir DIR` | `/dev/hpc-batch` | Where job inspection entries appear. |
| `--schedule POLICY` | `fifo-strict` | Scheduling policy (see below). |
| `--no-cgroups` | off | Development mode: skip cgroups, pin CPUs with `sched_setaffinity` only. |

Durations accept plain seconds or `s`/`m`/`h`/`d` suffixes, e.g. `45m`,
`2h`, `1h30m`.

After changing arguments: `systemctl daemon-reload && systemctl reload
hpc-batch` — running jobs survive the reload.

### Scheduling policies

Jobs are always ranked in submission (FIFO) order. The policy only decides
whether a job further back may use resources the blocked head-of-queue job
cannot yet use. Take the queue of GPU demands `[1, 4, 2, 1, 2]` (jobs 1..5)
on a 4-GPU machine as the running example.

- **`fifo-strict`** (default) — head-of-line blocking. Start jobs in order;
  stop at the first that does not fit. Nothing ever jumps the queue.
  *Example:* only job 1 runs; job 2 (needs 4) blocks jobs 3–5 behind it.
  Simplest and most predictable, but a big job leaves resources idle.

- **`easy-backfill`** — when the head job first cannot fit, freeze a
  *backfill budget* equal to the resources idle at that moment. Later jobs
  (including ones submitted afterwards) may start while they fit within that
  budget. Resources freed later by finishing jobs are **not** added to the
  budget — they are held for the head, so once the machine fills up nothing
  jumps ahead of it. No runtime estimates needed. *Example:* jobs 1, 3 and 4
  run (1+2+1 = 4 GPUs); job 2 is reserved; after a job finishes its GPUs are
  held until all 4 are free and job 2 runs. This is the behavior most people
  expect from "backfill".

- **`strict-backfill`** — like easy-backfill, but uses each job's
  `--max-time` to reserve a guaranteed start time for the head (assuming
  running jobs occupy their resources until their deadline). A later job may
  backfill past the head only if it provably finishes before that reserved
  start, so it can never delay the head. This keeps more resources busy than
  easy-backfill while still protecting the head. *Example:* a short job may
  backfill into idle GPUs even after job 2 blocks, but a long-running job
  that would still be running when job 2 is due is refused.

None of the policies starve the head: every job has a bounded lifetime
(`--max-lifetime`), so a blocked job always eventually runs.

## Usage

```sh
# Submit: everything after "--" is the job's command line.
# Output is saved to ./output.<id>.log when the job finishes.
dispatch new --cpu 2 --gpu-cores 3 --max-mem 84 --max-time 2h -- ./run_benchmark.sh --iterations 10

# Save the output somewhere else, or not at all:
dispatch new --output ~/results -- ./run_benchmark.sh      # ~/results/output.<id>.log
dispatch new --output ~/results/run1.log -- ./run_benchmark.sh
dispatch new --no-output -- ./noisy_job.sh

# Run alone on the machine (waits until idle, blocks others while running).
# Without --cpu this takes every core and all the memory:
dispatch new --exclusive -- ./timing_sensitive_bench

# Insist on memory local to the job's own NUMA node, waiting for a node that
# fits rather than spreading the budget across nodes:
dispatch new --cpu 4 --max-mem 64 --numa-local -- ./latency_sensitive_bench

# List my jobs / all jobs (all = admins, or everyone with --list-is-public):
dispatch list
dispatch list --all

# Include recently finished jobs, with their exit status:
dispatch list --finished

# Follow a job's output (admins can attach to any job):
dispatch attach 7

# Kill a running job or remove a queued one:
dispatch kill 7
```

`dispatch list` prints `<username> <id> <command> <start> <uptime> <max-time>
<mem> <exclusive>`; the start column is the local clock time the job began,
shown as `HH:MM:SS` for a job that started today and `MM-DD HH:MM` for an older
one. Queued jobs have not started, so they show `-` there and `queued` in the
uptime column. A `+` after the memory figure means that budget had to be spread
over more than one NUMA node. By default only queued and running jobs are
listed. `--finished` also includes finished ones and appends `<state> <exit>`,
where the exit column shows the exit code, or `killed`/`timeout`/`error`/`oom`
when the job did not exit on its own. `oom` means the job was killed for
exceeding its memory budget, and the listing says whether that budget was one
you asked for or one assigned by default.

`dispatch attach` follows a running job. Once a job has finished its buffered
output is gone, so read the `output.<id>.log` it left behind instead; attaching
to a finished job just tells you where that file is.

The socket path for the client can be overridden with `$HPC_BATCH_SOCKET`
(useful with a non-default `--socket`).

## Development

No root required: run the daemon in user mode against scratch paths.

```sh
hatch run test    # unit tests

# manual smoke test
S=$(mktemp -d)
python -m hpc_batch.daemon --no-cgroups --socket "$S/sock" \
    --state-dir "$S/state" --dev-dir "$S/dev" --max-lifetime 1h &
export HPC_BATCH_SOCKET="$S/sock"
dispatch new -- echo hello
dispatch list
```

In user mode the daemon only accepts jobs from its own uid (it cannot
setuid) and falls back from cgroups to CPU affinity pinning.
