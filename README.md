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
- **GPUs**: `--gpu-cores N` allocates N of the GPUs enumerated by
  `nvidia-smi -L`; the job sees them via `CUDA_VISIBLE_DEVICES` (jobs that
  requested no GPUs get an empty `CUDA_VISIBLE_DEVICES`).
- **/dev/hpc-batch/jobs/**: every queued/running job appears as
  `/dev/hpc-batch/jobs/<id>` (a symlink to its state directory) containing
  `info.json` (metadata) and `output` (combined stdout/stderr). Entries are
  owned by the submitting user with the admin group as group owner.
- **Authentication**: the daemon identifies clients by `SO_PEERCRED` on the
  unix socket, so users cannot impersonate each other. Jobs are executed
  under the submitting user's uid/gid with a clean environment.
- **Hot reload**: `systemctl reload hpc-batch` makes the daemon persist its
  state and re-exec itself in place. Running jobs are *not* killed; they are
  re-adopted by the new daemon (pid-reuse is guarded by comparing
  `/proc/<pid>/stat` start times). The same applies to `systemctl restart`
  thanks to `KillMode=process` + `Delegate=` in the unit.

## Install

```sh
hatch build
pip install dist/hpc_batch-*.whl            # as root, or pipx install for a system tool
cp systemd/hpc-batch.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now hpc-batch
```

Adjust `ExecStart=` in the unit for the install location of `hpc-batchd`
(e.g. `/usr/local/bin/hpc-batchd` or a pipx path).

## Admin configuration

All admin parameters are arguments to `hpc-batchd`, configured in the
systemd unit's `ExecStart=` line:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--max-lifetime DURATION` | `1d` | Jobs running longer than this are killed. Also the upper bound and default for `--max-time`. |
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
dispatch new --cpu 2 --gpu-cores 3 --max-mem 84 --max-time 2h -- ./run_benchmark.sh --iterations 10

# Run alone on the machine (waits until idle, blocks others while running):
dispatch new --cpu 8 --exclusive -- ./timing_sensitive_bench

# List my jobs / all jobs (all = admins, or everyone with --list-is-public):
dispatch list
dispatch list --all

# Follow a job's output (admins can attach to any job):
dispatch attach 7

# Kill a running job or remove a queued one:
dispatch kill 7
```

`dispatch list` prints `<username> <id> <command> <uptime> <max-time>
<exclusive>`; queued jobs show `queued` in the uptime column.

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
