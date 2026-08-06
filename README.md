# hpc-batch

[![CI](https://github.com/Yiannis128/hpc-batch/actions/workflows/ci.yml/badge.svg)](https://github.com/Yiannis128/hpc-batch/actions/workflows/ci.yml)

A single-node batch job system for shared HPC/benchmark machines. A root
daemon (`hpc-batchd`, run from systemd) accepts job submissions from users
via the `dispatch` CLI, queues them FIFO, and runs each one in its own
cgroup with the CPUs pinned to a single NUMA node — so memory stays local
and benchmark timings are stable.

## How it works

- **FIFO queue with pluggable scheduling** (`--schedule`, see below): jobs
  are always considered in submission order; the policy decides whether a
  later job may fill resources a blocked job cannot use.
- **cgroups v2**: each job runs in its own cgroup under
  `/sys/fs/cgroup/hpc-batch` with `cpuset.cpus`, `cpuset.mems` (same NUMA
  node as the allocated CPUs) and `memory.max` applied. That tree is
  deliberately not the daemon's service cgroup, so restarting the unit never
  disturbs a running job; the unit still needs `Delegate=cpuset memory pids`
  so those controllers exist at the cgroup root, and without it the daemon
  refuses to start rather than run without NUMA isolation. Swap is disabled
  for every job (`memory.swap.max=0`):
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
  waiting for a node that fits the whole budget. Under `--no-cgroups` a job
  really can allocate anywhere, and the daemon tracks one machine-wide pool
  instead; the startup log line says which mode is in force. Without that
  flag an unavailable cpuset controller is a refusal, not a downgrade.
- **GPUs**: `--gpu-cores N` allocates N of the GPUs enumerated by
  `nvidia-smi -L`; the job sees them via `CUDA_VISIBLE_DEVICES` (jobs that
  requested no GPUs get an empty `CUDA_VISIBLE_DEVICES`).
- **Which GPUs is not arbitrary.** A multi-GPU job gets the
  closest-connected free set in the interconnect that `nvidia-smi topo -m`
  reports, not the lowest free indices: with GPU1 busy, index order hands a
  2-GPU job GPU0+GPU2 across the machine even when GPU2+GPU3 share an
  NVLink. A set is judged by its worst link first, because a collective runs
  at the speed of its slowest pair, and the job's cpus then come from the
  NUMA node its GPUs hang off. Adjacency is a preference and never a reason
  to keep a job queued: when only distant GPUs are free the job takes them,
  and the line logged at start-up names the link class it ended up with.
  Where `nvidia-smi topo -m` is unavailable the daemon says so at startup and
  falls back to index order.
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
  under the submitting user's uid/gid.
- **Environment**: a job gets a clean environment (`PATH`, `HOME`, `USER`,
  `SHELL`, `LANG`) unless you pass `--env`, which forwards the one you
  submitted from. `HPC_BATCH_JOB_ID` and `CUDA_VISIBLE_DEVICES` are always
  set by the daemon and a forwarded value never wins: the latter is how a
  job is held to the gpus it was allocated. A forwarded environment is
  stored with the job so a queued one survives a daemon restart, which is
  why the state file is root-only; it is dropped once the job starts.
- **Hot reload**: `systemctl reload hpc-batch` makes the daemon persist its
  state and re-exec itself in place. Running jobs are *not* killed; they are
  re-adopted by the new daemon (pid-reuse is guarded by comparing
  `/proc/<pid>/stat` start times). The same applies to `systemctl restart`,
  thanks to `KillMode=process` in the unit and to job cgroups living outside
  it. Reload is still the better habit: it re-execs the same pid, so jobs
  stay its children and their exit codes are still collected, where a full
  restart can only see that they ended.

## Install

Needs Python >= 3.10, systemd, cgroups v2, and root for the daemon.

```sh
curl -fsSL https://raw.githubusercontent.com/Yiannis128/hpc-batch/master/install.sh | sudo sh
```

That creates a virtualenv at `/opt/hpc-batch`, installs `hpc-batch` from
PyPI into it, links `dispatch`, `hpc-batchd` and `hpc-batch-install` into
`/opt/bin`, adds `/opt/bin` to everyone's `PATH` via
`/etc/profile.d/hpc-batch.sh`, writes the systemd unit with an admin group
that exists on this distro (`wheel`, `sudo`, `adm`, whichever it finds
first), and starts the daemon.

`PATH` only changes for *new* login shells, so in the one you ran it from
use `/opt/bin/dispatch` or log in again. Then:

```sh
systemctl status hpc-batch --no-pager
dispatch new -- echo hello
dispatch list --finished
```

If you would rather not pipe a script into a shell, the same thing in two
steps:

```sh
sudo python3 -m venv /opt/hpc-batch
sudo /opt/hpc-batch/bin/pip install hpc-batch
sudo /opt/hpc-batch/bin/hpc-batch-install
```

`hpc-batch-install --help` covers the rest: `--prefix`, `--bin-dir`,
`--admin-group`, `--spec` (install a specific version, or a checkout), and
`--uninstall`, which removes everything except `/var/lib/hpc-batch` so job
history and queued jobs survive.

Do **not** install with `sudo pipx`: it puts the entry points in
`/root/.local/bin` (mode 700), where `hpc-batchd` still works but no other
user can run the client.

## Update

```sh
sudo /opt/bin/hpc-batch-install
```

Reinstalls from PyPI and *reloads* the daemon rather than restarting it, so
running jobs are re-adopted instead of losing their exit codes. The unit is
rewritten each time, so keep local changes in a drop-in (`systemctl edit
hpc-batch`) rather than editing the file in place.

## Refusals

The daemon does not start half-configured. If it was asked for isolation it
cannot deliver, it says which piece is missing and exits 78 (`EX_CONFIG`),
which the unit's `RestartPreventExitStatus=` turns into a clean stop rather
than a restart loop over the error message:

- cgroups v2 not mounted, or the daemon is not root
- the `cpuset` or `memory` controller is not available in its cgroup, which
  almost always means `Delegate=` was removed from the unit
- `--admin-group` names a group that does not exist here (it suggests ones
  that do)
- `--dev-dir` cannot be created

Each refusal names the flag that opts out of the thing it is refusing over
(`--no-cgroups`, `--no-dev-dir`), for the cases where you genuinely want to
run without it. `--no-cgroups` in particular means cpu-affinity pinning
only, with no enforced memory limit and no NUMA confinement: fine for
development on a laptop, not for a shared machine.

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
| `--admin-group GROUP` | `wheel` | Members can list, attach to and kill any user's jobs. The daemon refuses to start if the group does not exist; `hpc-batch-install` fills in one that does. |
| `--socket PATH` | `/run/hpc-batch/hpc-batch.sock` | Unix socket the daemon listens on. |
| `--state-dir DIR` | `/var/lib/hpc-batch` | Job state, metadata and output. |
| `--dev-dir DIR` | `/dev/hpc-batch` | Where job inspection entries appear. |
| `--schedule POLICY` | `fifo-strict` | Scheduling policy (see below). |
| `--no-cgroups` | off | Development mode: skip cgroups, pin CPUs with `sched_setaffinity` only, and enforce no memory limit. Without it, unavailable cgroups are a refusal to start. |
| `--no-dev-dir` | off | Skip the per-job inspection entries under `--dev-dir` instead of refusing to start when they cannot be created. |

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

# Pass your current environment through instead of getting a clean one.
# Drop anything you would rather not send in the same breath:
dispatch new --env -- ./run_benchmark.sh
HF_TOKEN= dispatch new --env -- ./run_benchmark.sh

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
shown as `HH:MM:SS` for a job that started today and `YYYY-MM-DD HH:MM` for an
older one. Queued jobs have not started, so they show `-` there and `queued` in
the uptime column. A `+` after the memory figure means that budget had to be
spread over more than one NUMA node. By default only queued and running jobs
are listed. `--finished` also includes finished ones and appends `<state>
<exit>`, where the exit column shows the exit code, or
`killed`/`timeout`/`error`/`oom` when the job did not exit on its own. `oom`
means the job was killed for exceeding its memory budget, and the listing says
whether that budget was one you asked for or one assigned by default.

`dispatch attach` follows a running job. Once a job has finished its buffered
output is gone, so read the `output.<id>.log` it left behind instead; attaching
to a finished job just tells you where that file is.

The socket path for the client can be overridden with `$HPC_BATCH_SOCKET`
(useful with a non-default `--socket`).

## Releasing

Publishing a GitHub release is what triggers a release; pushing a tag on its
own does nothing, so a mistagged commit costs nothing.

```sh
# 1. Bump the version. It lives in exactly one place.
vim hpc_batch/__init__.py          # __version__ = "0.2.0"
git commit -am "Release 0.2.0" && git push

# 2. Tag and publish. `gh release create` does both.
gh release create v0.2.0 --generate-notes
```

The workflow then checks the tag matches `__version__`, runs the tests,
builds the sdist and wheel, attaches them to the GitHub release, and
publishes to PyPI. A tag that disagrees with the package version fails
before anything is published: PyPI reads the version from the metadata and
GitHub reads it from the tag, and nothing else compares them.

### One-time PyPI setup

The workflow authenticates with [trusted
publishing](https://docs.pypi.org/trusted-publishers/), so there is no API
token to store, leak or rotate. PyPI verifies a short-lived OIDC token that
GitHub mints for this repository, and only for the workflow named below.

Before the first release, at
<https://pypi.org/manage/account/publishing/>, add a *pending* publisher
(the project need not exist yet) with exactly these values:

| Field | Value |
| --- | --- |
| PyPI project name | `hpc-batch` |
| Owner | `Yiannis128` |
| Repository name | `hpc-batch` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

The environment name has to match `environment: pypi` in
`.github/workflows/release.yml`. GitHub creates that environment on first
use; adding required reviewers to it under Settings → Environments turns
publishing into something that needs an explicit approval.

No repository secrets are involved. If you would rather use an API token
anyway, put it in a `PYPI_API_TOKEN` secret and give the publish step
`with: password: ${{ secrets.PYPI_API_TOKEN }}` — but then it is a
long-lived credential with upload rights, which is the thing trusted
publishing exists to avoid.

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
setuid), and `--no-cgroups` is required: without it the daemon refuses to
start rather than run a shared machine's worth of jobs unisolated.

CI runs the suite on Python 3.10 through 3.13, builds both artefacts, and
checks that the wheel still carries the systemd unit and that the installed
entry points refuse a non-root run.
