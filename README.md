# hpc-batch

[![CI](https://github.com/Yiannis128/hpc-batch/actions/workflows/ci.yml/badge.svg)](https://github.com/Yiannis128/hpc-batch/actions/workflows/ci.yml)

A batch queue for one shared machine: the box with the GPUs that four people
want at once. A root daemon (`hpc-batchd`, run from systemd) takes job
submissions over a unix socket, queues them in submission order, and runs
each one as the submitting user in its own cgroup.

The point is that a job's timings mean something. Its CPUs come from a single
NUMA node, its memory budget is charged to that node, and its GPUs are the
closest-connected set free at the time rather than the lowest free indices.
Nobody else's job lands on top of it while it runs.

Python >= 3.10, no dependencies outside the standard library. Linux only:
it needs cgroups v2 and systemd.

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/Yiannis128/hpc-batch/master/install.sh | sudo sh
```

That creates a virtualenv at `/opt/hpc-batch`, installs `hpc-batch` from PyPI
into it, links `dispatch`, `hpc-batchd` and `hpc-batch-install` into
`/opt/bin`, adds `/opt/bin` to everyone's `PATH` via
`/etc/profile.d/hpc-batch.sh`, writes the systemd unit with the first admin
group that exists on this distro (`wheel`, `sudo`, `adm`, `staff`), and
starts the daemon.

`PATH` only changes for *new* login shells, so in the one you ran it from use
`/opt/bin/dispatch` or log in again. Then:

```sh
systemctl status hpc-batch --no-pager
dispatch new -- echo hello
dispatch list --finished
```

The same thing in two steps, if you would rather not pipe a script into a
shell:

```sh
sudo python3 -m venv /opt/hpc-batch
sudo /opt/hpc-batch/bin/pip install hpc-batch
sudo /opt/hpc-batch/bin/hpc-batch-install
```

`hpc-batch-install --help` covers the rest: `--prefix`, `--bin-dir`,
`--admin-group`, `--spec` (a specific version, or a checkout) and
`--uninstall`, which removes everything except `/var/lib/hpc-batch` so job
history and queued jobs survive.

Do **not** install with `sudo pipx`. It puts the entry points in
`/root/.local/bin` (mode 700), where `hpc-batchd` still works but no other
user can run the client.

### Update

```sh
sudo /opt/bin/hpc-batch-install
```

Reinstalls from PyPI and *reloads* the daemon rather than restarting it, so
running jobs are re-adopted instead of losing their exit codes. The unit is
rewritten each time, so keep local changes in a drop-in (`systemctl edit
hpc-batch`) rather than editing the file in place.

## Submitting jobs

Everything after `--` is the job's command line.

```sh
dispatch new --cpu 2 --gpu-cores 3 --max-mem 84 --max-time 2h -- ./bench.sh --iters 10
```

`dispatch new` prints the id, the cores and the memory budget the job was
given, because with `--exclusive` or without `--max-mem` the daemon picks
those and this is your only chance to notice before a job is killed for
exceeding a limit it never named.

| Flag | Meaning |
| --- | --- |
| `--cpu N` | Cores to allocate, all from one NUMA node. Default 1, or every core with `--exclusive`. |
| `--gpu-cores N` | How many of the GPUs `nvidia-smi -L` enumerates. Default 0. |
| `--max-mem GB` | Hard RAM budget. Default is the share of a NUMA node your cores represent. |
| `--max-time DURATION` | Kill the job after this long, e.g. `30m`, `2h`, `1h30m`. Default and ceiling: the admin's `--max-lifetime`. |
| `--exclusive` | Run alone: wait for an idle machine, block others while running. |
| `--numa-local` | Keep the whole job on one node. Implies both flags below. |
| `--numa-local-mem` | Wait for a node that fits the whole memory budget instead of spreading it. |
| `--numa-local-gpu` | Wait until every GPU hangs off one node, and take the CPUs from that node. |
| `--gpu-link CLASS` | Refuse GPUs wired worse than this to each other, waiting instead. |
| `--env` | Forward your current environment instead of getting a clean one. |
| `--output PATH` | Where to save the output: a directory, or an exact file path. Default: the current directory. |
| `--no-output` | Discard the output when the job ends. |

```sh
# Save the output somewhere else, or not at all:
dispatch new --output ~/results -- ./bench.sh          # ~/results/output.<id>.log
dispatch new --output ~/results/run1.log -- ./bench.sh
dispatch new --no-output -- ./noisy_job.sh

# Pass your environment through, dropping anything you would rather not send:
dispatch new --env -- ./bench.sh
HF_TOKEN= dispatch new --env -- ./bench.sh

# Have the machine to yourself. Without --cpu this takes every core and all
# the memory:
dispatch new --exclusive -- ./timing_sensitive_bench

# Insist on memory local to the job's own node, waiting for one that fits
# rather than spreading the budget:
dispatch new --cpu 4 --max-mem 64 --numa-local-mem -- ./latency_bench

# Keep the whole job on one node, memory and gpus alike:
dispatch new --cpu 8 --gpu-cores 2 --max-mem 64 --numa-local -- ./collective_bench

# Measuring collectives: wait for gpus that share an NVLink rather than run on
# whatever is free and get numbers paced by PCIe:
dispatch new --cpu 8 --gpu-cores 4 --gpu-link NV -- ./allreduce_bench
```

### Watching them

```sh
dispatch list                # my queued and running jobs
dispatch list --all          # everyone's (admins, or anyone with --list-is-public)
dispatch list --finished     # also recently finished ones, with exit status
dispatch attach 7            # follow a job's output, like tail -f
dispatch kill 7              # kill a running job, or drop a queued one
```

```
USER   ID  COMMAND            START     UPTIME  MAX-TIME  MEM    GPU     EXCLUSIVE
alice  7   ./allreduce_bench  14:02:11  6m21s   2h        64G    4 NV    no
bob    8   ./train.py         14:05:40  2m52s   1d        192G+  2 PHB!  no
carol  9   ./sweep.sh         -         queued  30m       16G    -       no
```

- **START** is the local clock time the job began: `HH:MM:SS` for one that
  started today, `YYYY-MM-DD HH:MM` for an older one, `-` for a queued job.
- **MEM `+`** means the budget did not fit one NUMA node and was spread over
  several, so part of it is slower to reach.
- **GPU** gives the count and the link class pacing them, `!` marking a set
  that reaches across a host bridge or the socket interconnect. The column
  only appears when some job in the listing has GPUs.
- **EXIT** (with `--finished`) is the exit code, or
  `killed`/`timeout`/`error`/`oom`. For `oom` the listing also says whether
  the budget it exceeded was one you asked for or one assigned by default.

`dispatch attach` follows a running job. Once a job finishes its buffered
output is gone, so read the `output.<id>.log` it left behind instead;
attaching to a finished job just tells you where that file is.

`$HPC_BATCH_SOCKET` overrides the socket path, for a daemon started with a
non-default `--socket`.

### Output you keep

While a job runs, its combined stdout/stderr is buffered in the daemon's
state directory so `dispatch attach` has something to stream. When the job
finishes that buffer is copied to `output.<id>.log` in the directory you
submitted from and then dropped, so the daemon never stores your results: the
only lasting copy is yours, on your own storage.

The destination is checked for writability at submit time, so a bad path is
rejected immediately rather than after the job has run, and the copy is
written as *you*, never as root. If it fails the buffer is kept so nothing is
lost, and `dispatch list --finished` reports the error.

### Inspecting a job

Every queued or running job appears as `/dev/hpc-batch/jobs/<id>`, a symlink
to its state directory holding `info.json` (metadata) and `output` (the live
buffer). Entries are owned by the submitting user with the admin group as
group owner.

### The environment a job gets

A clean one: `PATH` and `LANG`, plus `HOME`, `USER`, `LOGNAME`, `SHELL` and
`HPC_BATCH_JOB_ID` for the submitting user. `--env` forwards the environment
you submitted from, but the daemon's own variables still win.

On a machine with GPUs, `CUDA_VISIBLE_DEVICES` is always set and a forwarded
value never wins: it is the whole of a job's GPU isolation. A job that asked
for no GPUs gets it empty.

A forwarded environment is stored with the job so a queued one survives a
daemon restart, which is why the state file is root-only. It is dropped once
the job starts.

## How resources are allocated

### CPUs and cgroups

Each job runs in its own cgroup under `/sys/fs/cgroup/hpc-batch` with
`cpuset.cpus`, `cpuset.mems` and `memory.max` applied. That tree is
deliberately not the daemon's service cgroup, so restarting the unit never
disturbs a running job. The unit needs `Delegate=cpuset memory pids` for
those controllers to exist at the cgroup root, and without it the daemon
refuses to start rather than run without NUMA isolation.

Swap is off for every job (`memory.swap.max=0`): `--max-mem` is a hard RAM
budget, and a job that exceeds it is OOM-killed as a whole group instead of
thrashing in swap.

### Memory

**Every job has a budget, whether or not it asks for one.** An unbounded job
would be invisible to the scheduler and could starve everything sharing its
node, so a job that omits `--max-mem` gets the share of a NUMA node its cores
represent (4 of 32 cores on a 256 GiB node is 32 GiB), floored at the admin's
`--min-job-mem`. An `--exclusive` job gets the whole machine.

**Budgets are charged per NUMA node**, not machine-wide, because
`cpuset.mems` confines a job to the nodes it was charged for. Normally that
is one node and the job gets purely local memory. A budget that fits no
single node is spread across nodes rather than made to wait; the extra nodes
go into `cpuset.mems`, so a job can never allocate where nobody budgeted for
it. Reaching the remote part is slower, so `dispatch list` marks those jobs
with `+`, and `--numa-local-mem` refuses to spread at all.

Under `--no-cgroups` a job really can allocate anywhere, so the daemon tracks
one machine-wide pool instead; the startup log says which mode is in force.
Without that flag an unavailable cpuset controller is a refusal, not a
downgrade.

### GPUs

`--gpu-cores N` allocates N of the GPUs `nvidia-smi -L` enumerates, and the
job sees them through `CUDA_VISIBLE_DEVICES`.

**Which N is not arbitrary.** A multi-GPU job gets the closest-connected free
set in the interconnect `nvidia-smi topo -m` reports. With GPU1 busy, index
order hands a 2-GPU job GPU0+GPU2 across the machine even when GPU2+GPU3
share an NVLink. A candidate set is judged by its worst link first, because a
collective runs at the speed of its slowest pair, and the job's CPUs then
come from the NUMA node its GPUs hang off.

**GPUs are spent worst-first.** Each level of the interconnect divides the
free GPUs into islands (what sits behind one switch, one host bridge, one
socket) and a job goes in the finest island that can hold it. Where several
serve equally well it takes from the one with least left to give, so a 2-GPU
job carves into a quad that is already broken instead of splitting an intact
one and leaving nothing for the job behind it that needs four. Same best-fit
reasoning the NUMA nodes get.

Closeness is a preference and never by itself a reason to keep a job queued:
when only distant GPUs are free the job takes them, and the line logged at
start-up names the link class it ended up with. Where `nvidia-smi topo -m` is
unavailable the daemon says so at startup and falls back to index order.

Two flags turn the preference into a requirement, and a request no machine
could ever satisfy is refused at submit time rather than queued forever.

**`--numa-local-gpu`** makes the job wait until every GPU it gets hangs off
one NUMA node, with its CPUs from that node, so nothing it does crosses the
socket interconnect. `--numa-local` turns this on together with
`--numa-local-mem`, which is how to ask for a job that is local in every
respect.

**`--gpu-link CLASS`** names the worst wiring you will accept between the
GPUs themselves, in `nvidia-smi topo -m`'s own vocabulary:

| Class | GPU to GPU over |
| --- | --- |
| `NV` | NVLink |
| `PIX` | at most one PCIe bridge |
| `PXB` | several PCIe bridges, still below the host bridge |
| `PHB` | up through a PCIe host bridge |
| `NODE` | between host bridges, within one NUMA node |
| `SYS` | the socket interconnect |

Closest first; `--gpu-link` takes any of them but `SYS`, which as a floor
would rule nothing out. From `PHB` down, peer-to-peer copies are staged
through host memory rather than going card to card, which is why `dispatch
list` flags those with `!`.

This is a finer thing than NUMA locality and is not implied by it: `NODE` is
by definition a hop between host bridges *within* one node, so
`--numa-local-gpu` alone can still hand you a pair that copies through host
memory. `--gpu-link PXB` keeps a set inside one host bridge, where they do
not; `--gpu-link NV` demands NVLink, for a job whose collectives are what it
is measuring.

A `--gpu-link` job that has to wait blocks the queue under the default
`fifo-strict`, which is worth knowing before turning it on for a busy
machine: one of the backfill policies keeps the rest of the machine working
while it waits.

### Scheduling policies

Jobs are always ranked in submission order. The policy only decides whether a
job further back may use resources the blocked head job cannot yet use. Take
the queue of GPU demands `[1, 4, 2, 1, 2]` (jobs 1 to 5) on a 4-GPU machine
as a running example.

**`fifo-strict`** (default) is head-of-line blocking. Start jobs in order,
stop at the first that does not fit, and let nothing jump the queue. Only job
1 runs; job 2 needs 4 GPUs and blocks jobs 3 to 5 behind it. The most
predictable policy, and the one most likely to leave the machine idle.

**`easy-backfill`** freezes a *backfill budget* when the head job first fails
to fit, equal to the resources idle at that moment. Later jobs, including
ones submitted afterwards, may start while they fit within that budget.
Resources freed later by finishing jobs are **not** added to it; they are
held for the head, so once the machine fills up nothing jumps ahead. Jobs 1,
3 and 4 run (1+2+1 = 4 GPUs), job 2 is reserved, and as jobs finish their
GPUs are held until all 4 are free. No runtime estimates needed. This is what
most people mean by "backfill".

**`strict-backfill`** does the same, but uses each job's `--max-time` to
reserve a guaranteed start time for the head, assuming running jobs occupy
their resources until their deadline. A later job may backfill past the head
only if it provably finishes before that reserved start, so it can never
delay the head. A short job may fill idle GPUs even after job 2 blocks; a
long one that would still be running when job 2 is due is refused. Keeps more
of the machine busy than easy-backfill, at the cost of trusting `--max-time`.

None of them starve the head: every job has a bounded lifetime
(`--max-lifetime`), so a blocked job always eventually runs.

## Running the daemon

All admin parameters are arguments to `hpc-batchd`, set in the systemd unit's
`ExecStart=` line.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--max-lifetime DURATION` | `1d` | Jobs running longer than this are killed. Also the ceiling and default for `--max-time`. |
| `--reserve-cpu N` | `2` | Cores held back for the OS and the daemon, never offered to jobs. Taken from the lowest-numbered NUMA node so the others keep their full width. Headroom, not a fence: nothing pins the OS to them. |
| `--reserve-mem GB` | `2` | Memory held back for the OS and the daemon, off the same node as `--reserve-cpu`, and never more than half of any one node. |
| `--min-job-mem GB` | `2` | Floor for an automatically assigned budget, so a single-core job on a core-dense machine is not left with a sliver. Ignored when the user passes `--max-mem`. |
| `--keep-finished N` | `50` | Finished jobs remembered **per user** for `dispatch list --finished`. Metadata only; output is never kept here. |
| `--list-is-public` | off | Let non-admins use `dispatch list --all`. |
| `--admin-group GROUP` | `wheel` | Members can list, attach to and kill any user's jobs. The daemon refuses to start if the group does not exist; `hpc-batch-install` fills in one that does. |
| `--socket PATH` | `/run/hpc-batch/hpc-batch.sock` | Unix socket to listen on. |
| `--state-dir DIR` | `/var/lib/hpc-batch` | Job state, metadata and buffered output. |
| `--dev-dir DIR` | `/dev/hpc-batch` | Where per-job inspection entries appear. |
| `--schedule POLICY` | `fifo-strict` | One of the policies above. |
| `--log-level LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR`. |
| `--no-cgroups` | off | Development mode: no cgroups, CPU pinning via `sched_setaffinity` only, no enforced memory limit. Without it, unavailable cgroups are a refusal to start. |
| `--no-dev-dir` | off | Skip the inspection entries instead of refusing to start when they cannot be created. |

Durations accept plain seconds or `s`/`m`/`h`/`d` suffixes: `45m`, `2h`,
`1h30m`.

After changing arguments: `systemctl daemon-reload && systemctl reload
hpc-batch`.

### Hot reload

`systemctl reload hpc-batch` makes the daemon persist its state and re-exec
itself in place. Running jobs are *not* killed; they are re-adopted by the new
daemon, with pid reuse guarded by comparing `/proc/<pid>/stat` start times.
The same survival applies to `systemctl restart`, thanks to
`KillMode=process` in the unit and to job cgroups living outside it.

Reload is still the better habit: it re-execs the same pid, so jobs stay its
children and their exit codes are still collected, where a full restart can
only see that they ended.

### Refusals

The daemon does not start half-configured. If it was asked for isolation it
cannot deliver it says which piece is missing and exits 78 (`EX_CONFIG`),
which the unit's `RestartPreventExitStatus=` turns into a clean stop rather
than a restart loop over the same error message.

- cgroups v2 not mounted, or the daemon is not root
- the `cpuset` or `memory` controller is not available in its cgroup, which
  almost always means `Delegate=` was removed from the unit
- `--admin-group` names a group that does not exist here (it suggests ones
  that do)
- `--dev-dir` cannot be created

Each refusal names the flag that opts out of the thing being refused over
(`--no-cgroups`, `--no-dev-dir`), for when you genuinely want to run without
it. `--no-cgroups` in particular means CPU-affinity pinning only, with no
enforced memory limit and no NUMA confinement: fine for development on a
laptop, not for a shared machine.

### Authentication

The daemon identifies clients by `SO_PEERCRED` on the unix socket, so users
cannot impersonate each other. Jobs run under the submitting user's uid and
gid.

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
publishes to PyPI. A tag that disagrees with the package version fails before
anything is published: PyPI reads the version from the metadata and GitHub
reads it from the tag, and nothing else compares them.

### One-time PyPI setup

The workflow authenticates with [trusted
publishing](https://docs.pypi.org/trusted-publishers/), so there is no API
token to store, leak or rotate. PyPI verifies a short-lived OIDC token that
GitHub mints for this repository, and only for the workflow named below.

Before the first release, at <https://pypi.org/manage/account/publishing/>,
add a *pending* publisher (the project need not exist yet) with exactly these
values:

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
anyway, put it in a `PYPI_API_TOKEN` secret and give the publish step `with:
password: ${{ secrets.PYPI_API_TOKEN }}`, but then it is a long-lived
credential with upload rights, which is the thing trusted publishing exists
to avoid.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
