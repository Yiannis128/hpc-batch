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

Python >= 3.10, no dependencies outside the standard library. Linux only: it
needs cgroups v2 and systemd.

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/Yiannis128/hpc-batch/master/install.sh | sudo sh
```

That creates a virtualenv at `/opt/hpc-batch`, installs `hpc-batch` from PyPI
into it, puts `dispatch`, `hpc-batchd` and `hpc-batch-install` on everyone's
`PATH`, writes the systemd unit with an admin group that exists on this
distro, and starts the daemon.

`PATH` only changes for *new* login shells, so in the one you ran it from use
`/opt/bin/dispatch` or log in again. Then:

```sh
systemctl status hpc-batch --no-pager
dispatch new -- echo hello
dispatch list --finished
```

Upgrades are `sudo /opt/bin/hpc-batch-install`, which reinstalls and
*reloads* the daemon rather than restarting it, so running jobs are re-adopted
instead of losing their exit codes. It rewrites the unit each time, so keep
local changes in a drop-in (`systemctl edit hpc-batch`). `--uninstall`
removes everything except `/var/lib/hpc-batch`, so job history and queued
jobs survive. `hpc-batch-install --help` covers the rest.

Do **not** install with `sudo pipx`. It puts the entry points in
`/root/.local/bin` (mode 700), where `hpc-batchd` still works but no other
user can run the client.

## Using it

Everything after `--` is the job's command line.

```sh
dispatch new --cpu 2 --gpu-cores 3 --max-mem 84 --max-time 2h -- ./bench.sh
dispatch list           # --all for everyone's, --finished for exit statuses
dispatch attach 7       # follow a job's output, like tail -f
dispatch kill 7         # kill a running job, or drop a queued one
```

`dispatch new --help` lists the options. The ones worth knowing about:
`--exclusive` waits for an idle machine and keeps it, `--numa-local` keeps
the whole job on one NUMA node, and `--gpu-link` refuses GPUs wired worse
than you asked for. All three make a job wait for what it wants rather than
run on what is free.

```
USER   ID  COMMAND            START     UPTIME  MAX-TIME  MEM    GPU     EXCLUSIVE
alice  7   ./allreduce_bench  14:02:11  6m21s   2h        64G    4 NV    no
bob    8   ./train.py         14:05:40  2m52s   1d        192G+  2 PHB!  no
carol  9   ./sweep.sh         -         queued  30m       16G    -       no
```

`+` on the memory means the budget was spread over more than one NUMA node,
so part of it is slower to reach. The GPU column gives the count and the link
class pacing them, `!` marking a set that talks across a host bridge or the
socket interconnect. Both are there so a job never quietly produces worse
numbers than you expected.

While a job runs its combined stdout/stderr is buffered so `dispatch attach`
has something to stream. When it finishes the buffer is copied to
`output.<id>.log` in the directory you submitted from and then dropped: the
daemon never keeps your results, the only lasting copy is yours. Queued and
running jobs also appear under `/dev/hpc-batch/jobs/<id>`, holding
`info.json` and the live output.

## How it works

**Memory.** Every job gets a budget whether it asks for one or not, because
an unbounded job is invisible to the scheduler and can starve everything
sharing its node. Without `--max-mem` that is the share of a node its cores
represent. Budgets are charged per NUMA node rather than machine-wide, so a
job normally gets purely local memory; one too big for any single node is
spread across several rather than made to wait, and `dispatch list` says so.
Swap is off for every job, so `--max-mem` is a hard RAM limit and exceeding
it is an OOM kill rather than a slow crawl.

**GPUs.** A multi-GPU job gets the closest-connected free set in the
interconnect `nvidia-smi topo -m` reports. With GPU1 busy, index order hands
a 2-GPU job GPU0+GPU2 across the machine even when GPU2+GPU3 share an
NVLink. A set is judged by its worst link first, because a collective runs at
the speed of its slowest pair, and the job's CPUs then come from the NUMA
node its GPUs hang off.

Each level of the interconnect divides the free GPUs into islands, and a job
goes in the finest island that can hold it, taking from the one with least
left to give where several would serve. A 2-GPU job carves into a quad that
is already broken rather than splitting an intact one and leaving nothing for
the job behind it that needs four.

Closeness is only a preference: when the free GPUs are distant the job takes
them anyway, and the start-up log names the link class it got. `--gpu-link`
and `--numa-local-gpu` turn it into a requirement the job will wait for.
Where `nvidia-smi topo -m` is unavailable the daemon says so at startup and
falls back to index order.

**Scheduling.** Jobs are always ranked in submission order; the policy only
decides whether a job further back may use resources the blocked head job
cannot yet use. `fifo-strict` (the default) says no, and leaves the machine
idle to keep the order honest. `easy-backfill` lets later jobs into whatever
was idle when the head first failed to fit, but holds everything freed after
that for the head. `strict-backfill` also lets a job past the head if its
`--max-time` proves it will finish before the head's reserved start. None of
them starve the head, since every job has a bounded lifetime.

**Isolation.** Clients are identified by `SO_PEERCRED` on the socket, so
users cannot impersonate each other, and jobs run under the submitting user's
uid and gid. Each job gets its own cgroup with `cpuset.cpus`, `cpuset.mems`
and `memory.max` applied, living outside the daemon's own service cgroup so
restarting the unit never disturbs a running job. A job gets a clean
environment unless it passes `--env`; `CUDA_VISIBLE_DEVICES` is set by the
daemon and a forwarded value never wins, since it is the whole of a job's GPU
isolation.

## Running the daemon

Admin settings are arguments to `hpc-batchd` on the systemd unit's
`ExecStart=` line; `hpc-batchd --help` lists them. They cover the scheduling
policy, the maximum job lifetime, how much CPU and memory to hold back for
the OS, which group counts as admin, and where the socket, state and
inspection entries live.

After changing them, `systemctl daemon-reload && systemctl reload
hpc-batch`. Reload re-execs the same pid, so running jobs stay its children
and their exit codes are still collected; a restart survives too, but can
only see that they ended.

The daemon does not start half-configured. If it was asked for isolation it
cannot deliver, it names the missing piece and exits 78 (`EX_CONFIG`), which
the unit turns into a clean stop rather than a restart loop over the same
error. That covers cgroups v2 not being mounted, the `cpuset` or `memory`
controller missing from its cgroup (almost always a removed `Delegate=`), an
admin group that does not exist, and a `--dev-dir` it cannot create. Each
refusal names the flag that opts out of the thing being refused over.
`--no-cgroups` in particular means CPU-affinity pinning only, with no
enforced memory limit and no NUMA confinement: fine for a laptop, not for a
shared machine.

## Development

No root required: run the daemon in user mode against scratch paths.

```sh
hatch run test    # unit tests

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
vim hpc_batch/__init__.py          # __version__ = "0.2.0"
git commit -am "Release 0.2.0" && git push
gh release create v0.2.0 --generate-notes
```

The workflow checks the tag matches `__version__`, runs the tests, builds the
sdist and wheel, attaches them to the release and publishes to PyPI. A tag
that disagrees with the package version fails before anything is published:
PyPI reads the version from the metadata and GitHub reads it from the tag,
and nothing else compares them.

It authenticates with [trusted
publishing](https://docs.pypi.org/trusted-publishers/), so there is no API
token to store, leak or rotate. Before the first release, add a *pending*
publisher at <https://pypi.org/manage/account/publishing/> for project
`hpc-batch`, owner `Yiannis128`, repository `hpc-batch`, workflow
`release.yml`, environment `pypi`. The environment name has to match
`environment: pypi` in `.github/workflows/release.yml`; adding required
reviewers to it under Settings → Environments turns publishing into something
that needs an explicit approval.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
