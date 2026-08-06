# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```sh
hatch run test                                  # whole suite (~1s)
hatch run test tests/test_resources.py -q       # one file
hatch run test tests/test_daemon.py::TestRunAsUser -k oom
hatch build && python scripts/check_wheel.py    # build; wheel must carry the systemd unit
```

There is no linter or formatter configured, and no `conftest.py`.

Run the daemon without root, against scratch paths:

```sh
S=$(mktemp -d)
python -m hpc_batch.daemon --no-cgroups --socket "$S/sock" \
    --state-dir "$S/state" --dev-dir "$S/dev" --max-lifetime 1h &
export HPC_BATCH_SOCKET="$S/sock"
dispatch new -- echo hello
```

In user mode the daemon only accepts jobs from its own uid, and `--no-cgroups` is mandatory.

## Constraints

Standard library only — `dependencies = []` is deliberate, so the install is a venv plus one wheel. Python >= 3.10, Linux only (cgroup v2, SO_PEERCRED, fork/setuid). `__version__` in `hpc_batch/__init__.py` is the single source of truth; the release workflow fails if the git tag disagrees.

## Architecture

One root daemon, one unix socket, a thin client. Layered so the interesting parts are testable without root:

- `protocol.py` — wire format (newline-delimited JSON) and the state/reason string constants. These are contract: `dispatch list` prints `reason` verbatim.
- `resources.py` — machine discovery plus `ResourcePool`, the allocator. It has no idea what a job is; it takes a `Request` and returns an `Allocation`. Everything about NUMA placement, memory domains and GPU topology lives here.
- `scheduling.py` — `plan(mode, queued, pool, running, now, reservation)`, pure functions over the pool. **Policies reserve in the pool themselves**; the daemon only spawns what it is handed.
- `jobs.py` — the `Job` dataclass shared by the queue, `state.json` and `info.json`. Converts itself to a `Request` and back to an `Allocation`.
- `cgroup.py` — writes `cpuset.cpus`, `cpuset.mems`, `memory.max`. Enforcement only; it makes no decisions.
- `daemon.py` — the only stateful actor: socket, queue, spawn, reap, persist, permissions.
- `client.py` / `install.py` — CLI and system-wide installer, both thin.

### Invariants worth knowing before changing anything

**One NUMA node per job.** `ResourcePool._max_nodes` is the only place that ceiling is decided, and both `validate()` (submit time) and `_plan()` (placement) read it. If they ever disagree, a job passes submission and can then never be placed. Only `--exclusive` (without a `--numa-local-*` flag) spans nodes.

**Memory is charged per domain, and the charge *is* the confinement.** The keys of `Allocation.mem_by_node` are what `cgroup.create` writes into `cpuset.mems`, so the books and the kernel cannot disagree about where a job's memory lives. When `cpuset.mems` cannot be written (no cgroups, `--no-cgroups`) the pool collapses to one machine-wide domain keyed `SHARED_MEM` (-1) rather than pretending to a split that is not enforced. `charged_nodes()` is the one filter that separates real node ids from that sentinel.

**easy-backfill's budget is a cloned `ResourcePool`, not a set of counters.** "Does this job fit" involves NUMA placement, spanning, exclusivity; a second implementation would drift. See the docstring on `Reservation`.

**Only jobs in `Daemon._reserved` release resources.** A job can be finalized without ever having been reserved (failed spawn, killed while queued), and double-release silently inflates the pool.

**Every filesystem operation on a user-chosen path goes through `run_as_user`** (fork, drop privileges, report back over a pipe). The daemon is root; touching a user's path in-process lends them root's privileges. `drop_privileges` order — initgroups, setgid, setuid — is load-bearing and defined once.

**The daemon owns `CUDA_VISIBLE_DEVICES`.** `_job_env` composes `defaults | job.env | ours` so a forwarded `--env` can never win: that variable is the whole of a job's GPU isolation.

**Reload is a re-exec, not a restart.** SIGHUP persists state and `execve`s in place, so running jobs stay children of the same pid and their exit codes are still collected. Anything added to `Job` must therefore survive a round trip through `state.json`; `Job.from_dict` drops unknown keys and `_backfill_placement` fills in fields an older state file lacks. A full restart also re-adopts jobs, via `/proc/<pid>/stat` starttime to guard pid reuse, but can only see that they ended.

**Startup is all-or-nothing.** A configuration problem raises `StartupError` and exits 78 (`EX_CONFIG`), which the unit's `RestartPreventExitStatus=78` turns into a clean stop. Every refusal names the flag that opts out of the thing being refused (`_refuse()` in `cgroup.py` enforces that by construction). Never add a silent fallback for isolation that was asked for.

**Job output in the state dir is a buffer, not storage.** It exists so `dispatch attach` has something to stream; on finish it is copied to the user's destination and unlinked. Only a failed delivery leaves a copy behind.

**GPU sets are chosen by island, not by score.** `_closest_gpus` walks link classes from closest out, looks only inside islands wide enough to hold the job, and takes from the island with least left to give. `_pick_gpus(search=False)` skips that search for callers only asking whether a request fits — except when `--gpu-link` makes the choice decide fit. `GpuTopology.__deepcopy__` returns `self` on purpose: `pool.clone()` runs on every tick and the wiring cannot change.

**The daemon's `Config` fields and the argparse `dest=` names match**, so `main()` builds the config mechanically. A new knob is one option plus one field.

## Tests

Pure unit tests, no root, no cgroups (`use_cgroups=False`, `use_dev_dir=False` in `make_config`). There is no `conftest.py`; test modules import each other's helpers directly (`from test_resources import held`), which works because pytest prepends the test directory to `sys.path`. Reusable fixtures live in `tests/test_resources.py` (`make_pool`, `topo_pool`, `quad_pool`, `held`) and `tests/test_daemon.py` (`make_config`, `make_daemon`, `add_job`).

## Comments in this codebase

The existing style is heavy on *why* and silent on *what*: docstrings explain the bargain a function is making, and inline comments mark the traps (pid reuse, symlinks at a destination, NVLink being a mesh rather than a hierarchy). Match that. Do not add comments that restate the code.
