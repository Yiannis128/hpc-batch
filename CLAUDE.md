# CLAUDE.md

## Commands

```sh
hatch run test                                  # whole suite (~0.5s)
hatch run test tests/test_resources.py -q       # one file
hatch build && python scripts/check_wheel.py    # build; wheel must carry the systemd unit
```

There is no linter or formatter configured, so there is nothing to run and no
suppression comment to add.

Run the daemon without root, against scratch paths:

```sh
S=$(mktemp -d)
python -m hpc_batch.daemon --no-cgroups --socket "$S/sock" \
    --state-dir "$S/state" --dev-dir "$S/dev" --max-lifetime 1h &
export HPC_BATCH_SOCKET="$S/sock"
dispatch new -- echo hello
```

In user mode the daemon cannot setuid, so it only accepts jobs from its own
uid, and `--no-cgroups` is mandatory.

## Constraints

Standard library only — `dependencies = []` is deliberate, so the install is a venv plus one wheel. Python >= 3.10, so `match` and `X | None` are available. Linux only (cgroup v2, SO_PEERCRED, fork/setuid).

## Architecture

One root daemon, one unix socket, a thin client. Layered so the interesting parts are testable without root:

- `protocol.py` — wire format (newline-delimited JSON) and the state/reason string constants. These are contract twice over: `dispatch list` prints `reason` verbatim, *and* both are persisted in `state.json`, so renaming one behind a client-side shim breaks re-adoption of running jobs across a restart.
- `resources.py` — machine discovery plus `ResourcePool`, the allocator. It has no idea what a job is; it takes a `Request` and returns an `Allocation`. Everything about NUMA placement, memory domains and GPU topology lives here.
- `scheduling.py` — pure functions over the pool. **Policies reserve in the pool themselves**; the daemon only spawns what it is handed.
- `jobs.py` — the `Job` dataclass shared by the queue, `state.json` and `info.json`. Converts itself to a `Request` and back to an `Allocation`.
- `cgroup.py` — writes `cpuset.cpus`, `cpuset.mems`, `memory.max`. Enforcement only; it makes no decisions.
- `daemon.py` — the only stateful actor: socket, queue, spawn, reap, persist, permissions. Every mutation of a `Job` goes through `_job_changed`, which rewrites `info.json` and marks state dirty; skip it and the on-disk view is stale until something else happens to persist.
- `client.py` — thin.
- `install.py` — picks the first of `util.ADMIN_GROUPS` that exists on the host and rewrites the systemd unit from the packaged template on *every* run, upgrades included. Reordering that tuple silently moves admin control over every job on a box that has both `wheel` and `sudo`. `install.json` in the prefix records what a run chose, and `settle()` is the only thing that reads it: an option not passed comes from there rather than from the default, so `detect_admin_group()` cannot hand admin to a group that appeared on the box after the install, and an upgrade cannot relocate the entry points and orphan the old ones. An option passed with a different value is refused, because nothing here migrates an existing install.

### Invariants worth knowing before changing anything

**One NUMA node per job.** The ceiling is `ResourcePool._max_nodes`, read by both `validate()` (submit time) and `_plan()` (placement); if they ever disagree a job passes submission and can then never be placed. Only `--exclusive` (without a `--numa-local-*` flag) spans nodes.

**Memory is charged per domain, and the charge *is* the confinement.** `Allocation.mem_nodes()` is what `cgroup.create` writes into `cpuset.mems`, so the books and the kernel cannot disagree about where a job's memory lives. When `cpuset.mems` cannot be written (no cgroups, `--no-cgroups`) the pool collapses to one machine-wide domain keyed `SHARED_MEM` (-1) rather than pretending to a split that is not enforced; `charged_nodes()` is then empty and `mem_nodes()` falls back to the cpu nodes.

**Every handler that names a job goes through `Daemon._find_job`.** It is the only ownership check ("yours, or you are an admin"), and `_client` hands each handler a raw uid and trusts it to ask. A handler that reaches into `self.jobs` directly serves other users' jobs. `list --all` is gated separately, because it is the one request not about a single job.

**Only jobs in `Daemon._reserved` release resources.** A job can be finalized without ever having been reserved (failed spawn, killed while queued), and double-release silently inflates the pool. What gets released is rebuilt from `Job.cpus`/`numa_nodes`/`mem_by_node`/`gpus`/`exclusive`, not remembered — so those fields must not change while a job is running, or it returns resources it never held.

**easy-backfill's budget is a cloned `ResourcePool`, not a set of counters.** See the docstring on `Reservation`. Two consequences: `clone()` is a plain `deepcopy`, so `ResourcePool` has to stay pure data (a logger or a back-reference to the daemon either blows up or is silently aliased), and `_backfill` books one `Allocation` into two pools, so `allocate` must not do anything beyond plan-and-reserve.

**Every filesystem operation on a user-chosen path goes through `run_as_user`** (fork, drop privileges, report back over a pipe). The daemon is root; touching a user's path in-process lends them root's privileges. `drop_privileges` order — initgroups, setgid, setuid — is load-bearing and defined once.

**The daemon owns `CUDA_VISIBLE_DEVICES`.** `_job_env` composes `defaults | job.env | ours` so a forwarded `--env` can never win: that variable is the whole of a job's GPU isolation. It is rendered by `format_id_list`, which also writes `cpuset.cpus` and `cpuset.mems` — cpuset accepts ranges (`0-3`) and `CUDA_VISIBLE_DEVICES` does not, so compressing them for the cgroup's benefit would quietly hide GPUs from every job.

**Reload is a re-exec, not a restart.** Anything added to `Job` must survive a round trip through `state.json`; `Job.from_dict` drops unknown keys and `_backfill_placement` fills in fields an older state file lacks.

**Startup is all-or-nothing.** Never add a silent fallback for isolation that was asked for. Every refusal names the flag that opts out of the thing being refused, which `_refuse()` in `cgroup.py` enforces by construction.

**Job output in the state dir is a buffer, not storage.** It exists so `dispatch attach` has something to stream; on finish it is copied to the user's destination and unlinked. Only a failed delivery leaves a copy behind.

**The order of `_LINK_CLASSES` is the GPU rank function.** `GPU_LINK_CHOICES` assumes the last entry is the worst class and `REMOTE_GPU_LINKS` slices at `PHB` to mean "left the PCIe tree", so inserting a class mid-list silently redefines what `--gpu-link PXB` accepts and what `dispatch list` flags as remote. Separately, `GpuTopology.__deepcopy__` returns `self` on purpose: `pool.clone()` runs on every tick and the wiring cannot change.

## Tests

Pure unit tests, no root, no cgroups (`use_cgroups=False`, `use_dev_dir=False`), and they have to stay that way. There is no `conftest.py`; test modules import each other's helpers directly (`from test_resources import held`), which works because pytest prepends the test directory to `sys.path` — so that import is correct, not a mistake to fix.

## Comments in this codebase

Docstrings explain the bargain a function is making; inline comments mark the traps (pid reuse, symlinks at a destination, NVLink being a mesh rather than a hierarchy). Match that.
