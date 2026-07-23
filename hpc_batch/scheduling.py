"""Scheduling policies.

Every policy walks the queued jobs in strict FIFO (submission) order and
decides which ones to start now. A policy is a pure function of (queued
jobs, current resource pool, running jobs, now, prior reservation) and
returns the jobs to start plus the head-of-queue reservation to carry into
the next tick. It reserves resources in the pool for each job it returns
(via ``pool.allocate``), exactly as the daemon would; the daemon then only
has to spawn them.

Three policies:

- ``fifo-strict`` -- head-of-line blocking. Start jobs in order; the moment
  one does not fit, stop. Nothing ever jumps the queue. Simplest and the
  default; a big job can leave resources idle while it waits.

- ``easy-backfill`` -- fill idle resources with later jobs, then reserve.
  When the head job first cannot fit, we freeze a *backfill budget* equal to
  the resources idle at that moment. Later jobs may start as long as they
  fit within that frozen budget (they "backfill" the idle capacity, even if
  they were submitted after the head blocked). Resources freed afterwards by
  finishing jobs are NOT added to the budget -- they are held for the head,
  so no job started once the machine is full jumps ahead of it. This
  prevents a stream of small jobs from starving a big one, with no runtime
  estimates. (The budget is a total across NUMA nodes; the real allocator
  still confines each job's CPUs to one node.)

- ``strict-backfill`` -- reserve using the jobs' ``max_time``. The head job
  gets a reservation: the earliest time it is guaranteed to be able to run,
  computed by assuming every running (and just-started) job occupies its
  resources until its ``max_time`` deadline. A later job may then backfill
  past the blocked head only if it provably finishes before that
  reservation, so it can never delay the head. This keeps more resources
  busy than easy-backfill while still guaranteeing the head is never
  delayed.
"""

import math
from dataclasses import dataclass

FIFO_STRICT = "fifo-strict"
EASY_BACKFILL = "easy-backfill"
STRICT_BACKFILL = "strict-backfill"
MODES = (FIFO_STRICT, EASY_BACKFILL, STRICT_BACKFILL)


@dataclass
class Reservation:
    """easy-backfill's frozen budget for a blocked head job, carried across
    ticks. Backfill jobs draw the budget down; it is never replenished, so
    freed resources accrue to the head instead."""

    head_id: int
    cpu: int
    gpu: int
    mem: float | None


def _alloc(pool, job):
    return pool.allocate(job.cpu, job.gpu_cores, job.max_mem_gb, job.exclusive)


def _fits(pool, job) -> bool:
    return pool.would_fit(job.cpu, job.gpu_cores, job.max_mem_gb, job.exclusive)


def _fill_fifo(queued, pool):
    """Start jobs in FIFO order until one does not fit. Returns
    ``(to_start, blocked_index)`` where blocked_index is the position of the
    first job that did not fit, or None if every job started."""
    to_start = []
    for i, job in enumerate(queued):
        alloc = _alloc(pool, job)
        if alloc is None:
            return to_start, i
        to_start.append((job, alloc))
    return to_start, None


def plan(mode, queued, pool, running, now, reservation):
    """Decide what to start this tick.

    ``queued`` must be FIFO-ordered. Returns ``(to_start, reservation)``
    where ``to_start`` is a list of ``(job, Allocation)`` already reserved in
    ``pool``, and ``reservation`` is the easy-backfill budget to carry into
    the next tick (a ``Reservation`` or None; always None for the other
    policies).
    """
    if mode == EASY_BACKFILL:
        return _plan_easy(queued, pool, reservation)
    if mode == STRICT_BACKFILL:
        return _plan_strict(queued, pool, running, now)
    return _plan_fifo(queued, pool)


def _plan_fifo(queued, pool):
    to_start, _ = _fill_fifo(queued, pool)
    return to_start, None


def _plan_easy(queued, pool, reservation):
    head = queued[0] if queued else None
    if (
        head is not None
        and reservation is not None
        and reservation.head_id == head.id
        and not _fits(pool, head)
    ):
        # The head blocked on an earlier tick and still cannot run. Backfill
        # later jobs only within the frozen budget; hold everything else.
        return _backfill(queued[1:], pool, reservation), reservation

    # Establishing pass: start the fitting prefix. The first job that does
    # not fit becomes the head; snapshot the now-idle resources as its
    # backfill budget and let the jobs behind it draw the budget down.
    to_start, blocked_at = _fill_fifo(queued, pool)
    if blocked_at is None:
        return to_start, None
    cpu, gpu, mem = pool.free_totals()
    reservation = Reservation(head_id=queued[blocked_at].id, cpu=cpu, gpu=gpu, mem=mem)
    to_start += _backfill(queued[blocked_at + 1:], pool, reservation)
    return to_start, reservation


def _backfill(candidates, pool, reservation):
    """Start each job that still fits within the reservation's frozen budget,
    drawing the budget down as we go."""
    to_start = []
    for job in candidates:
        if job.cpu > reservation.cpu or job.gpu_cores > reservation.gpu:
            continue
        if (
            reservation.mem is not None
            and job.max_mem_gb is not None
            and job.max_mem_gb > reservation.mem
        ):
            continue
        alloc = _alloc(pool, job)
        if alloc is None:
            continue  # budget said yes but the pool disagreed; leave it queued
        reservation.cpu -= job.cpu
        reservation.gpu -= job.gpu_cores
        if reservation.mem is not None and job.max_mem_gb is not None:
            reservation.mem -= job.max_mem_gb
        to_start.append((job, alloc))
    return to_start


def _plan_strict(queued, pool, running, now):
    to_start, blocked_at = _fill_fifo(queued, pool)
    if blocked_at is None:
        return to_start, None
    head = queued[blocked_at]
    # The head's guaranteed start accounts for both already-running jobs and
    # the ones we just started in this pass (which begin ~now).
    finishers = [(j.deadline(now), j.allocation()) for j in running]
    finishers += [(j.deadline(now), alloc) for j, alloc in to_start]
    deadline = _reservation_deadline(pool, head, finishers)
    if deadline == math.inf:
        return to_start, None  # head not guaranteeable: risk nothing behind it
    for job in queued[blocked_at + 1:]:
        # Backfill only if this job provably finishes before the head's
        # reserved start, so it cannot delay the head.
        if now + job.max_time_s <= deadline:
            alloc = _alloc(pool, job)
            if alloc is not None:
                to_start.append((job, alloc))
    return to_start, None


def _reservation_deadline(pool, head, finishers):
    """Earliest time ``head`` is guaranteed to fit, given that each running
    or just-started job frees its resources at its deadline.

    ``finishers`` is a list of ``(deadline, Allocation)``. We release them
    into a scratch copy of the pool in deadline order until the head fits;
    that deadline is the head's guaranteed start time. math.inf means it can
    never be guaranteed (should not happen: the submitter validated that the
    job fits on an empty machine, and every job has a bounded lifetime).
    """
    sim = pool.clone()
    for deadline, alloc in sorted(finishers, key=lambda item: item[0]):
        sim.release(alloc)
        if _fits(sim, head):
            return deadline
    return math.inf
