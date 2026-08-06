"""dispatch: the hpc-batch client CLI.

Talks to hpc-batchd over its unix socket. The daemon authenticates us via
SO_PEERCRED, so there is nothing to configure client-side.
"""

import argparse
import io
import json
import os
import socket
import sys
import time

from . import __version__
from .protocol import DONE, MAX_LINE, OOM, QUEUED, encode, socket_path
from .resources import GPU_LINK_CHOICES
from .util import duration_arg, format_duration, format_gb, format_table, format_time


class DispatchError(Exception):
    pass


def _connect() -> socket.socket:
    path = socket_path()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(path)
    except OSError as exc:
        sock.close()
        raise DispatchError(
            f"cannot reach the hpc-batch daemon on {path} ({exc.strerror or exc}); is the service running?"
        ) from None
    return sock


def _read_response(f: io.BufferedReader) -> dict:
    """Read and validate the daemon's one-line JSON response header."""
    line = f.readline(MAX_LINE + 1)
    if len(line) > MAX_LINE:
        raise DispatchError("oversized response from daemon")
    if not line.endswith(b"\n"):
        raise DispatchError("daemon closed the connection unexpectedly")
    try:
        resp = json.loads(line)
    except json.JSONDecodeError:
        raise DispatchError("malformed response from daemon") from None
    if not resp.get("ok"):
        raise DispatchError(resp.get("error", "unknown daemon error"))
    return resp


def _request(req: dict) -> dict:
    with _connect() as sock:
        sock.sendall(encode(req))
        return _read_response(sock.makefile("rb"))


# -- subcommands ---------------------------------------------------------


def cmd_new(args: argparse.Namespace, command: list[str]) -> int:
    # Resolve --output here: the daemon's cwd is "/", so a relative path only
    # means what the user intended if we make it absolute on their side.
    output = None if args.no_output else os.path.abspath(args.output or os.getcwd())
    req = {
        "cmd": "new",
        "argv": command,
        "cwd": os.getcwd(),
        "cpu": args.cpu,
        "gpu_cores": args.gpu_cores,
        "max_mem_gb": args.max_mem,
        "max_time_s": args.max_time,
        "exclusive": args.exclusive,
        "numa_local_mem": args.numa_local or args.numa_local_mem,
        "numa_local_gpu": args.numa_local or args.numa_local_gpu,
        "min_gpu_link": args.gpu_link,
        "env": dict(os.environ) if args.env else {},
        "output": output,
    }
    resp = _request(req)
    # Always report the cores and the memory budget: with --exclusive, or with
    # no --max-mem, the daemon picks these, and this is the user's only chance
    # to notice before a job is killed for exceeding a limit it never named.
    mem = resp.get("max_mem_gb")
    budget = "" if mem is None else (
        f", memory {format_gb(mem)}{' (default)' if resp.get('mem_defaulted') else ''}"
    )
    print(
        f"job {resp['id']} {resp['state']} "
        f"({resp.get('cpu')} cpu, max time "
        f"{format_duration(resp.get('max_time_s'))}{budget})"
    )
    if output:
        print(f"output will be saved to {resp.get('output_dest', output)}")
    return 0


def _exit_label(job: dict) -> str:
    """The EXIT column: why the job stopped, or its exit code."""
    if job["state"] != DONE:
        return "-"
    if job.get("reason"):
        return job["reason"]
    code = job.get("exit_code")
    return "-" if code is None else str(code)


#: Link classes where a peer-to-peer copy has left the GPUs' own PCIe tree
#: and is paying for it. Flagged in `dispatch list` the way spread memory is.
_FAR_GPU_LINKS = ("PHB", "NODE", "SYS")


def _gpu_link_is_far(job: dict) -> bool:
    return job.get("gpu_link") in _FAR_GPU_LINKS


def _gpu_label(job: dict) -> str:
    """The GPU column: how many, and what they talk to each other over."""
    count = job.get("gpus") or 0
    if not count:
        return "-"
    link = job.get("gpu_link")  # absent for a 1-gpu job: there is no pair
    return f"{count} {link}{'!' if _gpu_link_is_far(job) else ''}" if link else str(count)


def cmd_list(args: argparse.Namespace) -> int:
    resp = _request({"cmd": "list", "all": args.all, "finished": args.finished})
    jobs = resp.get("jobs", [])
    if not jobs:
        print("no jobs")
        return 0
    headers = ["USER", "ID", "COMMAND", "START", "UPTIME", "MAX-TIME", "MEM"]
    # Only where there are gpus to talk about: on a cpu-only machine the
    # column would be a row of dashes.
    show_gpus = any(job.get("gpus") for job in jobs)
    if show_gpus:
        headers += ["GPU"]
    headers += ["EXCLUSIVE"]
    if args.finished:
        headers += ["STATE", "EXIT"]
    # The daemon stamps the response with the clock it built the rows against.
    # Falling back to ours only matters against a daemon too old to send it,
    # which is also too old to send start_time, so START is "-" either way.
    now = resp.get("now", time.time())
    rows = []
    for job in jobs:
        uptime = "queued" if job["state"] == QUEUED else format_duration(job["uptime_s"])
        # "+" marks a budget that did not fit one NUMA node and had to be
        # spread across several, which makes part of it slower to reach.
        label = format_gb(job.get("max_mem_gb"))
        if job.get("mem_spans_nodes"):
            label += "+"
        row = [
            job["user"],
            str(job["id"]),
            job["command"],
            format_time(job.get("start_time"), now),
            uptime,
            format_duration(job["max_time_s"]),
            label,
        ]
        if show_gpus:
            row += [_gpu_label(job)]
        row += ["yes" if job["exclusive"] else "no"]
        if args.finished:
            row += [job["state"], _exit_label(job)]
        rows.append(row)
    print(format_table(headers, rows))
    if any(job.get("mem_spans_nodes") for job in jobs):
        print("\n+ memory spans NUMA nodes; the remote part is slower to access. "
              "Use --numa-local-mem to wait for a node that fits instead.")
    if any(_gpu_link_is_far(job) for job in jobs):
        print("\n! these gpus reach each other across a host bridge or the "
              "socket interconnect, so peer-to-peer copies are slower. Use "
              "--gpu-link to wait for closer ones instead.")
    for job in jobs:
        if job.get("reason") == OOM:
            how = "assigned by default" if job.get("mem_defaulted") else "requested"
            print(
                f"job {job['id']}: killed for exceeding its "
                f"{format_gb(job.get('max_mem_gb'))} memory budget ({how}); "
                "pass --max-mem to raise it",
                file=sys.stderr,
            )
    # Saving the durable copy can fail long after submission (directory
    # removed mid-run, quota hit), so surface it rather than leaving the user
    # to wonder where their output went.
    for job in jobs:
        if job.get("output_error"):
            print(
                f"job {job['id']}: could not save output to "
                f"{job.get('output_dest')}: {job['output_error']}",
                file=sys.stderr,
            )
    return 0


def cmd_attach(args: argparse.Namespace) -> int:
    with _connect() as sock:
        sock.sendall(encode({"cmd": "attach", "id": args.id}))
        f = sock.makefile("rb")
        resp = _read_response(f)
        if resp.get("state") == QUEUED:
            print(f"job {args.id} is queued; waiting for it to start...", file=sys.stderr)
        elif resp.get("state") == DONE:
            # The streamed copy is dropped when a job ends, so point at the
            # file the user actually kept rather than printing nothing.
            dest = resp.get("output_dest")
            note = (
                f"its output is at {dest}" if dest
                else "it ran with --no-output, so nothing was kept"
            )
            print(f"job {args.id} has finished; {note}", file=sys.stderr)
        out = sys.stdout.buffer
        # read1: forward each chunk as it arrives rather than blocking to
        # fill a full buffer.
        while chunk := f.read1(65536):
            out.write(chunk)
            out.flush()
    return 0


def cmd_kill(args: argparse.Namespace) -> int:
    resp = _request({"cmd": "kill", "id": args.id})
    if resp.get("state") == "removed":
        print(f"job {args.id} removed from the queue")
    else:
        print(f"job {args.id}: kill signal sent")
    return 0


# -- entry point ---------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dispatch",
        description="Submit and manage hpc-batch jobs.",
    )
    parser.add_argument("--version", action="version", version=f"dispatch {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_new = sub.add_parser(
        "new",
        help="submit a job",
        usage="dispatch new [options] -- <command> [args...]",
        description="Submit a job to the FIFO queue. Everything after '--' is "
                    "executed as your user once the requested resources are free. "
                    "Output is captured and available via 'dispatch attach'.",
    )
    p_new.add_argument("--cpu", type=int, default=None, metavar="N",
                       help="cpu cores to allocate, all from one NUMA node "
                            "(default: 1, or every core with --exclusive)")
    p_new.add_argument("--gpu-cores", type=int, default=0, metavar="N",
                       help="how many of the nvidia-smi -L gpus to allocate (default: 0)")
    p_new.add_argument("--max-mem", type=float, default=None, metavar="GB",
                       help="memory limit in GiB. The default is the share of "
                            "a NUMA node your cores represent; the job is "
                            "killed if it exceeds the limit either way")
    p_new.add_argument("--numa-local", action="store_true",
                       help="keep the whole job on one NUMA node: implies both "
                            "--numa-local-mem and --numa-local-gpu")
    p_new.add_argument("--numa-local-mem", action="store_true",
                       help="only run where the whole memory budget fits on "
                            "the same NUMA node as the cpus, waiting if "
                            "necessary, instead of spreading it across nodes")
    p_new.add_argument("--numa-local-gpu", action="store_true",
                       help="only run where every gpu hangs off one NUMA node "
                            "and the cpus come from that node, waiting if "
                            "necessary, instead of reaching across the "
                            "interconnect")
    p_new.add_argument("--gpu-link", type=str.upper, default=None, metavar="CLASS",
                       choices=GPU_LINK_CHOICES,
                       help="refuse gpus wired worse than this to each other, "
                            "waiting instead: one of "
                            f"{', '.join(GPU_LINK_CHOICES)} (closest first, as "
                            "`nvidia-smi topo -m` names them). NV demands "
                            "NVLink; PXB or better keeps a set behind one PCIe "
                            "host bridge, where peer-to-peer copies do not have "
                            "to go through host memory")
    p_new.add_argument("--max-time", type=duration_arg, default=None, metavar="DURATION",
                       help="kill the job after this long, e.g. 30m or 2h "
                            "(default and upper bound: the admin's max lifetime)")
    p_new.add_argument("--env", action="store_true",
                       help="run the job with your current environment instead of a "
                            "clean one; the daemon's own variables still win")
    p_new.add_argument("--exclusive", action="store_true",
                       help="run alone: wait for an idle machine and block others while running")
    p_new.add_argument("--output", default=None, metavar="PATH",
                       help="where to save the output when the job finishes: a "
                            "directory, in which case it is written as "
                            "output.<id>.log, or an exact file path "
                            "(default: the current directory)")
    p_new.add_argument("--no-output", action="store_true",
                       help="discard the output when the job ends; 'dispatch "
                            "attach' still works while it runs")

    p_list = sub.add_parser(
        "list",
        help="list current jobs",
        description="List queued and running jobs: "
                    "<username> <id> <command> <start> <uptime> <max-time> "
                    "<mem> <exclusive>. With --finished, also lists recently "
                    "finished jobs and adds <state> <exit> columns.",
    )
    p_list.add_argument(
        "--all", action="store_true",
        help="list every user's jobs (admins; everyone if the daemon "
             "was started with --list-is-public)",
    )
    p_list.add_argument(
        "--finished", action="store_true",
        help="also show recently finished jobs with their exit status",
    )

    p_attach = sub.add_parser(
        "attach",
        help="follow a job's output",
        description="Stream a job's combined stdout/stderr to your terminal, "
                    "following it live (like tail -f) until the job ends or you "
                    "press Ctrl-C. Detaching does not affect the job. Admins can "
                    "attach to any user's job.",
    )
    p_attach.add_argument("id", type=int, help="job id (see 'dispatch list')")

    p_kill = sub.add_parser(
        "kill",
        help="kill a job (or remove it from the queue)",
        description="Kill your running job (SIGTERM, escalating to SIGKILL "
                    "after a grace period) or remove it from the queue if it "
                    "has not started yet. Admins can kill any user's job.",
    )
    p_kill.add_argument("id", type=int, help="job id (see 'dispatch list')")

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Everything after the first bare "--" is the job's command line; split it
    # off before argparse so job arguments are never mistaken for our options.
    command: list[str] = []
    if argv and argv[0] == "new" and "--" in argv:
        split = argv.index("--")
        command = argv[split + 1:]
        argv = argv[:split]

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "new":
            if not command:
                parser.error("no command given; usage: dispatch new [options] -- <command>")
            return cmd_new(args, command)
        if args.command == "list":
            return cmd_list(args)
        if args.command == "attach":
            return cmd_attach(args)
        if args.command == "kill":
            return cmd_kill(args)
    except DispatchError as exc:
        print(f"dispatch: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
