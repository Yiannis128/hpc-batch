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

from . import __version__
from .protocol import MAX_LINE, QUEUED, encode, socket_path
from .util import duration_arg, format_duration, format_table


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
    req = {
        "cmd": "new",
        "argv": command,
        "cwd": os.getcwd(),
        "cpu": args.cpu,
        "gpu_cores": args.gpu_cores,
        "max_mem_gb": args.max_mem,
        "max_time_s": args.max_time,
        "exclusive": args.exclusive,
    }
    resp = _request(req)
    print(
        f"job {resp['id']} {resp['state']} "
        f"(max time {format_duration(resp.get('max_time_s'))})"
    )
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    resp = _request({"cmd": "list", "all": args.all})
    jobs = resp.get("jobs", [])
    if not jobs:
        print("no jobs")
        return 0
    rows = []
    for job in jobs:
        uptime = "queued" if job["state"] == QUEUED else format_duration(job["uptime_s"])
        rows.append(
            [
                job["user"],
                str(job["id"]),
                job["command"],
                uptime,
                format_duration(job["max_time_s"]),
                "yes" if job["exclusive"] else "no",
            ]
        )
    print(format_table(["USER", "ID", "COMMAND", "UPTIME", "MAX-TIME", "EXCLUSIVE"], rows))
    return 0


def cmd_attach(args: argparse.Namespace) -> int:
    with _connect() as sock:
        sock.sendall(encode({"cmd": "attach", "id": args.id}))
        f = sock.makefile("rb")
        resp = _read_response(f)
        if resp.get("state") == QUEUED:
            print(f"job {args.id} is queued; waiting for it to start...", file=sys.stderr)
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
    p_new.add_argument("--cpu", type=int, default=1, metavar="N",
                       help="cpu cores to allocate, all from one NUMA node (default: 1)")
    p_new.add_argument("--gpu-cores", type=int, default=0, metavar="N",
                       help="how many of the nvidia-smi -L gpus to allocate (default: 0)")
    p_new.add_argument("--max-mem", type=float, default=None, metavar="GB",
                       help="memory limit in GiB (default: no limit)")
    p_new.add_argument("--max-time", type=duration_arg, default=None, metavar="DURATION",
                       help="kill the job after this long, e.g. 30m or 2h "
                            "(default and upper bound: the admin's max lifetime)")
    p_new.add_argument("--exclusive", action="store_true",
                       help="run alone: wait for an idle machine and block others while running")

    p_list = sub.add_parser(
        "list",
        help="list current jobs",
        description="List queued and running jobs: "
                    "<username> <id> <command> <uptime> <max-time> <exclusive>.",
    )
    p_list.add_argument(
        "--all", action="store_true",
        help="list every user's jobs (admins; everyone if the daemon "
             "was started with --list-is-public)",
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
