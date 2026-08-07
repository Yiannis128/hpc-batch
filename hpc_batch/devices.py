"""Mask the GPU device nodes a job was not allocated.

Enforcement only, like `cgroup.py`: the allocator decides which GPUs a job
holds, this makes the rest unopenable. See CLAUDE.md for why the environment
variable alone is not enough, and for the two orderings that are load-bearing.
"""

import ctypes
import os
import re
from pathlib import Path

_NO_GPU_MASK = "pass --no-gpu-mask to run without it"

#: Per-card nodes only. `nvidiactl`, `nvidia-uvm`, `nvidia-uvm-tools`,
#: `nvidia-modeset` and the `nvidia-caps` entries name no card and every job
#: needs them; masking those breaks the GPUs a job *was* given.
_GPU_NODE = re.compile(r"^nvidia(\d+)$")

CLONE_NEWNS = 0x00020000
MS_BIND = 0x1000
MS_REC = 0x4000
MS_PRIVATE = 0x40000


class GpuMaskError(Exception):
    pass


def _load_libc() -> ctypes.CDLL:
    lib = ctypes.CDLL("libc.so.6", use_errno=True)
    lib.unshare.argtypes = [ctypes.c_int]
    lib.mount.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_ulong, ctypes.c_void_p,
    ]
    return lib


# Bound at import so the child never has to dlopen: `mask_foreign_gpus` runs
# between fork and exec, where taking the loader lock is not safe.
_LIBC = _load_libc()


def gpu_nodes(dev: Path = Path("/dev")) -> dict[int, Path]:
    """GPU index -> device node, for the per-card nodes present here."""
    try:
        names = os.listdir(dev)
    except OSError:
        return {}
    found = {}
    for name in names:
        match = _GPU_NODE.match(name)
        if match:
            found[int(match.group(1))] = dev / name
    return found


def nodes_to_mask(allowed: list[int], nodes: dict[int, Path]) -> list[Path]:
    """The nodes a job holding `allowed` must not be able to open.

    A job given no GPUs masks all of them: "none" is an allocation too, and
    leaving the nodes readable would let a cpu-only job take a card the
    scheduler still believes is free.
    """
    keep = set(allowed)
    return [path for index, path in sorted(nodes.items()) if index not in keep]


def _checked(rc: int, what: str) -> None:
    if rc != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"{what}: {os.strerror(err)}")


def _unshare_private() -> None:
    """Enter a private mount namespace.

    Private *before* any bind, or the masks propagate back to the host and
    hide the GPUs from every other job and from this daemon.
    """
    _checked(_LIBC.unshare(CLONE_NEWNS), "unshare")
    _checked(_LIBC.mount(None, b"/", None, MS_REC | MS_PRIVATE, None), "make-rprivate")


def mask_foreign_gpus(targets: list[Path]) -> None:
    """Make `targets` unopenable, in a mount namespace of this process's own.

    Call between fork and exec, while still root: unsharing needs
    CAP_SYS_ADMIN, so it has to happen before privileges are dropped. Takes
    the nodes rather than the allocation so the parent does the listing and
    the child does nothing but syscalls.
    """
    if not targets:
        return
    _unshare_private()
    for path in targets:
        _checked(
            _LIBC.mount(b"/dev/null", os.fsencode(path), None, MS_BIND, None),
            f"mask {path}",
        )


def check_supported() -> None:
    """Refuse at startup rather than at the first job that needs it.

    Attempts the unshare in a throwaway child instead of inferring it from
    the uid: root without CAP_SYS_ADMIN, or a seccomp filter, would pass a
    uid test and then fail every spawn.
    """
    pid = os.fork()
    if pid == 0:
        try:
            _unshare_private()
        except OSError as exc:
            os._exit(min(exc.errno or 1, 125))
        os._exit(0)
    _, status = os.waitpid(pid, 0)
    code = os.waitstatus_to_exitcode(status)
    if code != 0:
        raise GpuMaskError(
            f"cannot unshare a mount namespace to hide unallocated GPUs "
            f"({os.strerror(code)}); {_NO_GPU_MASK}"
        )
