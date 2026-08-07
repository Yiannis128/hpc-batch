"""Hide the GPUs a job was not given, in a mount namespace of its own.

`CUDA_VISIBLE_DEVICES` decides what a job's CUDA runtime *will* use, which is
not the same as what it *can* use: the job may rewrite the variable, and a
container runtime told `--device nvidia.com/gpu=N` skips it entirely by
injecting the device node itself. Masking the nodes a job was not allocated
closes both, because the driver enumerates by probing `/dev/nvidiaN`: a card
whose node is `/dev/null` is invisible to NVML and to CUDA alike, so
`nvidia-smi` inside a job agrees with the allocation instead of showing the
whole machine.

Enforcement only, like `cgroup.py`: the allocator decides which GPUs a job
holds, this makes the rest unreachable.
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


def _refuse(why: str) -> GpuMaskError:
    """Every refusal names the flag that opts out of it, as in `cgroup.py`."""
    return GpuMaskError(f"{why}; {_NO_GPU_MASK}")


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


def _libc() -> ctypes.CDLL:
    lib = ctypes.CDLL("libc.so.6", use_errno=True)
    lib.unshare.argtypes = [ctypes.c_int]
    lib.mount.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_ulong, ctypes.c_void_p,
    ]
    return lib


def _checked(rc: int, what: str) -> None:
    if rc != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"{what}: {os.strerror(err)}")


def mask_foreign_gpus(allowed: list[int], dev: Path = Path("/dev")) -> None:
    """Unshare a mount namespace and mask every GPU node outside `allowed`.

    Call between fork and exec, while still root: unsharing needs
    CAP_SYS_ADMIN, so this has to happen before privileges are dropped, and
    the masks have to be in place before the job's first instruction.
    """
    targets = nodes_to_mask(allowed, gpu_nodes(dev))
    if not targets:
        return
    lib = _libc()
    _checked(lib.unshare(CLONE_NEWNS), "unshare")
    # Private *before* any bind. Without this the masks propagate back to the
    # host and hide the GPUs from every other job and from this daemon.
    _checked(lib.mount(None, b"/", None, MS_REC | MS_PRIVATE, None), "make-rprivate")
    for path in targets:
        _checked(
            lib.mount(b"/dev/null", os.fsencode(path), None, MS_BIND, None),
            f"mask {path}",
        )


def check_supported(gpu_ids: list[int]) -> None:
    """Refuse at startup rather than at the first job that needs it."""
    if gpu_ids and os.geteuid() != 0:
        raise _refuse("masking unallocated GPUs needs root to unshare a mount namespace")
