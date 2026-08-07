import errno
import os

import pytest

from hpc_batch.devices import (
    GpuMaskError,
    check_supported,
    gpu_nodes,
    mask_foreign_gpus,
    nodes_to_mask,
)


def make_dev(tmp_path, names):
    for name in names:
        (tmp_path / name).touch()
    return tmp_path


#: What a 4-gpu box actually has: four per-card nodes plus the shared ones.
DEV_NAMES = [
    "nvidia0", "nvidia1", "nvidia2", "nvidia3",
    "nvidiactl", "nvidia-uvm", "nvidia-uvm-tools", "nvidia-modeset",
    "nvidia-caps", "null", "zero",
]


class TestGpuNodes:
    def test_only_per_card_nodes(self, tmp_path):
        dev = make_dev(tmp_path, DEV_NAMES)
        assert sorted(gpu_nodes(dev)) == [0, 1, 2, 3]

    def test_machine_without_gpus(self, tmp_path):
        assert gpu_nodes(make_dev(tmp_path, ["null"])) == {}

    def test_missing_dev_is_not_an_error(self, tmp_path):
        assert gpu_nodes(tmp_path / "nope") == {}


class TestNodesToMask:
    def test_masks_the_complement(self, tmp_path):
        nodes = gpu_nodes(make_dev(tmp_path, DEV_NAMES))
        assert [p.name for p in nodes_to_mask([0, 1], nodes)] == ["nvidia2", "nvidia3"]

    def test_no_gpus_masks_every_card(self, tmp_path):
        nodes = gpu_nodes(make_dev(tmp_path, DEV_NAMES))
        assert len(nodes_to_mask([], nodes)) == 4

    def test_all_gpus_masks_nothing(self, tmp_path):
        nodes = gpu_nodes(make_dev(tmp_path, DEV_NAMES))
        assert nodes_to_mask([0, 1, 2, 3], nodes) == []

    def test_allocation_naming_an_absent_card(self, tmp_path):
        nodes = gpu_nodes(make_dev(tmp_path, ["nvidia0", "nvidia1"]))
        assert [p.name for p in nodes_to_mask([0, 7], nodes)] == ["nvidia1"]


class TestMaskForeignGpus:
    def test_nothing_to_mask_makes_no_syscall(self):
        # The early return is what keeps a fully-allocated job out of a mount
        # namespace it does not need; without it this would need root.
        mask_foreign_gpus([])


class TestCheckSupported:
    def test_refusal_names_the_flag(self, monkeypatch):
        def denied():
            raise OSError(errno.EPERM, "unshare")

        monkeypatch.setattr("hpc_batch.devices._unshare_private", denied)
        with pytest.raises(GpuMaskError, match="--no-gpu-mask"):
            check_supported()

    def test_reports_the_reason_it_was_refused(self, monkeypatch):
        # The errno survives the probe child, which is the point of probing
        # rather than inferring the answer from the uid.
        def denied():
            raise OSError(errno.ENOSYS, "unshare")

        monkeypatch.setattr("hpc_batch.devices._unshare_private", denied)
        with pytest.raises(GpuMaskError, match=os.strerror(errno.ENOSYS)):
            check_supported()

    def test_silent_when_it_works(self, monkeypatch):
        monkeypatch.setattr("hpc_batch.devices._unshare_private", lambda: None)
        check_supported()
