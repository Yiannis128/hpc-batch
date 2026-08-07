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

    def test_shared_nodes_are_not_cards(self, tmp_path):
        # Masking nvidiactl or nvidia-uvm breaks the GPUs the job *was* given,
        # so they must never be mistaken for a card.
        dev = make_dev(tmp_path, DEV_NAMES)
        assert not any(
            path.name.startswith(("nvidiactl", "nvidia-uvm", "nvidia-modeset", "nvidia-caps"))
            for path in gpu_nodes(dev).values()
        )

    def test_machine_without_gpus(self, tmp_path):
        assert gpu_nodes(make_dev(tmp_path, ["null"])) == {}

    def test_missing_dev_is_not_an_error(self, tmp_path):
        assert gpu_nodes(tmp_path / "nope") == {}


class TestNodesToMask:
    def test_masks_the_complement(self, tmp_path):
        nodes = gpu_nodes(make_dev(tmp_path, DEV_NAMES))
        assert [p.name for p in nodes_to_mask([0, 1], nodes)] == ["nvidia2", "nvidia3"]

    def test_no_gpus_masks_every_card(self, tmp_path):
        # "none" is an allocation too: a cpu-only job left able to open a card
        # takes one the scheduler still believes is free.
        nodes = gpu_nodes(make_dev(tmp_path, DEV_NAMES))
        assert len(nodes_to_mask([], nodes)) == 4

    def test_all_gpus_masks_nothing(self, tmp_path):
        nodes = gpu_nodes(make_dev(tmp_path, DEV_NAMES))
        assert nodes_to_mask([0, 1, 2, 3], nodes) == []

    def test_allocation_naming_an_absent_card(self, tmp_path):
        nodes = gpu_nodes(make_dev(tmp_path, ["nvidia0", "nvidia1"]))
        assert [p.name for p in nodes_to_mask([0, 7], nodes)] == ["nvidia1"]


class TestMaskForeignGpus:
    def test_nothing_to_mask_makes_no_syscall(self, tmp_path):
        # The early return is what keeps a fully-allocated job out of a mount
        # namespace it does not need; without it this would need root.
        mask_foreign_gpus([0, 1], make_dev(tmp_path, ["nvidia0", "nvidia1"]))

    def test_no_gpus_on_the_machine(self, tmp_path):
        mask_foreign_gpus([], make_dev(tmp_path, ["null"]))


class TestCheckSupported:
    def test_refuses_without_root_and_names_the_flag(self, monkeypatch):
        monkeypatch.setattr("os.geteuid", lambda: 1000)
        with pytest.raises(GpuMaskError, match="--no-gpu-mask"):
            check_supported([0, 1])

    def test_no_gpus_needs_nothing(self, monkeypatch):
        monkeypatch.setattr("os.geteuid", lambda: 1000)
        check_supported([])

    def test_root_is_fine(self, monkeypatch):
        monkeypatch.setattr("os.geteuid", lambda: 0)
        check_supported([0, 1])
