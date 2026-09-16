import os

import numpy as np
import pytest

from src.infrastructure.frame_buffer import FrameBuffer


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows named mapping")


def test_shared_mapping_preserves_values_and_detaches_returned_arrays():
    owner = FrameBuffer(1024)
    worker = FrameBuffer(owner.size, owner.name)
    try:
        arrays = {
            "means": np.arange(12, dtype=np.float32).reshape(4, 3),
            "presence": np.array([True, False, True, False]),
            "shN": np.empty((4, 0, 3), dtype=np.float32),
        }
        fields = worker.write(arrays)
        copied = owner.read(fields)
        for name, value in arrays.items():
            np.testing.assert_array_equal(copied[name], value)
        arrays["means"][:] = 99
        worker.write(arrays)
        assert copied["means"][0, 0] == 0
        assert owner.read(fields)["means"][0, 0] == 99
        with pytest.raises(ValueError, match="bounds"):
            owner.read({"means": ["<f4", [1000, 3], 0]})
        with pytest.raises(ValueError, match="dtype"):
            owner.read({"means": ["O", [4, 3], 0]})
    finally:
        worker.close()
        owner.close()


def test_model_tensor_cache_is_pristine_and_keeps_sh_order(monkeypatch, tmp_path):
    from src.models.gsav import GsavModel

    values = {
        "means": np.ones((2, 3), np.float32),
        "scales": np.zeros((2, 3), np.float32),
        "quats": np.array([[1, 0, 0, 0]] * 2, np.float32),
        "opacities": np.zeros(2, np.float32),
        "sh0": np.zeros((2, 3), np.float32),
        "shN": np.arange(90, dtype=np.float32).reshape(2, 15, 3),
    }

    class Stream:
        metadata = {"frames": 2, "fps": 30}

        def __init__(self, path):
            pass

        def frame(self, index):
            return {k: v.copy() for k, v in values.items()}

    monkeypatch.setattr("src.models.gsav.GsavStream", Stream)
    model = GsavModel(tmp_path / "mock.gsav", device="cpu")
    model.processing_mode = "all_gpu"
    first = model.get_gaussians_at_normalized_time(0)
    np.testing.assert_array_equal(first.shN.numpy(), values["shN"])
    first.means[:] = 99
    first.shN[:] = 99
    again = model.get_gaussians_at_normalized_time(0)
    np.testing.assert_array_equal(again.means.numpy(), values["means"])
    np.testing.assert_array_equal(again.shN.numpy(), values["shN"])
    assert again._format == first._format
    model.get_gaussians_at_normalized_time(1)
    assert model._gpu_cache[0][0] == 1
