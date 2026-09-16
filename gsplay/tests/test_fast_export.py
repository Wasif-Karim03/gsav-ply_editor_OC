"""Conservative fast-export eligibility and complete real-codec integration."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.gsplay.gsav_export_layout import SourceLayout
from src.models.gsav import GsavModel


def model():
    obj = GsavModel.__new__(GsavModel)
    obj.total_frames = 4
    obj.gsav_metadata = {"fps": 30, "chunk_size": 2, "chunks": [[0, 2], [2, 4]]}
    obj._raw = lambda i: {"presence": np.array([True, False])}
    return obj


def frame():
    return SimpleNamespace(
        means=torch.zeros(2, 3),
        scales=torch.ones(2, 3),
        quats=torch.ones(2, 4),
        opacities=torch.tensor([0.5, 0]),
    )


def test_layout_tracks_explicit_presence_and_slot_order():
    layout = SourceLayout(model(), range(4), 30)
    data = frame()
    np.testing.assert_array_equal(layout.after(0, layout.before(data), data), [True, False])
    assert layout.valid and layout.chunk_size == 2


@pytest.mark.parametrize("field", ["means", "scales", "quats", "opacities"])
def test_geometry_or_opacity_edits_fall_back(field):
    layout = SourceLayout(model(), range(4), 30)
    data = frame()
    original = layout.before(data)
    getattr(data, field)[0] += 1
    assert layout.after(0, original, data) is None
    assert not layout.valid


@pytest.mark.parametrize("times,fps", [([0], 30), ([3, 2, 1, 0], 30), ([0, 1, 2, 3], 24)])
def test_changed_timeline_falls_back(times, fps):
    assert not SourceLayout(model(), times, fps).valid


def test_missing_metadata_masks_and_disable_fall_back(monkeypatch):
    obj = model()
    obj.gsav_metadata["chunks"] = [[0, 1], [1, 4]]
    assert not SourceLayout(obj, range(4), 30).valid
    obj = model()
    obj._raw = lambda i: {}
    layout = SourceLayout(obj, range(4), 30)
    data = frame()
    assert layout.after(0, layout.before(data), data) is None
    monkeypatch.setenv("GSPLAY_FAST_EXPORT", "0")
    assert not SourceLayout(model(), range(4), 30).valid


@pytest.mark.integration
def test_real_gsav_fast_export_and_standard_fallback(tmp_path):
    import gsply
    from gsmod import ColorValues

    from src.gsplay.config.settings import GSPlayConfig
    from src.gsplay.core.container import create_edit_manager
    from src.gsplay.gsav_controls import write_edited_sequence
    from src.infrastructure.gsav import encode_gsav

    source = tmp_path / "source"
    source.mkdir()
    rng = np.random.default_rng(42)
    for i in range(4):
        data = gsply.GSData(
            means=rng.uniform(-1, 1, (16, 3)).astype(np.float32),
            scales=np.full((16, 3), -3, np.float32),
            quats=np.tile(np.array([1, 0, 0, 0], np.float32), (16, 1)),
            opacities=np.ones(16, np.float32),
            sh0=np.zeros((16, 3), np.float32),
            shN=np.full((16, 15, 3), 0.02 + i * 0.01, np.float32),
        )
        data.opacities[i] = -np.inf
        gsply.plywrite(source / f"frame_{i:06d}.ply", data)
    original = tmp_path / "original.gsav"
    encode_gsav(source, original, device="cpu")
    loaded = GsavModel(original, device="cpu")
    config = GSPlayConfig()
    config.edits_active = True
    config.color_values = ColorValues(brightness=0.7)
    manager = create_edit_manager(config, "cpu")
    messages = []
    output = tmp_path / "fast.gsav"
    try:
        result = write_edited_sequence(
            loaded,
            list(range(4)),
            manager.apply_edits,
            output,
            fps=30,
            device="cpu",
            status=messages.append,
        )
        assert result["frames"] == 4
        assert "Original GSAV arrangement preserved" in messages
        replay = GsavModel(output, device="cpu")
        try:
            for i in range(4):
                a = loaded._raw(i)
                b = replay._raw(i)
                np.testing.assert_array_equal(a["presence"], b["presence"])
                mask = a["presence"]
                np.testing.assert_allclose(a["means"][mask], b["means"][mask], atol=0.003)
                expected = (a["sh0"][mask] * 0.28209479177387814 + 0.5) * 0.7
                np.testing.assert_allclose(
                    b["sh0"][mask] * 0.28209479177387814 + 0.5, expected, atol=0.01
                )
                np.testing.assert_allclose(b["shN"][mask], a["shN"][mask] * 0.7, atol=0.01)
        finally:
            replay.on_shutdown()

        # A geometry edit is detected even when performed by an arbitrary callback.
        def translated(data):
            data.means = data.means + 1
            data._base = None
            return data

        messages = []
        write_edited_sequence(
            loaded,
            list(range(4)),
            translated,
            tmp_path / "fallback.gsav",
            fps=30,
            device="cpu",
            status=messages.append,
        )
        assert "Geometry, opacity, filtering or row order changed" in messages
    finally:
        loaded.on_shutdown()
