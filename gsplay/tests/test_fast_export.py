"""Conservative fast-export eligibility and complete real-codec integration."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.gsplay.gsav_export_layout import SourceLayout
from src.models.gsav import GsavModel


@pytest.mark.parametrize("count", [1, 4, 180, 600, 10000])
def test_original_frame_selection_preserves_exact_gsav_indices(count):
    from src.gsplay.gsav_controls import export_times

    obj = model()
    obj.total_frames = count
    obj.source_fps = 30
    obj.gsav_metadata = {
        "fps": 30,
        "chunk_size": 30,
        "chunks": [[i, min(i + 30, count)] for i in range(0, count, 30)],
    }
    times = export_times(obj, "Original Frames", 0)
    assert times == list(range(count))
    assert all(isinstance(t, int) for t in times)
    assert SourceLayout(obj, times, 30).valid
    if count == 600:
        old_times = [obj.time_domain.from_normalized(obj.get_frame_time(i)) for i in range(count)]
        assert old_times != times  # Reproduces the actual browser-path regression.


def test_custom_frame_selection_is_not_silently_rounded():
    from src.gsplay.gsav_controls import export_times

    obj = model()
    obj.source_fps = 30
    times = export_times(obj, "Custom Time Range", 0, 0, 3, 0.5)
    assert times == [0, 0.5, 1, 1.5, 2, 2.5, 3]
    assert not SourceLayout(obj, times, 30).valid


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
def test_real_gsav_fast_export_and_standard_fallback(tmp_path, monkeypatch):
    import gsply
    from gsmod import ColorValues

    from src.gsplay.config.settings import GSPlayConfig
    from src.gsplay.core.container import create_edit_manager
    from src.gsplay.gsav_controls import export_times, write_edited_sequence
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
            export_times(loaded, "Original Frames", 0),
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
                for field in ("means", "scales", "quats", "opacities"):
                    np.testing.assert_array_equal(a[field], b[field])
                mask = a["presence"]
                np.testing.assert_allclose(a["means"][mask], b["means"][mask], atol=0.003)
                expected = (a["sh0"][mask] * 0.28209479177387814 + 0.5) * 0.7
                np.testing.assert_allclose(
                    b["sh0"][mask] * 0.28209479177387814 + 0.5, expected, atol=0.01
                )
                np.testing.assert_allclose(b["shN"][mask], a["shN"][mask] * 0.7, atol=0.01)
        finally:
            replay.on_shutdown()

        from gsmod import FilterValues

        from src.gsplay.gsav_visibility import create_export_manager

        config.filter_values = FilterValues(sphere_radius=0.9)
        filtered_manager, visibility = create_export_manager(
            config, "cpu", loaded, range(4), 30, "GSAV"
        )
        assert visibility is not None
        filtered_path = tmp_path / "filtered.gsav"
        messages = []
        write_edited_sequence(
            loaded,
            list(range(4)),
            filtered_manager.apply_edits,
            filtered_path,
            fps=30,
            device="cpu",
            visibility=visibility,
            status=messages.append,
        )
        assert "Original GSAV arrangement preserved" in messages
        filtered = GsavModel(filtered_path, device="cpu")
        try:
            for i in range(4):
                data = loaded.get_frame_at_source_time(i).to_gstensor("cpu")
                preview = create_edit_manager(config, "cpu").apply_edits(data)
                actual = filtered._raw(i)
                active = actual["presence"]
                expected = preview.means[preview.opacities.reshape(-1) > 0].numpy()
                np.testing.assert_array_equal(actual["means"][active], expected)
                assert len(actual["means"]) <= len(loaded._raw(i)["means"])
        finally:
            filtered.on_shutdown()

        # Crop-only transport must never serialize edited PLY frames.
        from src.infrastructure.exporters.ply_exporter import PlyExporter

        config.color_values = ColorValues()
        crop_manager, crop_visibility = create_export_manager(
            config, "cpu", loaded, range(4), 30, "GSAV"
        )
        assert crop_visibility.crop_only

        def forbidden_ply(*args, **kwargs):
            raise AssertionError("Crop-only export must skip PLY transport")

        with monkeypatch.context() as patch:
            patch.setattr(PlyExporter, "export_frame", forbidden_ply)
            write_edited_sequence(
                loaded,
                list(range(4)),
                crop_manager.apply_edits,
                tmp_path / "crop.gsav",
                fps=30,
                device="cpu",
                visibility=crop_visibility,
            )
        crop = GsavModel(tmp_path / "crop.gsav", device="cpu")
        try:
            for i in range(4):
                crop_manager.apply_edits(loaded.get_frame_at_source_time(i).to_gstensor("cpu"))
                original = loaded._raw(i)
                expected_mask = original["presence"] & crop_visibility.mask
                actual = crop._raw(i)
                for field in ("means", "scales", "quats", "opacities", "sh0", "shN"):
                    np.testing.assert_array_equal(
                        original[field][expected_mask], actual[field][actual["presence"]]
                    )
        finally:
            crop.on_shutdown()

        # A geometry edit is detected even when performed by an arbitrary callback.
        def translated(data):
            data.means = data.means + 1
            data._base = None
            return data

        messages = []
        write_edited_sequence(
            loaded,
            export_times(loaded, "Original Frames", 0),
            translated,
            tmp_path / "fallback.gsav",
            fps=30,
            device="cpu",
            status=messages.append,
        )
        assert "Geometry, opacity, filtering or row order changed" in messages
    finally:
        loaded.on_shutdown()
