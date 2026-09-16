"""Cross-repository regression checks from the full pipeline audit."""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import gsply
import numpy as np
import pytest

from src.domain.data import GaussianData
from src.gsplay.core.components.model_component import ModelComponent
from src.gsplay.gsav_controls import write_edited_sequence
from src.infrastructure.gsav import GsavError, codec_python
from src.models.gsav import GsavModel


def test_unload_releases_uploads_and_metadata(tmp_path):
    component = ModelComponent(device="cpu")
    directory = tempfile.TemporaryDirectory(dir=tmp_path)
    upload = Path(directory.name) / "uploaded.gsav"
    upload.write_bytes(b"owned temporary upload")
    previous = Mock()
    component.model = previous
    component.source_path = upload
    component.gsav_metadata = {"audio": "old-audio.ogg", "fps": 30}
    component._gsav_directories.append(directory)
    component.unload()
    previous.on_shutdown.assert_called_once()
    assert not upload.exists()
    assert component.source_path is None and component.gsav_metadata == {}
    assert component._gsav_directories == []


def test_reload_retains_current_upload(tmp_path):
    component = ModelComponent(device="cpu")
    directory = tempfile.TemporaryDirectory(dir=tmp_path)
    upload = Path(directory.name) / "uploaded.gsav"
    upload.write_bytes(b"source")
    component.model = Mock()
    component.source_path = upload
    component._gsav_directories.append(directory)
    previous = Mock()
    component._release_previous(previous)
    assert upload.read_bytes() == b"source"
    previous.on_shutdown.assert_called_once()
    component.unload()


def test_config_switch_clears_gsav_metadata_and_closes_ply(monkeypatch):
    from src.infrastructure.model_factory import ModelFactory

    component = ModelComponent(device="cpu")
    previous = Mock()
    component.model = previous
    component.gsav_metadata = {"audio": "stale.ogg"}
    candidate = Mock()
    monkeypatch.setattr(ModelFactory, "create", lambda **kwargs: (candidate, None, {}))
    component.load_from_config({"module": "load-ply", "config": {}})
    assert component.gsav_metadata == {}
    previous.on_shutdown.assert_called_once()


def test_config_switch_updates_renderer_before_closing_source(monkeypatch):
    from src.gsplay.interaction.events import EventBus, EventType
    from src.infrastructure.model_factory import ModelFactory

    bus = EventBus()
    component = ModelComponent(device="cpu", event_bus=bus)
    previous, candidate = Mock(), Mock()
    component.model = previous
    rendered_source = [previous]

    def replace_renderer(event):
        previous.on_shutdown.assert_not_called()
        rendered_source[0] = component.model

    def close_previous():
        assert rendered_source[0] is candidate

    previous.on_shutdown.side_effect = close_previous
    bus.subscribe(EventType.MODEL_LOADED, replace_renderer)
    monkeypatch.setattr(ModelFactory, "create", lambda **kwargs: (candidate, None, {}))
    component.load_from_config({"module": "load-ply", "config": {}})
    previous.on_shutdown.assert_called_once()


@pytest.mark.integration
def test_zero_opacity_survives_ply_gsav_and_reload(tmp_path):
    try:
        codec_python()
    except GsavError:
        pytest.skip("Isolated codec unavailable")
    rng = np.random.default_rng(19)
    n = 32
    raw = gsply.GSData(
        means=rng.uniform(-1, 1, (n, 3)).astype(np.float32),
        scales=np.full((n, 3), -3, np.float32),
        quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
        opacities=np.full(n, -1, np.float32),
        sh0=rng.uniform(-1, 1, (n, 3)).astype(np.float32),
        shN=None,
    )
    raw.opacities[:8] = -np.inf
    active = raw.denormalize(inplace=False).to_rgb(inplace=False)
    model = SimpleNamespace(get_frame_at_source_time=lambda t: GaussianData.from_gsdata(active))
    for fmt in ("PLY", "GSAV"):
        output = tmp_path / ("visible.gsav" if fmt == "GSAV" else "ply")
        write_edited_sequence(
            model, [0, 1], lambda data: data, output, fps=30, device="cpu", output_format=fmt
        )
        if fmt == "PLY":
            saved = gsply.plyread(output / "frame_000000.ply")
            assert np.isneginf(saved.opacities).sum() == 8
        else:
            loaded = GsavModel(output, device="cpu")
            try:
                for frame in (0, 1):
                    data = loaded.get_frame_at_source_time(frame).to_gsdata()
                    assert np.count_nonzero(data.opacities == 0) == 8
                    assert np.count_nonzero(data.opacities > 0) == 24
            finally:
                loaded.on_shutdown()


@pytest.mark.parametrize(
    "mode", ["all_cpu", "all_gpu", "color_gpu", "transform_gpu", "color_transform_gpu"]
)
@pytest.mark.parametrize("degree", [0, 3])
def test_combined_edits_export_consistently(tmp_path, mode, degree):
    import torch
    from gsmod import ColorValues, TransformValues

    from src.gsplay.config.settings import GSPlayConfig
    from src.gsplay.core.container import create_edit_manager
    from src.infrastructure.exporters.ply_exporter import PlyExporter

    rng = np.random.default_rng(11)
    n = 32
    colors = rng.uniform(0.1, 0.8, (n, 3)).astype(np.float32)
    raw = gsply.GSData(
        means=rng.uniform(-1, 1, (n, 3)).astype(np.float32),
        scales=np.full((n, 3), -3, np.float32),
        quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
        opacities=np.full(n, -1, np.float32),
        sh0=(colors - 0.5) / 0.28209479177387814,
        shN=np.full((n, 15, 3), 0.02, np.float32) if degree else None,
    )
    active = raw.denormalize(inplace=False)
    if not degree:
        active = active.to_rgb(inplace=False)
    original = active.means.copy()
    config = GSPlayConfig()
    config.edits_active = True
    config.volume_filter.processing_mode = mode
    config.transform_values = TransformValues(
        scale=(1.2, 1.2, 1.2), translation=(0.1, 0.2, 0.3), center=(0, 0, 0)
    )
    config.color_values = ColorValues(brightness=0.7)
    config.alpha_scaler = 0.6
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    manager = create_edit_manager(config, device)
    edited = manager.apply_edits(GaussianData.from_gsdata(active).to_gstensor(device))
    output = tmp_path / "edited.ply"
    PlyExporter().export_frame(edited, output)
    saved = gsply.plyread(output)
    np.testing.assert_allclose(saved.means, original * 1.2 + [0.1, 0.2, 0.3], atol=1e-5)
    np.testing.assert_allclose(saved.scales, raw.scales + np.log(1.2), atol=1e-5)
    np.testing.assert_allclose(
        1 / (1 + np.exp(-saved.opacities)), (1 / (1 + np.exp(1))) * 0.6, atol=1e-5
    )
    np.testing.assert_allclose(saved.sh0, (colors * 0.7 - 0.5) / 0.28209479177387814, atol=0.002)
    if degree:
        np.testing.assert_allclose(saved.shN, raw.shN * 0.7, atol=0.002)
    np.testing.assert_array_equal(active.means, original)
