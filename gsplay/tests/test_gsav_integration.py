"""Real isolated-codec round trip plus boundary regressions (optional codec runtime)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.domain.entities import GSTensor
from src.domain.time import TimeDomain
from src.gsplay.gsav_controls import export_times, write_edited_gsav, write_edited_sequence
from src.infrastructure.gsav import GsavError, codec_python, decode_gsav, encode_gsav, validate_gsav


def test_bad_gsav_rejected_before_codec(tmp_path):
    path = tmp_path / "bad.gsav"
    path.write_bytes(b"not a gsav")
    with pytest.raises(GsavError, match="header"):
        validate_gsav(path)


def test_legacy_degree_without_sidecar():
    from src.infrastructure.gsav_worker import stored_sh_degree

    header = {"version": 1, "flags": 0, "sh_bands": 3, "sh_payload_offset": 0}
    assert stored_sh_degree(header) == 0
    assert stored_sh_degree(dict(header, flags=2, sh_payload_offset=128)) == 3
    # A file claiming a stored payload must still undergo strict validation.
    assert stored_sh_degree(dict(header, flags=2)) == 3
    assert stored_sh_degree(dict(header, version=3)) == 3


def test_missing_codec_is_actionable(monkeypatch, tmp_path):
    monkeypatch.setenv("GSPLAY_CODEC_PYTHON", str(tmp_path / "absent.exe"))
    with pytest.raises(GsavError, match="GSPLAY_CODEC_PYTHON"):
        codec_python()


def test_time_selection_and_limits():
    model = SimpleNamespace(
        time_domain=TimeDomain.discrete(3, source_fps=24),
        get_total_frames=lambda: 3,
        get_frame_time=lambda i: i / 2,
    )
    assert export_times(model, "Original Frames", 1) == [0, 1, 2]
    assert export_times(model, "Snapshot at Current Time", 1) == [1]
    assert export_times(model, "Custom Time Range", 0, 0, 2, 1) == [0, 1, 2]
    with pytest.raises(GsavError, match="positive"):
        export_times(model, "Custom Time Range", 0, 0, 2, 0)
    with pytest.raises(GsavError, match="10,000"):
        export_times(model, "Custom Time Range", 0, 0, 2, 0.00001)


def test_export_does_not_overwrite(tmp_path):
    path = tmp_path / "original.gsav"
    path.write_bytes(b"original")
    with pytest.raises(GsavError, match="exists"):
        encode_gsav(tmp_path, path)
    assert path.read_bytes() == b"original"


def test_failed_codec_does_not_publish(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise GsavError("synthetic failure")

    monkeypatch.setattr("src.infrastructure.gsav._run", fail)
    target = tmp_path / "failed.gsav"
    with pytest.raises(GsavError, match="synthetic"):
        encode_gsav(tmp_path, target)
    assert not target.exists()
    assert not list(tmp_path.glob(".gsav-export-*"))


def test_ply_codec_failure_and_existing_folder_are_safe(monkeypatch, tmp_path):
    from src.infrastructure.gsav import export_ply

    existing = tmp_path / "existing"
    existing.mkdir()
    original = existing / "frame.ply"
    original.write_bytes(b"original")
    with pytest.raises(GsavError, match="already exists"):
        export_ply(tmp_path, existing)
    assert original.read_bytes() == b"original"

    def fail(*args, **kwargs):
        raise GsavError("synthetic failure")

    monkeypatch.setattr("src.infrastructure.gsav._run", fail)
    target = tmp_path / "failed-ply"
    with pytest.raises(GsavError, match="synthetic"):
        export_ply(tmp_path, target)
    assert not target.exists()
    assert not list(tmp_path.glob(".ply-export-*"))


def test_direct_cache_is_bounded_and_returns_copies(monkeypatch, tmp_path):
    from src.models.gsav import GsavModel

    class FakeStream:
        metadata = {"frames": 20, "fps": 30}

        def __init__(self, path):
            self.calls = []

        def frame(self, index):
            self.calls.append(index)
            return {"means": np.full((2, 3), index, dtype=np.float32)}

    monkeypatch.setattr("src.models.gsav.GsavStream", FakeStream)
    model = GsavModel(tmp_path / "fake.gsav")
    model._raw(0)["means"][:] = 99
    assert model._raw(0)["means"][0, 0] == 0
    assert model._stream.calls == [0]
    for index in range(20):
        model._raw(index)
    assert len(model._cache) <= 4
    assert model._cache_bytes == sum(a.nbytes for v in model._cache.values() for a in v.values())


@pytest.mark.integration
def test_sh3_edited_sequence_round_trip(tmp_path):
    try:
        codec_python()
    except GsavError:
        pytest.skip("Configure the isolated gs-encoder v3 environment")
    import gsply

    rng = np.random.default_rng(7)
    count = 32
    means = torch.tensor(rng.uniform(-1, 1, (count, 3)), dtype=torch.float32)
    # Repeated SH profiles keep the test small while exercising palette + labels.
    profiles = rng.normal(0, 0.1, (4, 15, 3)).astype(np.float32)
    harmonics = torch.from_numpy(profiles[np.arange(count) % 4])
    frames = [
        GSTensor(
            means=means + i * 0.05,
            scales=torch.full((count, 3), -2.0),
            quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
            opacities=torch.full((count,), 2.0),
            sh0=torch.zeros(count, 3),
            shN=harmonics.clone(),
        )
        for i in range(2)
    ]
    from src.domain.data import GaussianData

    model = SimpleNamespace(
        get_frame_at_source_time=lambda t: GaussianData.from_gstensor(frames[int(t)])
    )
    from gsmod import TransformValues

    from src.gsplay.config.settings import GSPlayConfig
    from src.gsplay.core.container import create_edit_manager

    config = GSPlayConfig()
    config.edits_active = True
    config.volume_filter.processing_mode = "all_cpu"
    config.transform_values = TransformValues(translation=(2.0, 0.0, 0.0))
    manager = create_edit_manager(config, "cpu")
    edited_means = [(frame.means + torch.tensor([2.0, 0.0, 0.0])).numpy() for frame in frames]

    def edit(frame):
        return manager.apply_edits(frame)

    output = tmp_path / "edited.gsav"
    result = write_edited_gsav(model, [0.0, 1.0], edit, output, fps=24, device="cpu")
    assert result == {"frames": 2, "fps": 24, "sh_bands": 3}
    ply_output = tmp_path / "edited-ply"
    ply_result = write_edited_sequence(
        model, [0.0, 1.0], edit, ply_output, fps=24, device="cpu", output_format="PLY"
    )
    assert ply_result == {"frames": 2, "format": "ply"}
    for i, path in enumerate(sorted(ply_output.glob("*.ply"))):
        frame = gsply.plyread(path)
        np.testing.assert_allclose(frame.means, edited_means[i], atol=1e-6)
        np.testing.assert_array_equal(frame.shN, harmonics.numpy())
    destination = tmp_path / "decoded"
    metadata = decode_gsav(output, destination)
    assert metadata["frames"] == 2 and metadata["fps"] == 24 and metadata["sh_bands"] == 3
    for i, path in enumerate(sorted(destination.glob("*.ply"))):
        frame = gsply.plyread(path)
        assert frame.shN.shape == (count, 15, 3)
        # Morton sorting/matching may reorder rows: compare corresponding positions.
        distances = np.linalg.norm(frame.means[:, None] - edited_means[i][None], axis=-1)
        matches = distances.argmin(axis=1)
        assert distances.min(axis=1).max() < 0.02
        assert np.unique(matches).size == count
        assert np.abs(frame.shN - harmonics.numpy()[matches]).mean() < 0.03
    # Original model data remains untouched by export.
    torch.testing.assert_close(frames[0].means, means)

    reencoded = tmp_path / "again.gsav"
    encode_gsav(destination, reencoded, fps=24, device="cpu")
    second = decode_gsav(reencoded, tmp_path / "decoded-again")
    assert second["sh_bands"] == 3 and second["frames"] == 2

    from src.models.gsav import GsavModel

    direct = GsavModel(reencoded, device="cpu")
    process = direct._stream._process
    try:
        assert direct.source_fps == 24 and direct.total_frames == 2
        assert not list(Path(direct._stream._directory.name).rglob("*.ply"))
        # Random seek and compare with the codec's explicit PLY output.
        for index in (1, 0, 1):
            expected = gsply.plyread(tmp_path / "decoded-again" / f"frame_{index:06d}.ply")
            raw = direct._raw(index)
            for field in ("means", "scales", "quats", "opacities", "sh0", "shN"):
                np.testing.assert_array_equal(raw[field], getattr(expected, field))
        initial = direct.get_frame_at_time(0).means.copy()
        direct.get_frame_at_time(0).means[:] = 123
        np.testing.assert_array_equal(direct.get_frame_at_time(0).means, initial)
        # Both exports accept the direct source with the ordinary edit manager.
        direct_ply = tmp_path / "direct-ply"
        write_edited_sequence(
            direct, [0, 1], edit, direct_ply, fps=24, device="cpu", output_format="PLY"
        )
        exported = gsply.plyread(direct_ply / "frame_000000.ply")
        np.testing.assert_allclose(exported.means, initial + [2, 0, 0], atol=1e-6)
        direct_gsav = tmp_path / "direct.gsav"
        assert (
            write_edited_gsav(direct, [0, 1], edit, direct_gsav, fps=24, device="cpu")["sh_bands"]
            == 3
        )
    finally:
        direct.on_shutdown()
    assert process.poll() is not None

    # Existing conftest installs a viser stub; use a clean real-app interpreter.
    import subprocess
    import sys

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from src.gsplay.core.components.model_component import ModelComponent; "
            "component = ModelComponent(device='cpu'); "
            "model, _, _ = component.load_from_path(sys.argv[1]); "
            "assert model.get_total_frames() == 2; "
            "assert model.time_domain.source_fps == 24; "
            "assert model.get_frame_at_source_time(0).to_gstensor('cpu').shN.shape[1:] == (15, 3); "
            "assert not component._gsav_directories; component.unload()",
            str(reencoded),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        timeout=120,
    )
    assert probe.returncode == 0, probe.stderr.decode(errors="replace")
