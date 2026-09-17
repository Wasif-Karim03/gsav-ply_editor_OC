"""Color-only exports preserve decoded geometry and compressed low positions."""

import io
import wave

import numpy as np
import pytest
import torch

from gscodec.common.binary_format import read_header, write_header
from gscodec.common.types import HAS_MASK_FLAG
from gscodec.decoder import SequenceDecoder
from gscodec.decoder.providers import GSAVFileProvider
from gscodec.encoder import ChunkConfig, SequenceEncoder
from gscodec.encoder.source_geometry import SourceGeometry


@pytest.mark.parametrize("degree", [0, 3])
@pytest.mark.parametrize("version", [1, 3])
@pytest.mark.parametrize("brightness", [0.7, 3.78])
def test_color_export_keeps_exact_geometry(tmp_path, mock_gstensor, degree, version, brightness):
    frames = [mock_gstensor[:16].clone() for _ in range(4)]
    for i, frame in enumerate(frames):
        frame.means += i * 0.01
        frame.scales[:] = -12  # Below the editor export's old normalization floor.
        frame.opacities[:] = 10
        frame.masks[i] = False
        if degree:
            frame.shN = torch.full((16, 15, 3), 0.03)
    cfg = ChunkConfig(
        identity_mode="preserve", size=2, sh_bands=degree, lo_snap_k=19, keyframe_snap=False
    )
    # Source encoder's standard identity mode permits snapped position detail.
    cfg.identity_mode = "stable"
    source_bytes = SequenceEncoder(device="cpu", chunk_config=cfg).encode_frames(frames)
    if version == 1:
        stream = io.BytesIO(source_bytes)
        header = read_header(stream)
        header["version"] = 1
        header["flags"] &= ~HAS_MASK_FLAG
        stream.seek(0)
        write_header(stream, header)
        source_bytes = stream.getvalue()
    source = tmp_path / "source.gsav"
    source.write_bytes(source_bytes)
    before = SequenceDecoder(GSAVFileProvider(source_bytes)).decode_all()
    edited = [f.clone() for f in frames]
    for f in edited:
        f.sh0 = (f.sh0 + 0.5 / 0.28209479177387814) * brightness - 0.5 / 0.28209479177387814
        if degree:
            f.shN *= brightness
    encoder = SequenceEncoder(
        device="cpu",
        chunk_config=ChunkConfig(
            identity_mode="preserve", size=2, sh_bands=degree, lo_snap_k=1, keyframe_snap=False
        ),
    )
    result = encoder.encode_frames(edited, geometry_source=source)
    target = tmp_path / "target.gsav"
    target.write_bytes(result)
    assert SourceGeometry(source).means_lo == SourceGeometry(target).means_lo
    after = SequenceDecoder(GSAVFileProvider(result)).decode_all()
    for a, b in zip(before, after, strict=True):
        for field in ("means", "scales", "quats", "opacities", "masks"):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    for expected, actual in zip(edited, after, strict=True):
        mask = actual.masks
        np.testing.assert_allclose(expected.sh0.numpy()[mask], actual.sh0[mask], atol=0.1)
        if degree:
            np.testing.assert_array_equal(expected.shN.numpy()[mask], actual.shN[mask])
    with pytest.raises(ValueError, match="unchanged timeline"):
        encoder.encode_frames(edited[:2], geometry_source=source)


def test_reuse_rejects_unverified_rows(tmp_path, mock_gstensor):
    with pytest.raises(ValueError, match="verified preserved rows"):
        SequenceEncoder(device="cpu").encode_frames(
            [mock_gstensor], geometry_source=tmp_path / "unused.gsav"
        )


def test_audio_is_copied_without_reencoding(tmp_path, mock_gstensor, monkeypatch):
    from gscodec.encoder import sequence_encoder

    wav = tmp_path / "tone.wav"
    with wave.open(str(wav), "w") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(48000)
        stream.writeframes(np.zeros(4800, dtype=np.int16).tobytes())
    frames = [mock_gstensor[:16].clone() for _ in range(3)]
    encoder = SequenceEncoder(
        device="cpu",
        chunk_config=ChunkConfig(
            identity_mode="preserve", size=2, sh_bands=0, lo_snap_k=1, keyframe_snap=False
        ),
    )
    source = tmp_path / "audio.gsav"
    source.write_bytes(encoder.encode_frames(frames, audio_path=wav))

    def forbidden(*args, **kwargs):
        raise AssertionError("Color-only export must not transcode source audio")

    monkeypatch.setattr(sequence_encoder, "transcode_to_opus", forbidden)
    result = encoder.encode_frames(frames, geometry_source=source, audio_path=wav)
    assert GSAVFileProvider(result).get_audio() == GSAVFileProvider(source).get_audio()
    from gscodec.encoder.crop_source import crop_source

    cropped = crop_source(source, [np.arange(16) < 4 for _ in frames])
    assert GSAVFileProvider(cropped).get_audio() == GSAVFileProvider(source).get_audio()


def test_visibility_retains_geometry_and_never_requires_matching(tmp_path, mock_gstensor):
    frames = [mock_gstensor[:16].clone() for _ in range(4)]
    frames[1].masks[3] = False
    encoder = SequenceEncoder(
        device="cpu",
        chunk_config=ChunkConfig(
            identity_mode="preserve", size=2, sh_bands=0, lo_snap_k=1, keyframe_snap=False
        ),
    )
    source = tmp_path / "source.gsav"
    source.write_bytes(encoder.encode_frames(frames))
    masks = [np.arange(16) % 2 == i % 2 for i in range(4)]
    masks[1][3] = False
    masks[3][:] = False  # Entirely hidden frames remain valid.
    data = encoder.encode_frames(frames, geometry_source=source, visibility=masks)
    original = SequenceDecoder.from_file(source).decode_all()
    edited = SequenceDecoder(GSAVFileProvider(data)).decode_all()
    for a, b, mask in zip(original, edited, masks, strict=True):
        np.testing.assert_array_equal(b.masks, mask)
        for field in ("means", "scales", "quats"):
            np.testing.assert_array_equal(getattr(a, field)[mask], getattr(b, field)[mask])
    with pytest.raises(ValueError, match="visibility dimensions"):
        encoder.encode_frames(frames, geometry_source=source, visibility=masks[:2])
    with pytest.raises(ValueError, match="Visibility overrides"):
        encoder.encode_frames(frames, visibility=masks)
    invalid = [mask.copy() for mask in masks]
    invalid[1][3] = True
    with pytest.raises(ValueError, match="cannot revive"):
        encoder.encode_frames(frames, geometry_source=source, visibility=invalid)
