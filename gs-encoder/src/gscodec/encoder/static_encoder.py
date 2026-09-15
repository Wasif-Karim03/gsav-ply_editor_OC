"""Static assets encoded with the same Gaussian codec as dynamic frames."""

from pathlib import Path

import numpy as np
import torch
from gsply import GSData, GSTensor

from gscodec.common.static_track import static_track_from_gsav, validate_static_track
from gscodec.common.types import STATIC_ENCODING_NATIVE


def encode_static_frame(frame: GSTensor, device: str = "cpu", sh_bands: int = -1) -> bytes:
    from gscodec.encoder.config import ChunkConfig
    from gscodec.encoder.sequence_encoder import SequenceEncoder

    encoder = SequenceEncoder(
        device=device, chunk_config=ChunkConfig(size=1, sh_bands=sh_bands, lo_snap_k=1)
    )
    return static_track_from_gsav(encoder.encode_frames([frame], prune=False))


def tensor_from_gsdata(data: GSData) -> GSTensor:
    """Preserve source masks and SH in the codec's log/logit representation."""
    return GSTensor(
        means=torch.from_numpy(data.means).float(),
        scales=torch.from_numpy(data.scales).float(),
        quats=torch.from_numpy(data.quats).float(),
        opacities=torch.from_numpy(data.opacities.reshape(-1, 1)).float(),
        sh0=torch.from_numpy(data.sh0).float(),
        shN=torch.from_numpy(data.shN).float() if data.shN is not None and data.shN.size else None,
        masks=torch.from_numpy(
            np.ones(len(data.means), dtype=bool) if data.masks is None else data.masks.reshape(-1)
        ).bool(),
    )


def encode_static_file(path: str | Path, device: str = "cpu") -> tuple[bytes, int]:
    import gsply

    path = Path(path)
    if path.suffix.lower() == ".gsst":
        data = path.read_bytes()
        validate_static_track(data)
        return data, STATIC_ENCODING_NATIVE
    if path.suffix.lower() not in (".ply", ".spz"):
        raise ValueError("Static asset must be PLY, SPZ, or GSST")
    data = gsply.read_spz(path) if path.suffix.lower() == ".spz" else gsply.plyread(path)
    return encode_static_frame(tensor_from_gsdata(data), device=device), STATIC_ENCODING_NATIVE
