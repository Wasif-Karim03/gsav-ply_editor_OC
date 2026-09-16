"""On-demand GSAV source with a bounded raw-frame cache and existing edit types."""

import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from src.domain.data import GaussianData
from src.domain.entities import GSData
from src.domain.interfaces import SourceMetadata
from src.domain.time import TimeDomain
from src.infrastructure.gsav_stream import GsavStream
from src.infrastructure.processing.gaussian_constants import GaussianConstants as GC


class GsavModel:
    def __init__(self, path: Path, device: str = "cpu"):
        self.path = Path(path)
        self.device = device
        self.processing_mode = "all_cpu" if device == "cpu" else "all_gpu"
        self._stream = GsavStream(self.path)
        self.gsav_metadata = self._stream.metadata
        self.total_frames = self.gsav_metadata["frames"]
        self.source_fps = float(self.gsav_metadata["fps"])
        self.playback_fps = self.source_fps
        self.lock_playback_fps = False
        self.autoplay = False
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self._lock = threading.RLock()

    @classmethod
    def metadata(cls):
        return SourceMetadata(
            name="GSAV", description="Direct v3 Gaussian decoding", file_extensions=[".gsav"]
        )

    @classmethod
    def can_load(cls, path):
        return str(path).lower().endswith(".gsav")

    @property
    def time_domain(self):
        return TimeDomain.discrete(self.total_frames, source_fps=self.source_fps)

    def get_total_frames(self):
        return self.total_frames

    def get_frame_time(self, index):
        return index / max(1, self.total_frames - 1)

    def _raw(self, index):
        with self._lock:
            if index not in self._cache:
                arrays = self._stream.frame(index)
                self._cache[index] = arrays
                self._cache_bytes += sum(a.nbytes for a in arrays.values())
                while len(self._cache) > 1 and (
                    len(self._cache) > 4 or self._cache_bytes > 128 * 1024**2
                ):
                    _, old = self._cache.popitem(last=False)
                    self._cache_bytes -= sum(a.nbytes for a in old.values())
            self._cache.move_to_end(index)
            # Editing and activation must never mutate cached decoder output.
            return {name: array.copy() for name, array in self._cache[index].items()}

    def get_gaussians_at_normalized_time(self, normalized_time):
        index = max(0, min(round(normalized_time * (self.total_frames - 1)), self.total_frames - 1))
        arrays = self._raw(index)
        self._last_loaded_filename = self.path.name
        self._last_loaded_frame_index = index
        arrays.pop("presence", None)
        raw = GSData(**arrays)
        data = raw.denormalize(inplace=True)
        data.scales = np.clip(data.scales, GC.Numerical.MIN_SCALE, GC.Numerical.MAX_SCALE)
        data.opacities = np.clip(data.opacities, 0.0, 1.0)
        data.quats /= np.maximum(np.linalg.norm(data.quats, axis=-1, keepdims=True), 1e-8)
        if data.shN is None or not data.shN.size:
            data = data.to_rgb(inplace=True)
            data.sh0 = np.clip(data.sh0, 0.0, 1.0)
        if self.processing_mode in ("all_gpu", "gpu"):
            tensor = GaussianData.from_gsdata(data).to_gstensor(self.device)
            tensor._base = None
            return tensor
        return data

    def get_frame_at_time(self, normalized_time):
        data = self.get_gaussians_at_normalized_time(normalized_time)
        if hasattr(data, "device"):
            return GaussianData.from_gstensor(data, str(self.path))
        return GaussianData.from_gsdata(data, str(self.path))

    def get_frame_at_source_time(self, source_time):
        return self.get_frame_at_time(self.time_domain.to_normalized(source_time))

    def get_recommended_max_scale(self):
        return None

    def get_last_profile(self):
        return None

    def get_camera_data(self):
        return None

    def get_points_for_initialization(self):
        return self._raw(0)["means"]

    def invalidate_frame_cache(self):
        # Cache contains unedited raw arrays, independent of editor settings.
        pass

    def on_shutdown(self, timeout=5.0):
        with self._lock:
            self._stream.close()
            self._cache.clear()
            self._cache_bytes = 0
