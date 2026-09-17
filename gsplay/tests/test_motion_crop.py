"""Offline prototype contracts; accepted correspondence is not ground truth."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.gsplay.motion_crop.benchmark import quality_gate, synthetic_report
from src.gsplay.motion_crop.cache import MotionCache, build
from src.gsplay.motion_crop.tracking import Features, MatchSettings, Tracker


def test_synthetic_links_reject_ambiguity_and_handle_permutations():
    results = {item["case"]: item for item in synthetic_report()}
    for name, result in results.items():
        if name != "dense_similar_appearance":
            assert result["wrong_links"] == 0, name
    for name in (
        "static_shuffled",
        "translation_shuffled",
        "crossing_groups",
        "births_and_occlusion",
    ):
        assert results[name]["correct_link_recall"] > 0.95
    assert results["indistinguishable_duplicates"]["accepted_links"] == 0


def test_quality_gate_rejects_false_links_and_requires_real_review():
    assert not quality_gate([{"case": "hard", "wrong_links": 1}])["synthetic_pass"]
    gate = quality_gate([{"case": "hard", "wrong_links": 0}])
    assert gate["synthetic_pass"] and not gate["ready_for_editor"]


def test_empty_frames_break_tracks_and_bad_features_rejected():
    tracker = Tracker(MatchSettings(1))
    frame = Features(np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3)))
    first, _ = tracker.update(frame)
    tracker.update(Features(*(np.empty((0, 3)) for _ in range(3))))
    last, links = tracker.update(frame)
    assert first[0] != last[0] and links[0] == -1
    with pytest.raises(ValueError):
        tracker.update(Features(np.full((1, 3), np.nan), frame.colors, frame.log_scales))
    with pytest.raises(ValueError):
        MatchSettings(float("inf"))


@pytest.fixture
def source_model(tmp_path, monkeypatch):
    from src.models import gsav

    source = tmp_path / "test.gsav"
    source.write_bytes(b"synthetic source fingerprint")
    frames = []
    for t in range(4):
        # First row crosses radius 1, second stays outside, third stays inside.
        positions = np.array([[0.9 + t * 0.05, 0, 0], [3, 0, 0], [0, 0, 0]], np.float32)
        frames.append(
            SimpleNamespace(
                means=positions,
                scales=np.full((3, 3), 0.01, np.float32),
                opacities=np.full(3, 0.5, np.float32),
                sh0=np.eye(3, dtype=np.float32),
                is_sh0_rgb=True,
            )
        )

    class Model:
        def __init__(self, *args):
            self.closed = False

        def get_total_frames(self):
            return 4

        def get_frame_time(self, i):
            return i

        def get_gaussians_at_normalized_time(self, i):
            return frames[i]

        def _raw(self, i):
            return {"presence": np.ones(3, bool)}

        def on_shutdown(self):
            self.closed = True

    monkeypatch.setattr(gsav, "GsavModel", Model)
    return source


def test_cache_reuses_geometry_for_boundary_and_policy_changes(source_model, tmp_path):
    from gsmod import FilterValues

    directory = tmp_path / "cache"
    build(source_model, directory, radius_fraction=0.1, progress=lambda _: None)
    cache = MotionCache(directory, source_model)
    try:
        keep, _ = cache.evaluate(FilterValues(sphere_radius=1), "keep")
        remove, _ = cache.evaluate(FilterValues(sphere_radius=1), "remove")
        moved, _ = cache.evaluate(FilterValues(sphere_radius=0.5), "keep")
        for i in range(4):
            np.testing.assert_array_equal(np.unpackbits(keep[i], count=3), [1, 0, 1])
            np.testing.assert_array_equal(np.unpackbits(remove[i], count=3), [0, 0, 1])
            np.testing.assert_array_equal(np.unpackbits(moved[i], count=3), [0, 0, 1])
    finally:
        cache.close()
    with pytest.raises(FileExistsError):
        build(source_model, directory)
    source_model.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="different source"):
        MotionCache(directory, source_model)


def test_incomplete_cache_cannot_be_reused(source_model, tmp_path):
    directory = tmp_path / "incomplete"
    with pytest.raises(ValueError, match="budget"):
        build(source_model, directory, max_bytes=1)
    assert not (directory / "complete.json").exists()
    with pytest.raises(FileNotFoundError):
        MotionCache(directory, source_model)


def test_invalid_track_ids_fail_closed(source_model, tmp_path):
    from gsmod import FilterValues

    directory = tmp_path / "cache"
    build(source_model, directory, progress=lambda _: None)
    tracks = np.load(directory / "tracks.npy", mmap_mode="r+")
    tracks[0, 0] = -1
    tracks.flush()
    tracks._mmap.close()
    cache = MotionCache(directory, source_model)
    try:
        with pytest.raises(ValueError, match="track IDs"):
            cache.evaluate(FilterValues(sphere_radius=1), "keep")
    finally:
        cache.close()
