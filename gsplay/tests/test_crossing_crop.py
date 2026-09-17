"""Crossing decisions, invalidation, CPU/GPU masks and real export parity."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from src.gsplay.crossing_crop import POLICIES, CrossingPlan, active_plan, choose_rows, filter_key


@pytest.mark.parametrize(
    "policy,expected",
    [
        (POLICIES[0], [[1, 0, 1, 0, 0], [1, 0, 0, 1, 0]]),
        (POLICIES[1], [[1, 0, 1, 0, 0], [1, 0, 1, 1, 0]]),
        (POLICIES[2], [[1, 0, 0, 0, 0], [1, 0, 0, 1, 0]]),
    ],
)
def test_inside_outside_crossing_and_absent(policy, expected):
    inside = np.array([[1, 0, 1, 0, 1], [1, 0, 0, 1, 1]], bool)
    presence = np.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], bool)
    np.testing.assert_array_equal(choose_rows(inside, presence, policy), expected)


def test_chunk_boundaries_do_not_share_identity():
    first = choose_rows(np.array([[1], [0]], bool), np.ones((2, 1), bool), POLICIES[1])
    last = choose_rows(np.array([[0]], bool), np.ones((1, 1), bool), POLICIES[1])
    assert first.all() and not last.any()


def test_plan_requires_same_source_boundary_and_policy(tmp_path):
    from gsmod import FilterValues

    config = SimpleNamespace(
        crossing_policy=POLICIES[1], filter_values=FilterValues(sphere_radius=1)
    )
    model = SimpleNamespace(path=tmp_path / "source.gsav", get_total_frames=lambda: 1)
    mask = np.array([1, 0, 1], bool)
    plan = CrossingPlan(
        str(model.path.resolve()),
        filter_key(config.filter_values),
        POLICIES[1],
        3,
        (np.packbits(mask),),
    )
    config.crossing_plan = plan
    assert active_plan(model, config, required=True) is plan
    np.testing.assert_array_equal(plan.mask(0), mask)
    config.crossing_method = "Motion-aware (preview)"
    assert active_plan(model, config) is None
    with pytest.raises(ValueError, match="Analyze & Apply"):
        active_plan(model, config, required=True)
    config.crossing_method = "Chunk-based"
    config.filter_values = replace(config.filter_values, sphere_radius=2)
    assert active_plan(model, config) is None
    with pytest.raises(ValueError, match="Analyze & Apply"):
        active_plan(model, config, required=True)
    config.crossing_policy = POLICIES[0]
    assert active_plan(model, config, required=True) is None


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fixed_mask_preserves_fields_and_capture(device):
    import torch

    from src.domain.entities import GSTensor
    from src.gsplay.processing.fixed_mask import FixedMaskFilter

    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    data = GSTensor(
        means=torch.arange(12, device=device).reshape(4, 3).float(),
        scales=torch.ones((4, 3), device=device),
        quats=torch.ones((4, 4), device=device),
        opacities=torch.ones(4, device=device),
        sh0=torch.arange(12, device=device).reshape(4, 3).float(),
        shN=torch.ones((4, 15, 3), device=device),
    )
    mask = np.array([0, 1, 0, 1], bool)
    selected = FixedMaskFilter(mask).filter_gpu(data, None, None)
    assert selected.means.shape == (2, 3)
    assert torch.equal(selected.sh0, data.sh0[[1, 3]])
    capture = SimpleNamespace(mask=None)
    assert FixedMaskFilter(mask, capture).filter_gpu(data, None, None) is data
    np.testing.assert_array_equal(capture.mask, mask)
    with pytest.raises(ValueError):
        FixedMaskFilter(np.ones(3, bool)).filter_gpu(data, None, None)
    cpu = data.to_gsdata()
    np.testing.assert_array_equal(
        FixedMaskFilter(mask).filter_cpu(cpu, None, None).means, cpu.means[[1, 3]]
    )
    assert len(FixedMaskFilter(np.zeros(4, bool)).filter_cpu(cpu, None, None).means) == 0


@pytest.mark.integration
@pytest.mark.parametrize("method", ["Chunk-based", "Motion-aware (preview)"])
def test_real_analyze_preview_and_exports(tmp_path, method):
    import gsply
    from gsmod import FilterValues

    from src.gsplay.config.settings import GSPlayConfig
    from src.gsplay.core.container import create_edit_manager
    from src.gsplay.crossing_crop import analyze, analyze_motion
    from src.gsplay.gsav_controls import write_edited_sequence
    from src.gsplay.gsav_visibility import create_export_manager
    from src.infrastructure.gsav import encode_gsav
    from src.models.gsav import GsavModel

    source = tmp_path / "ply"
    source.mkdir()
    for i in range(4):
        points = np.zeros((16, 3), np.float32)
        points[4:8, 0] = 3  # always outside
        points[8:12, 0] = 0.2 if i % 2 == 0 else 2  # crossing
        points[12:, 0] = 0.1
        data = gsply.GSData(
            means=points,
            scales=np.full((16, 3), -3, np.float32),
            quats=np.tile(np.array([1, 0, 0, 0], np.float32), (16, 1)),
            opacities=np.ones(16, np.float32),
            sh0=np.zeros((16, 3), np.float32),
            shN=np.full((16, 15, 3), 0.02, np.float32),
        )
        data.opacities[12:] = -np.inf if i == 0 else 1
        gsply.plywrite(source / f"frame_{i:06d}.ply", data)
    original = tmp_path / "original.gsav"
    encode_gsav(source, original, device="cpu")
    model = GsavModel(original, "cpu")
    try:
        config = GSPlayConfig()
        config.edits_active = True
        config.filter_values = FilterValues(sphere_radius=1)
        config.crossing_method = method
        cache_dir = tmp_path / "motion-cache"
        cache_dir.mkdir()
        for policy in POLICIES[1:]:
            config.crossing_policy = policy
            plan = (
                analyze_motion(model, config.filter_values, policy, cache_dir, lambda _: None)
                if method == "Motion-aware (preview)"
                else analyze(model, config.filter_values, policy, "cpu")
            )
            config.crossing_plan = plan
            preview = create_edit_manager(config, "cpu")
            manager, visibility = create_export_manager(config, "cpu", model, range(4), 30, "GSAV")
            destination = tmp_path / (policy.split()[0] + ".gsav")
            write_edited_sequence(
                model,
                range(4),
                manager.apply_edits,
                destination,
                fps=30,
                device="cpu",
                visibility=visibility,
                crossing_plan=plan,
            )
            result = GsavModel(destination, "cpu")
            try:
                for i in range(4):
                    mask = plan.mask(i)
                    before = model._raw(i)
                    after = result._raw(i)
                    for name in ("means", "scales", "quats", "opacities", "sh0", "shN"):
                        np.testing.assert_array_equal(
                            before[name][mask], after[name][after["presence"]]
                        )
                    frame = model.get_frame_at_source_time(i).to_gstensor("cpu")
                    edited = preview.apply_edits(frame, filter_mask=mask)
                    np.testing.assert_array_equal(edited.means.numpy(), before["means"][mask])
            finally:
                result.on_shutdown()
            # PLY export also uses the frozen mask instead of cropping it a second time.
            ply_path = tmp_path / policy.split()[0]
            write_edited_sequence(
                model,
                range(4),
                preview.apply_edits,
                ply_path,
                fps=30,
                device="cpu",
                output_format="PLY",
                crossing_plan=plan,
            )
            files = sorted(ply_path.glob("*.ply"))
            assert len(files) == 4
            for i, file in enumerate(files):
                np.testing.assert_array_equal(
                    gsply.plyread(file).means, model._raw(i)["means"][plan.mask(i)]
                )
    finally:
        model.on_shutdown()
