"""Run offline tracking benchmarks. Never changes the editor or exports a scene."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.gsplay.motion_crop.tracking import Features, MatchSettings, Tracker


def synthetic_cases():
    rng = np.random.default_rng(431)
    base = rng.uniform(-2, 2, (128, 3)).astype(np.float32)
    colors = rng.uniform(0, 1, (128, 3)).astype(np.float32)
    scales = np.sort(rng.uniform(-4, -2, (128, 3)), axis=1).astype(np.float32)
    for name in (
        "static_shuffled",
        "translation_shuffled",
        "crossing_groups",
        "births_and_occlusion",
        "fast_jump",
        "indistinguishable_duplicates",
        "dense_similar_appearance",
    ):
        frames = []
        for t in range(24):
            positions = base.copy()
            appearance, sizes = colors.copy(), scales.copy()
            if name != "static_shuffled":
                positions[:, 0] += t * 0.025
            if name == "crossing_groups":
                positions[64:, 0] -= t * 0.05
            if name == "fast_jump" and t >= 12:
                positions[:, 0] += 2
            if name == "indistinguishable_duplicates":
                positions[64:] = positions[:64]
                appearance[64:] = appearance[:64]
                sizes[64:] = sizes[:64]
            if name == "dense_similar_appearance":
                positions = base * 0.03
                positions[:, 0] += t * 0.025
                appearance[:] = 0.5
                sizes[:] = -3
            ids = np.arange(128)
            if name == "births_and_occlusion" and 8 <= t <= 13:
                ids = ids[32:]
            order = rng.permutation(ids)
            frames.append((Features(positions[order], appearance[order], sizes[order]), order))
        yield name, frames


def synthetic_report():
    report = []
    for name, frames in synthetic_cases():
        tracker = Tracker(MatchSettings(radius=0.2))
        accepted, wrong, possible = 0, 0, 0
        previous_truth = None
        started = time.perf_counter()
        for frame, truth in frames:
            _, links = tracker.update(frame)
            linked = links >= 0
            if previous_truth is not None:
                possible += len(np.intersect1d(truth, previous_truth))
                accepted += int(linked.sum())
                wrong += int((truth[linked] != previous_truth[links[linked]]).sum())
            previous_truth = truth
        report.append(
            {
                "case": name,
                "accepted_links": accepted,
                "wrong_links": wrong,
                "precision": (accepted - wrong) / accepted if accepted else None,
                "correct_link_recall": (accepted - wrong) / max(1, possible),
                "seconds": time.perf_counter() - started,
            }
        )
    return report


def quality_gate(results):
    """Conservative research gate: false associations prohibit UI promotion.

    These thresholds are engineering acceptance criteria, not calibrated
    guarantees. Real-scene visual review is separately required even on a pass.
    """
    failures = [r["case"] for r in results if r["wrong_links"] > 0]
    easy = {"static_shuffled", "translation_shuffled", "crossing_groups", "births_and_occlusion"}
    failures += [
        r["case"] for r in results if r["case"] in easy and r["correct_link_recall"] < 0.95
    ]
    return {
        "synthetic_pass": not failures,
        "failed_cases": sorted(set(failures)),
        "real_scene_visual_review": "not completed",
        "ready_for_editor": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--center", nargs=3, type=float, default=(0, 0, 0))
    parser.add_argument("--radius", type=float, default=1.0)
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError("Choose a new report filename")
    report = {"synthetic": synthetic_report(), "editor_enabled": False}
    report["quality_gate"] = quality_gate(report["synthetic"])
    if args.source:
        from gsmod import FilterValues

        from src.gsplay.motion_crop.cache import MotionCache, build

        if args.cache is None:
            raise ValueError("--source requires --cache")
        if not args.cache.exists():
            build(args.source, args.cache, progress=lambda s: print(s, flush=True))
        cache = MotionCache(args.cache, args.source)
        try:
            report["cache_build_seconds"] = cache.meta["seconds"]
            report["cache_bytes"] = sum(
                p.stat().st_size for p in args.cache.iterdir() if p.is_file()
            )
            report["frames"] = cache.meta["frames"]
            report["crop_evaluations"] = []
            for radius, policy in (
                (args.radius, "keep"),
                (args.radius, "remove"),
                (args.radius * 0.95, "keep"),
            ):
                masks, stats = cache.evaluate(
                    FilterValues(sphere_center=tuple(args.center), sphere_radius=radius), policy
                )
                stats.update(
                    radius=radius, policy=policy, visible_samples=int(np.unpackbits(masks).sum())
                )
                report["crop_evaluations"].append(stats)
            report["mean_accepted_link_fraction"] = float(
                np.mean(
                    [step["linked"] / max(1, step["active"]) for step in cache.meta["steps"][1:]]
                )
            )
        finally:
            cache.close()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
