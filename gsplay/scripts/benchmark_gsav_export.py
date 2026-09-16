"""Run from gsplay: python -m scripts.benchmark_gsav_export INPUT OUTPUT.gsav.

Add --standard for the ordinary matching path. Uses separate output files.
"""

import argparse
import json
import time
from pathlib import Path

from gsmod import ColorValues

from src.gsplay.config.settings import GSPlayConfig
from src.gsplay.core.container import create_edit_manager
from src.gsplay.gsav_controls import export_times, write_edited_sequence
from src.models.gsav import GsavModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--brightness", type=float, default=0.7)
    parser.add_argument("--standard", action="store_true")
    args = parser.parse_args()
    report = args.output.with_suffix(".benchmark.json")
    if args.output.exists() or report.exists():
        parser.error("Choose a new output filename")
    started = time.monotonic()
    events = []

    def status(message):
        event = {"seconds": round(time.monotonic() - started, 3), "stage": message}
        events.append(event)
        print(json.dumps(event), flush=True)

    model = GsavModel(args.source, device=args.device)
    config = GSPlayConfig()
    config.edits_active = True
    config.color_values = ColorValues(brightness=args.brightness)
    manager = create_edit_manager(config, args.device)
    try:
        result = write_edited_sequence(
            model,
            export_times(model, "Original Frames", 0),
            manager.apply_edits,
            args.output,
            fps=int(model.source_fps),
            device=args.device,
            audio=model.gsav_metadata.get("audio"),
            status=status,
            fast_export=not args.standard,
        )
        report.write_text(
            json.dumps(
                {
                    "seconds": time.monotonic() - started,
                    "bytes": args.output.stat().st_size,
                    "brightness": args.brightness,
                    "device": args.device,
                    "standard_requested": args.standard,
                    "result": result,
                    "events": events,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Benchmark report: {report}")
    finally:
        model.on_shutdown()


if __name__ == "__main__":
    main()
