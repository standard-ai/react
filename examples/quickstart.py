"""REACT quickstart: the ADE/FDE-vs-REACT motivating story.

Run:  python examples/quickstart.py

Shows that two forecasters can have similar average accuracy (ADE/FDE) yet very
different *adaptation speed* (REACT) after regime changes -- the gap REACT is
designed to surface. Defaults to the bundled Eindhoven-format reference sample,
and can also run on the full Eindhoven Zenodo dataset with --dataset eindhoven
--download.
"""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Optional

import reactmetric as rm


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run an end-to-end REACT benchmark and write example plots."
    )
    parser.add_argument(
        "--dataset",
        choices=("eindhoven-sample", "eindhoven", "synthetic", "ethucy", "tracks"),
        default="eindhoven-sample",
        help="Dataset to load. Full real datasets require --path, --root, or --download.",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="Direct path to a local dataset table/text file (Eindhoven/Tracks/ETH).",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Dataset directory. ETH/UCY expects <scene>.txt; Tracks expects a table file.",
    )
    parser.add_argument("--scene", default="eth", help="ETH/UCY scene name.")
    parser.add_argument(
        "--dataset-split",
        default=None,
        help=(
            "Optional named partition file under --root/--path directory "
            "(e.g. eval -> eval.parquet). Not required for single-table loads."
        ),
    )
    parser.add_argument(
        "--prepare-output-dir",
        default=None,
        help=(
            "If set with --path, package the source table into on-disk "
            "dataset_split files under this directory (train/eval by default), "
            "then load the eval partition."
        ),
    )
    parser.add_argument(
        "--prepare-eval-fraction",
        type=float,
        default=0.25,
        help="Eval fraction used when --prepare-output-dir is set (remainder=train).",
    )
    parser.add_argument(
        "--mapping",
        default=None,
        help=(
            "Optional mapping name or YAML path for --path loads. Defaults come from "
            "rm.datasets.mappings.<dataset>."
        ),
    )
    parser.add_argument(
        "--days",
        default="01_10",
        help="Eindhoven 10-day shard to download/load, e.g. 01_10 or 51_60.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download a full Eindhoven shard from Zenodo when not cached.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Dataset cache root for downloaded Eindhoven shards.",
    )
    parser.add_argument("--dt", type=float, default=None, help="Override dataset dt.")
    parser.add_argument("--max-tracks", type=int, default=None, help="Optional track cap.")
    parser.add_argument("--synthetic-n", type=int, default=300, help="Synthetic track count.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lead", type=int, default=10)
    parser.add_argument(
        "--event-preset",
        choices=("auto", "high", "medium", "low", "very-low"),
        default="auto",
        help="Event detector sensitivity. Auto uses low for Eindhoven and medium otherwise.",
    )
    parser.add_argument(
        "--target-events-per-track",
        type=float,
        default=None,
        help="Tune event detector penalty to this average event count per calibration track.",
    )
    parser.add_argument(
        "--target-events-per-minute",
        type=float,
        default=None,
        help="Tune event detector penalty to this average event rate per trajectory minute.",
    )
    parser.add_argument("--output-dir", default=".", help="Directory for JSON and plots.")
    return parser.parse_args()


def _limit_tracks(data, max_tracks: Optional[int]):
    if max_tracks is None or len(data) <= max_tracks:
        return data
    return rm.TrajectorySet({tid: data[tid] for tid in data.ids[:max_tracks]})


def _resolve_mapping_arg(args):
    if args.mapping is None:
        # Infer from dataset choice when preparing/loading a local table.
        if args.dataset in {"tracks", "eindhoven", "eindhoven-sample", "ethucy"}:
            return "tracks" if args.dataset == "tracks" else (
                "eindhoven" if args.dataset.startswith("eindhoven") else "ethucy"
            )
        return None
    if os.path.exists(args.mapping) and (
        args.mapping.endswith(".yml") or args.mapping.endswith(".yaml")
    ):
        return rm.datasets.DatasetMapping.from_yaml(args.mapping)
    return args.mapping


def _load_dataset(args):
    # Optional packaging step: single source table -> on-disk dataset_split files.
    if args.prepare_output_dir is not None:
        if args.path is None:
            raise SystemExit("--prepare-output-dir requires --path to a source table.")
        mapping = _resolve_mapping_arg(args)
        if mapping is None:
            raise SystemExit(
                "--prepare-output-dir needs --mapping or a dataset with a known mapping."
            )
        result = rm.datasets.prepare(
            args.path,
            mapping,
            output_dir=args.prepare_output_dir,
            dataset_split=args.prepare_eval_fraction,
            max_tracks=args.max_tracks,
            seed=args.seed,
            fmt="parquet",
        )
        print(
            "Prepared dataset_split files: "
            + ", ".join(f"{k}={v}" for k, v in result.n_tracks.items())
        )
        # Prefer an 'eval' partition when present; otherwise load the first written file.
        split_name = "eval" if "eval" in result.paths else next(iter(result.paths))
        return rm.datasets.load(
            result.paths[split_name],
            mapping,
            dt=args.dt,
        )

    # Custom local table: unified mapping-driven path.
    if args.mapping is not None and args.path is not None:
        mapping = _resolve_mapping_arg(args)
        return rm.datasets.load(
            args.path,
            mapping,
            max_tracks=args.max_tracks,
            dt=args.dt,
        )

    if args.dataset == "eindhoven-sample":
        return rm.datasets.load_dataset("eindhoven-sample", dt=args.dt)
    if args.dataset == "eindhoven":
        return rm.datasets.load_dataset(
            "eindhoven",
            path=args.path,
            days=args.days,
            cache_dir=args.cache_dir,
            download=args.download,
            max_tracks=args.max_tracks,
            dt=args.dt,
        )
    if args.dataset == "synthetic":
        return rm.datasets.load_dataset(
            "synthetic",
            n=args.synthetic_n,
            seed=args.seed,
            max_tracks=args.max_tracks,
            dt=args.dt,
        )
    if args.dataset == "ethucy":
        source = args.path or args.scene
        if args.path is None and args.root is None:
            raise SystemExit(
                "ETH/UCY needs either --path /path/to/scene.txt or "
                "--root /path/to/ethucy --scene eth|hotel|univ|zara1|zara2"
            )
        return rm.datasets.load_dataset(
            "ethucy",
            path=source if args.path else None,
            scene=args.scene,
            root=args.root,
            dt=args.dt,
            max_tracks=args.max_tracks,
        )
    if args.dataset == "tracks":
        if args.path is None and args.root is None:
            raise SystemExit(
                "Tracks needs a local table: --path /path/to/tracks.csv|.parquet "
                "or --root /path/to/dir. Single-file loads are the default; "
                "--dataset-split only selects an optional named partition file. "
                "Use --prepare-output-dir to create partitions from a large source."
            )
        return rm.datasets.load_dataset(
            "tracks",
            path=args.path,
            root=args.root,
            dataset_split=args.dataset_split,
            max_tracks=args.max_tracks,
            dt=args.dt,
        )
    raise AssertionError(args.dataset)


def _event_preset(args) -> str:
    if args.event_preset != "auto":
        return args.event_preset
    if args.dataset in {"eindhoven", "eindhoven-sample"}:
        return "low"
    return "medium"


def _build_detector(args, calib):
    preset = _event_preset(args)
    detector = rm.events.KinematicChangepoints.from_preset(preset)
    if args.target_events_per_track is not None and args.target_events_per_minute is not None:
        raise SystemExit(
            "Use only one of --target-events-per-track or --target-events-per-minute."
        )
    if args.target_events_per_track is not None:
        penalty = detector.calibrate_events_per_track(calib, args.target_events_per_track)
        print(
            "Calibrated event detector "
            f"(preset={preset}, target={args.target_events_per_track:g}/track, "
            f"penalty={penalty:.2f})"
        )
    elif args.target_events_per_minute is not None:
        penalty = detector.calibrate_events_per_minute(calib, args.target_events_per_minute)
        print(
            "Calibrated event detector "
            f"(preset={preset}, target={args.target_events_per_minute:g}/minute, "
            f"penalty={penalty:.2f})"
        )
    else:
        print(f"Using event detector preset={preset}")
    return detector


def main() -> None:
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load trajectories. Synthetic has known regime changes, but the
    # quickstart intentionally uses the same detector path for all datasets.
    data = _limit_tracks(_load_dataset(args), args.max_tracks)
    calib, test = data.split(fraction=0.25, seed=0)

    # 2. Detect events with the built-in kinematic changepoint detector.
    detector = _build_detector(args, calib)
    detected_events = detector.detect_set(test)
    n_events = sum(len(v) for v in detected_events.values())
    print(f"Detected {n_events} events across {len(test)} evaluation tracks.")

    # 3. Fit the reference climatology (speed/heading-conditioned).
    print("Fitting reference climatology ...")
    leads = tuple(sorted({1, 5, 10, 15, args.lead}))
    ref = rm.reference.Climatology.fit(calib, leads=leads, mode="kinematic")

    # 4. Benchmark adaptive baselines + pretrained pedestrian GRUs + oracle.
    print("Running benchmark ...")
    models = {
        "ConstantVelocity": rm.baselines.ConstantVelocity(),
        "CTRV": rm.baselines.CTRV(),
        "IMM": rm.baselines.IMM(),
    }
    gru_baselines = []
    try:
        models["GRUPosition"] = rm.baselines.GRUPosition()
        models["GRUOrientation"] = rm.baselines.GRUOrientation()
        gru_baselines = ["GRUPosition", "GRUOrientation"]
        print(
            "Loaded pretrained pedestrian GRUs "
            "(position + orientation checkpoints trained on a large retail "
            "pedestrian corpus)."
        )
    except ImportError:
        print("(install reactmetric[torch] to include pretrained GRU baselines)")
    except FileNotFoundError as e:
        print(f"(GRU checkpoints unavailable: {e})")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # calibration warnings are demonstrated separately
        report = rm.benchmark(
            forecasts={"IMM": models["IMM"]},
            data=test,
            calibration=calib,
            baselines=["ConstantVelocity", "CTRV", *gru_baselines],
            oracle="C",
            events=detected_events,
            lead=args.lead,
            epsilon=0.0,
        )

    print(f"\n=== {rm.constants.react_at_lead(args.lead)} benchmark (lead={args.lead}) ===")
    print(report.table())

    # 5. The motivating contrast: accuracy vs adaptation.
    print("\n=== Accuracy (ADE/FDE) -- the metric REACT complements ===")
    acc = rm.forecast_accuracy(
        models,
        test,
        leads=leads,
    )
    for name, a in acc.items():
        print(f"  {name:18s}  ADE={a['ade_m']:.3f} m  FDE={a['fde_m']:.3f} m")

    print(f"\n=== Adaptation ({rm.constants.react_at_lead(args.lead)}) -- what ADE/FDE hides ===")
    for name, res in report.results.items():
        print(
            f"  {name:18s}  KM-median {res.react_label}={res.km_median():5.1f} frames  "
            f"anticipation={100 * res.anticipation_rate:4.1f}%  "
            f"excess vs oracle={res.mean_excess_react:5.1f} frames"
        )

    # 6. Pairwise significance and serialization.
    print("\n=== Wilcoxon vs ConstantVelocity ===")
    print(report.wilcoxon(baseline="ConstantVelocity"))
    report_path = os.path.join(args.output_dir, "react_results.json")
    report.to_json(report_path)
    print(f"\nWrote {report_path}")

    try:
        km_path = os.path.join(args.output_dir, "km_curves.png")
        report.plot_km(km_path)
        print(f"Wrote {km_path}")
        overview_path = os.path.join(args.output_dir, "dataset_trajectory_overview.png")
        rm.viz.dataset_trajectory_overview(
            test,
            detected_events,
            max_tracks=60,
            save=overview_path,
        )
        print(f"Wrote {overview_path}")
        task_path = os.path.join(args.output_dir, "task_react_km_curves.png")
        rm.viz.task_react_km_curves(
            models,
            test,
            ref,
            detected_events,
            lead=args.lead,
            distortion_m=0.5,
            save=task_path,
        )
        print(f"Wrote {task_path}")
        performance_path = os.path.join(args.output_dir, "forecast_performance_summary.png")
        rm.viz.forecast_performance_summary(
            acc,
            leads=leads,
            save=performance_path,
        )
        print(f"Wrote {performance_path}")
    except ImportError:
        print("(install reactmetric[viz] for plots)")


if __name__ == "__main__":
    main()
