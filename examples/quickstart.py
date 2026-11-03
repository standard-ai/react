"""REACT quickstart: kinematic baselines on Eindhoven pedestrian tracks.

    python examples/quickstart.py                          # bundled 40-track sample (~2 min)
    python examples/quickstart.py --csv eindhoven.csv      # a larger export (see scripts/)

Fits REACT on 30% of the tracks (climatology, noise floor, truth model, overlap tables),
scores constant velocity, CTRV and IMM on the rest, and prints each model's REACT profile:
anticipation, recovery, censoring and relapse. The bundled sample is far too small for
conclusions; it shows the workflow. Bring your own model by adding it to `models`: anything
with predict(traj) -> rm.Predictive (or rm.MixturePredictive), or a dict
{track id: Predictive} (docs/forecasters.md).
"""

from __future__ import annotations

import argparse
import os

import reactmetric as rm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None, help="Eindhoven-format CSV (default: bundled sample)")
    ap.add_argument("--leads", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--info", type=float, nargs="+", default=[3.0, 4.2, 6.0],
                    help="assumed information levels (nats); the headline is chosen by the "
                         "paper's rule")
    ap.add_argument("--distortion", type=float, default=0.1, help="target distance d (m)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="react_quickstart", help="output directory")
    args = ap.parse_args()

    data = (rm.datasets.load_eindhoven(path=args.csv) if args.csv
            else rm.datasets.load_eindhoven_sample())
    calib, test = data.split(fraction=0.3, seed=0)
    print(f"{len(calib)} calibration / {len(test)} test tracks")

    scorer = rm.ReactScorer.fit(calib, info=args.info, leads=args.leads,
                                distortions=[args.distortion], n_jobs=args.jobs)
    print(f"noise floor {scorer.noise_floor['sigma_cm']:.2f} cm")
    models = {
        "ConstantVelocity": rm.baselines.ConstantVelocity.fit(calib),
        "CTRV": rm.baselines.CTRV(),
        "IMM": rm.baselines.IMM(),
    }
    result = scorer.evaluate(models, test, n_jobs=args.jobs)

    os.makedirs(args.out, exist_ok=True)
    result.to_json(os.path.join(args.out, "result.json"))   # ReactResult.load() reads it back
    info, reason = result.headline_info()
    print(f"\ninformation level {info:g} nats: {reason}")
    for lead in args.leads:
        for d in (None, args.distortion):
            print(f"\n{rm.viz.react_label(lead, d, info)}  (frames; see rm.results.GLOSSARY)")
            print(result.table(lead=lead, distortion=d).drop(columns=["lead", "info",
                                                                    "distortion"])
                  .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    lead = args.leads[-1]
    cmp = result.compare("IMM", "ConstantVelocity", "rmst", lead=lead, distortion=args.distortion)
    print(f"\nIMM - ConstantVelocity, restricted mean REACT at lead {lead}: "
          f"{cmp['difference']:+.1f} frames (95% CI {cmp['low']:+.1f} to {cmp['high']:+.1f}, "
          f"{cmp['tracks']} tracks)")
    try:
        rm.viz.km_curves(result, lead=lead, distortion=args.distortion,
                         save=os.path.join(args.out, "km.png"))
    except ImportError:
        pass
    print(f"\noutputs in {args.out}/")


if __name__ == "__main__":
    main()
