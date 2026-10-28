"""A mixture-density model (Trajectron++ here) scored natively.

Any model that outputs, per issue frame, mode weights and a Gaussian per mode and lead fits
``rm.MixturePredictive(issue_frames, weights, means, covs)``. REACT then scores the mixture
itself, not its moment-matched Gaussian. This example wraps Trajectron++ through the adapter
used for the paper (``experiments/trajectron_forecaster.py``), which needs:

* a clone of https://github.com/StanfordASL/Trajectron-plus-plus (``TPP_DIR``), and
* a trained single-agent model directory (``TPP_MODEL``), e.g. the paper's Eindhoven
  checkpoint in the react-reproducibility repository (artifacts/eindhoven/trajectron_run1).

    TPP_DIR=... TPP_MODEL=... python examples/integrations/mixture_model.py

The same wrapper pattern works for any mixture model: replace ``predict_mixture``.
"""

from __future__ import annotations

import os
import sys

from _common import fit_scorer, load, parse_args, report

import reactmetric as rm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "experiments"))


class MixtureModel:
    """Wrap a model that returns issue frames (last observed), weights (n, J), means
    (n, H, J, 2) and covariances (n, H, J, 2, 2)."""

    def __init__(self):
        from trajectron_forecaster import TrajectronForecaster

        self.net = TrajectronForecaster(threads=1, tpp_dir=os.environ["TPP_DIR"],
                                        model_dir=os.environ["TPP_MODEL"])

    def predict(self, traj):
        mix = self.net.predict_mixture(traj)
        if mix is None:
            return None
        return rm.MixturePredictive(mix["issue_frames"], mix["weights"], mix["means"],
                                    mix["covs"], self.net.horizon)


class MomentMatched:
    """The same model collapsed to one Gaussian, for comparison."""

    def __init__(self):
        self.inner = MixtureModel()

    def predict(self, traj):
        p = self.inner.predict(traj)
        return None if p is None else rm.Predictive(p.issue_frames, p.means, p.covs)


def make_models():  # built once per worker process: the network is not picklable
    return {"Trajectron++ (mixture)": MixtureModel(), "Trajectron++ (collapsed)": MomentMatched(),
            "ConstantVelocity": rm.baselines.ConstantVelocity()}


def main():
    args = parse_args(__doc__)
    calib, test = load(args)
    scorer = fit_scorer(calib, args)
    result = scorer.evaluate(make_models, test, n_jobs=args.jobs)
    report(result, ["Trajectron++ (mixture)", "Trajectron++ (collapsed)", "ConstantVelocity"])


if __name__ == "__main__":
    main()
