"""The paper's rule for the headline information level (section 3.7)."""

import glob
import json
import os

import pytest

from reactmetric.scorer import select_info_level


def _summary(shares, d=0.1):
    return {(lead, info, d, "oracle_B"): {"recovered": v}
            for info, row in shares.items() for lead, v in row.items()}


def test_lowest_level_passing_at_every_lead():
    s = _summary({3.0: {5: 0.93, 10: 0.96}, 4.2: {5: 0.97, 10: 0.96}, 6.0: {5: 1.0, 10: 1.0}})
    level, shares = select_info_level(s, distortion=0.1)
    assert level == 4.2 and shares[3.0][5] == 0.93


def test_no_level_passes():
    s = _summary({3.0: {5: 0.5}, 6.0: {5: 0.9}})
    assert select_info_level(s, distortion=0.1)[0] is None


def test_other_targets_and_models_ignored():
    s = _summary({3.0: {5: 0.99}}, d=0.0)
    s[(5, 3.0, 0.1, "oracle_B")] = {"recovered": 0.5}
    s[(5, 3.0, 0.0, "CV")] = {"recovered": 0.1}
    assert select_info_level(s, distortion=0.0)[0] == 3.0
    assert select_info_level(s, distortion=0.1)[0] is None


PAPER = os.path.join(os.path.dirname(__file__), "..", "outputs", "final", "eindhoven_floorA")


@pytest.mark.skipif(not glob.glob(os.path.join(PAPER, "summary_*.json")),
                    reason="needs the paper run's outputs")
def test_reproduces_the_papers_level_on_eindhoven():
    res = json.load(open(glob.glob(os.path.join(PAPER, "summary_*.json"))[0]))["results"]
    summary = {(r["lead"], r["info"], r["distortion"], r["model"]): r for r in res}
    assert select_info_level(summary, distortion=0.1)[0] == 4.2
    assert select_info_level(summary, distortion=0.0)[0] == 3.0
