import warnings

import pytest

import reactmetric as rm


@pytest.fixture(scope="session")
def synthetic():
    data, events = rm.datasets.synthetic_regime_changes(n=80, seed=21)
    calib, test = data.split(0.3, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = rm.reference.Climatology.fit(calib, leads=(1, 5, 10, 15), mode="kinematic")
    return {"data": data, "events": events, "calib": calib, "test": test, "ref": ref}
