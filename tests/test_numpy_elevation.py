from pathlib import Path
import numpy as np
import pytest
from go1_sim2real.policy import NumpyElevationPolicy

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'artifacts/yuanzhe/collective_harder_19499_1072d_20261014'


def test_numpy_matches_original_checkpoint_reference():
    policy=NumpyElevationPolicy(BUNDLE/'policy/policy.ts')
    fixture=np.load(BUNDLE/'validation/parity.npz')
    actual=np.stack([policy(obs) for obs in fixture['observations']])
    np.testing.assert_allclose(actual,fixture['actions'],rtol=1e-5,atol=2e-6)


def test_numpy_rejects_nonfinite_or_wrong_observations():
    policy=NumpyElevationPolicy(BUNDLE/'policy/policy.ts')
    with pytest.raises(ValueError):policy(np.zeros(48))
    with pytest.raises(ValueError):policy(np.full(1072,np.nan))
