"""Lightweight sanity checks -- not a full numerical validation suite, but
enough to catch import errors, interface mismatches, or broken metric
files. Run with: `pytest tests/` (needs synthetic or real data files;
see conftest-style fixture below, adjust paths as needed).

These tests are deliberately cheap: small n_points, no MCMC. They check
that every metric integrates a closed orbit and returns finite,
physically sane observables -- they do NOT check astrophysical accuracy.
"""
import os
import numpy as np
import pandas as pd
import pytest

from snope.orbit_model import S2OrbitModel
from snope.priors import PriorSet, PriorMode


@pytest.fixture(scope="module")
def synthetic_data(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(0)

    t_pos = np.linspace(1995, 2020, 40)
    alpha = 50 * np.sin((t_pos - 2002.32) / 16 * 2 * np.pi) + rng.normal(0, 0.5, t_pos.size)
    delta = 50 * np.cos((t_pos - 2002.32) / 16 * 2 * np.pi) + rng.normal(0, 0.5, t_pos.size)
    pos_path = tmp / "pos.csv"
    pd.DataFrame({'t': t_pos, 'a': alpha, 'ae': np.full_like(t_pos, 0.5),
                  'd': delta, 'de': np.full_like(t_pos, 0.5)}).to_csv(pos_path, header=False, index=False)

    t_rv = np.linspace(2000, 2005, 15)
    vlos = 2000 * np.sin((t_rv - 2002.32) / 16 * 2 * np.pi) + rng.normal(0, 20, t_rv.size)
    rv_path = tmp / "rv.csv"
    pd.DataFrame({'t': t_rv, 'v': vlos, 've': np.full_like(t_rv, 20.0)}).to_csv(rv_path, header=False, index=False)

    return str(pos_path), str(rv_path)


@pytest.mark.parametrize("metric_module,extra_kwargs", [
    ("snope.metrics.sds", {"log10_lambda": -52.0}),
    ("snope.metrics.quadratic_gravity", {"k": -1.0}),
    ("snope.metrics.brans_dicke", {"omega_bd": 10.0}),
])
def test_metric_integrates_and_returns_finite_observables(synthetic_data, metric_module, extra_kwargs):
    import importlib
    mod = importlib.import_module(metric_module)
    pos_path, rv_path = synthetic_data

    model = S2OrbitModel(mod.metric, pos_path, rv_path, verbose=False)
    model.set_parameters(M_bh=4.3e6, distance=8.33, a=125.5, e=0.884, i=134.18,
                          omega=65.49, Omega=226.95, t_peri=2002.32,
                          n_points=300, verbose=False, **extra_kwargs)

    obs = model.compute_observables(model.data['t_pos'])
    assert np.all(np.isfinite(obs['ra']))
    assert np.all(np.isfinite(obs['dec']))

    vel = model.compute_observables(model.data['t_rv'])
    assert np.all(np.isfinite(vel['rv']))
    # S2 never exceeds ~10,000 km/s even at closest periastron passages.
    assert np.max(np.abs(vel['rv'])) < 15000


@pytest.mark.parametrize("mode", [PriorMode.FLAT, PriorMode.MIXED, PriorMode.GAUSSIAN])
def test_prior_modes_are_finite_at_init(mode):
    from snope.metrics.sds import param_priors
    priors = PriorSet(param_priors, mode=mode)
    lp = priors.log_prior(priors.initial_guess())
    assert np.isfinite(lp)
