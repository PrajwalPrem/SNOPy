"""Schwarzschild metric plug-in -- pure GR, no new-physics parameter.

*** THIS IS THE FILE TO EDIT to change the Schwarzschild setup. ***
Everything below -- the metric functions and the parameter priors -- is
self-contained; nothing elsewhere in the package needs to change.

    g_tt(r) = -(1 - 2M/r)      g_rr(r) = 1 / (1 - 2M/r)

M is fixed at the Schwarzschild convention M = 0.5 (r is measured in
units of R_s = 2 G M_bh / c^2, so the horizon sits at r=1). This is the
baseline GR case: no extra parameter, so `extra_param_names` is empty
and `param_priors` only has the usual 13 orbital/offset parameters.

Useful as a sanity check -- fit this first and confirm you recover the
standard S2/Sgr A* orbital elements (and the expected Schwarzschild
precession) before trying a modified-gravity metric on the same data.
"""
from . import MetricSpec
from ..priors import ParamPrior
from ..pipeline import make_main
from ..priors import PriorMode

M_VAL = 0.5  # fixed geometric mass; r is in units of R_s


def gtt(r, R_s):
    return -(1 - 2 * M_VAL / r)


def grr(r, R_s):
    return 1.0 / (1 - 2 * M_VAL / r)


def dgtt_dr(r, R_s):
    return -2 * M_VAL / r ** 2


def dgrr_dr(r, R_s):
    f = 1 - 2 * M_VAL / r
    df_dr = 2 * M_VAL / r ** 2
    return -df_dr / f ** 2


metric = MetricSpec(
    name="schwarzschild",
    extra_param_names=(),
    gtt=gtt, grr=grr, dgtt_dr=dgtt_dr, dgrr_dr=dgrr_dr,
    horizon_radius=lambda R_s, **extra: 1.0,
    keplerian_mass_guess=lambda R_s, **extra: M_VAL,
    description="Schwarzschild: g_tt = -(1-2M/r), g_rr = 1/(1-2M/r), pure GR",
)

# ---------------------------------------------------------------------
# PARAMETER PRIORS -- edit bounds / (mean, std) here to change the fit.
# No new-physics parameter for this metric, so just the standard 13.
# `prior_mode` (passed to main()) decides flat vs mixed vs all-Gaussian,
# exactly as in the other metric files.
# ---------------------------------------------------------------------
param_priors = {
    'M_bh':     ParamPrior(bounds=(3.7e6, 4.9e6),     gaussian=(4.3e6, 0.2e6)),
    'distance': ParamPrior(bounds=(5.33, 11.33),       gaussian=(8.33, 0.5)),
    'a':        ParamPrior(bounds=(112.0, 139.0),      gaussian=(125.5, 2.0)),
    'e':        ParamPrior(bounds=(0.855, 0.912),      gaussian=(0.884, 0.005)),
    'i':        ParamPrior(bounds=(128.17, 140.20),    gaussian=(134.18, 1.5)),
    'omega':    ParamPrior(bounds=(56.95, 74.03),      gaussian=(65.49, 2.0)),
    'Omega':    ParamPrior(bounds=(217.95, 235.94),    gaussian=(226.95, 2.0)),
    't_peri':   ParamPrior(bounds=(2002.07, 2002.57),  gaussian=(2002.32, 0.05)),
    'x0':       ParamPrior(bounds=(-0.001, 0.001),     gaussian=(0.0, 0.0002)),
    'y0':       ParamPrior(bounds=(-0.001, 0.001),     gaussian=(0.0, 0.0002)),
    'vx0':      ParamPrior(bounds=(-0.0005, 0.0005),   gaussian=(0.0, 0.0001)),
    'vy0':      ParamPrior(bounds=(-0.0005, 0.0005),   gaussian=(0.0, 0.0001)),
    'vz0':      ParamPrior(bounds=(-25.0, 25.0),       gaussian=(0.0, 5.0)),
}

main = make_main(
    metric, param_priors, default_prior_mode=PriorMode.MIXED,
    data_path_pos="tab_gillessen_pos.csv", data_path_rv="tab_gillessen_vr.csv",
    n_walkers=40, n_steps=30000, burn_in=3000, n_points=1000,
)
