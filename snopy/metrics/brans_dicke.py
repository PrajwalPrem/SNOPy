"""Brans-Dicke (PPN-parametrized) metric plug-in, deformation parameter omega_bd.

*** THIS IS THE FILE TO EDIT to change the Brans-Dicke setup. ***
Everything below -- the metric functions and the parameter priors -- is
self-contained; nothing elsewhere in the package needs to change.

    g_tt(r) = -(1 - 2M/r)
    g_rr(r) =  1 + beta * (2M/r),      beta = (1+omega_bd)/(2+omega_bd)

M is fixed at the Schwarzschild convention M = 0.5 (r in units of
R_s = 2GM_bh/c^2, horizon at r=1). g_tt is exactly Schwarzschild;
g_rr carries the single Brans-Dicke deformation through beta(omega_bd).
As omega_bd -> infinity, beta -> 0 and g_rr -> 1, so this metric ->
Schwarzschild (GR) exactly, as expected in the Brans-Dicke limit.

Like quadratic gravity, g_rr != -1/g_tt here for finite omega_bd, so
this metric relies on the shared integrator's general diagonal-metric
Christoffel symbols (snope/orbit_model.py) rather than a Schwarzschild
shortcut.

NOTE ON THE PRIOR: solar-system tests constrain the real Brans-Dicke
omega_bd to > ~40000 (i.e. very close to GR), but this file's default
prior spans a much smaller, more sensitive range so an S2-only fit can
actually constrain omega_bd from this dataset alone -- widen/narrow
`param_priors['omega_bd']` below to whatever range your analysis needs.
"""
from . import MetricSpec
from ..priors import ParamPrior
from ..pipeline import make_main
from ..priors import PriorMode

M_VAL = 0.5  # fixed geometric mass; r is in units of R_s


def _beta(omega_bd):
    return (1.0 + omega_bd) / (2.0 + omega_bd)


def gtt(r, R_s, omega_bd):
    return -(1 - 2 * M_VAL / r)


def grr(r, R_s, omega_bd):
    return 1.0 + _beta(omega_bd) * (2 * M_VAL / r)


def dgtt_dr(r, R_s, omega_bd):
    return -2 * M_VAL / r ** 2


def dgrr_dr(r, R_s, omega_bd):
    return -_beta(omega_bd) * (2 * M_VAL) / r ** 2


metric = MetricSpec(
    name="bransdicke",
    extra_param_names=("omega_bd",),
    gtt=gtt, grr=grr, dgtt_dr=dgtt_dr, dgrr_dr=dgrr_dr,
    horizon_radius=lambda R_s, **extra: 1.0,
    keplerian_mass_guess=lambda R_s, **extra: M_VAL,
    description="Brans-Dicke (PPN form): g_tt Schwarzschild, g_rr deformed by omega_bd",
)

# ---------------------------------------------------------------------
# PARAMETER PRIORS -- edit bounds / (mean, std) here to change the fit.
# `prior_mode` (passed to main()) decides flat vs mixed vs all-Gaussian,
# exactly as in snope/metrics/sds.py and quadratic_gravity.py.
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
    # EXAMPLE range only -- see the module docstring above.
    'omega_bd': ParamPrior(bounds=(2.5, 200.0),        gaussian=(20.0, 15.0), init=10.0, init_std=5.0),
}

main = make_main(
    metric, param_priors, default_prior_mode=PriorMode.MIXED,
    data_path_pos="tab_gillessen_pos.csv", data_path_rv="tab_gillessen_vr.csv",
    n_walkers=40, n_steps=30000, burn_in=3000, n_points=1000,
)
