"""Schwarzschild-de Sitter (SdS) metric plug-in.

*** THIS IS THE FILE TO EDIT to change the SdS setup. *** Everything
below -- the metric functions and the parameter priors -- is
self-contained; nothing elsewhere in the package needs to change.

    f(r) = 1 - 2M/r - (Lambda_geo * r^2)/3        g_tt = -f(r),  g_rr = 1/f(r)

M is fixed at the Schwarzschild convention M = 0.5 (r is measured in
units of R_s = 2 G M_bh / c^2, so the horizon sits at r=1). The single
extra parameter is the cosmological constant Lambda, sampled as
log10_lambda (physical units of m^-2, since Lambda's natural scale spans
many orders of magnitude). Lambda_geo = Lambda * R_s^2 converts it into
the dimensionless units used by the integrator -- so, unlike the other
two metrics here, SdS's g_tt/g_rr genuinely depend on R_s (i.e. on
M_bh) as well as on log10_lambda; that's why gtt/grr below take R_s.
"""
from . import MetricSpec
from ..priors import ParamPrior
from ..pipeline import make_main
from ..priors import PriorMode

M_VAL = 0.5  # fixed geometric mass; r is in units of R_s


def _f(r, R_s, log10_lambda):
    Lambda_geo = (10 ** log10_lambda) * R_s ** 2
    return 1 - 2 * M_VAL / r - (Lambda_geo * r ** 2) / 3


def _df_dr(r, R_s, log10_lambda):
    Lambda_geo = (10 ** log10_lambda) * R_s ** 2
    return 2 * M_VAL / r ** 2 - (2 * Lambda_geo * r) / 3


def gtt(r, R_s, log10_lambda):
    return -_f(r, R_s, log10_lambda)


def grr(r, R_s, log10_lambda):
    return 1.0 / _f(r, R_s, log10_lambda)


def dgtt_dr(r, R_s, log10_lambda):
    return -_df_dr(r, R_s, log10_lambda)


def dgrr_dr(r, R_s, log10_lambda):
    f = _f(r, R_s, log10_lambda)
    return -_df_dr(r, R_s, log10_lambda) / f ** 2


metric = MetricSpec(
    name="sds",
    extra_param_names=("log10_lambda",),
    gtt=gtt, grr=grr, dgtt_dr=dgtt_dr, dgrr_dr=dgrr_dr,
    horizon_radius=lambda R_s, **extra: 1.0,
    keplerian_mass_guess=lambda R_s, **extra: M_VAL,
    description="Schwarzschild-de Sitter: f(r) = 1 - 2M/r - Lambda*r^2/3",
)

# ---------------------------------------------------------------------
# PARAMETER PRIORS -- edit bounds / (mean, std) here to change the fit.
# Every parameter carries BOTH a flat range and a Gaussian description;
# `prior_mode` (passed to main()) decides which is actually used:
#   PriorMode.FLAT      - every parameter flat
#   PriorMode.MIXED      - t_peri/offsets Gaussian, rest flat  (default)
#   PriorMode.GAUSSIAN   - every parameter Gaussian
# ---------------------------------------------------------------------
param_priors = {
    'M_bh':     ParamPrior(bounds=(3.5e6, 5.0e6),     gaussian=(4.2e6, 0.3e6)),
    'distance': ParamPrior(bounds=(7.2, 9.2),          gaussian=(8.2, 0.3)),
    'a':        ParamPrior(bounds=(115.0, 135.0),      gaussian=(124.0, 1.0)),
    'e':        ParamPrior(bounds=(0.86, 0.90),        gaussian=(0.884, 0.005)),
    'i':        ParamPrior(bounds=(130.0, 138.0),      gaussian=(134.18, 2.0)),
    'omega':    ParamPrior(bounds=(60.0, 72.0),        gaussian=(65.1, 3.0)),
    'Omega':    ParamPrior(bounds=(224.0, 230.0),      gaussian=(226.07, 3.0)),
    't_peri':   ParamPrior(bounds=(2002.17, 2002.47),  gaussian=(2002.33, 0.05)),
    'x0':       ParamPrior(bounds=(-0.017, 0.017),     gaussian=(0.0, 0.0057735)),
    'y0':       ParamPrior(bounds=(-0.017, 0.017),     gaussian=(0.0, 0.0057735)),
    'vx0':      ParamPrior(bounds=(-0.00017, 0.00017), gaussian=(0.0, 0.000057735)),
    'vy0':      ParamPrior(bounds=(-0.00017, 0.00017), gaussian=(0.0, 0.000057735)),
    'vz0':      ParamPrior(bounds=(-86.6, 86.6),       gaussian=(0.0, 28.8675)),
    'log10_lambda': ParamPrior(bounds=(-70.0, -20.0),  gaussian=(-52.0, 1.0)),
}

main = make_main(
    metric, param_priors, default_prior_mode=PriorMode.MIXED,
    data_path_pos="tab_gillessen_pos.csv", data_path_rv="tab_gillessen_vr.csv",
    n_walkers=40, n_steps=30000, burn_in=3000, n_points=1000,
)
