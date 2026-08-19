"""Quadratic-gravity black hole metric plug-in (deformation parameter k).

*** THIS IS THE FILE TO EDIT to change the quadratic-gravity setup. ***
Everything below -- the metric functions and the parameter priors -- is
self-contained; nothing elsewhere in the package needs to change.

    g_tt(r,k) = -(1 - rh/r) * (1 - k*(rh/r)*(y1(r,rh)/y2(r,rh)))
    g_rr(r,k) = 1 / [ (1 - rh/r) * (1 - k*(rh/r)*(z1(r,rh)/z2(r,rh))) ]

g_tt and g_rr are INDEPENDENT functions here (g_rr != -1/g_tt for k!=0,
unlike Schwarzschild/SdS) -- that's exactly why the shared integrator in
snope/orbit_model.py uses the fully general diagonal-metric Christoffel
symbols rather than the Schwarzschild shortcut.

The horizon radius rh(k) = 1/(1+ALPHA_RH*k) is chosen (via a symbolic
large-r expansion of g_tt) so the asymptotic geometric mass M_eff stays
fixed at 0.5 for every k -- i.e. so the MCMC's M_bh parameter is always
the true asymptotic gravitational mass, with k a pure deformation on top
of it (no M_bh-k degeneracy at leading order). This is metric-specific
physics, so it lives here, not in the shared core.

y1, y2, z1, z2 are large rational-function coefficients from the
quadratic-gravity solution; they're built symbolically once at import
time and lambdified to fast numpy ufuncs, so there's no per-integration-
step overhead (safe to use inside solve_ivp's right-hand side at full
MCMC scale, same as the original code).
"""
import sympy as sp

from . import MetricSpec
from ..priors import ParamPrior
from ..pipeline import make_main
from ..priors import PriorMode

_r, _rh, _k = sp.symbols('r rh k', real=True)

_y1 = (7094296364854698294656777815 * _r ** 3
       + 2700140790021572890363934045 * _r ** 2 * _rh
       + 32852984866789222219083981378 * _r * _rh ** 2
       - 4194480693404458083513273360 * _rh ** 3)
_y2 = (61001803863561 * _r
       * (39646131244569649 * _r ** 2
          - 24556525364789942 * _r * _rh
          + 156809140779977329 * _rh ** 2))
_gtt_expr = -(1 - _rh / _r) * (1 - _k * (_rh / _r) * (_y1 / _y2))

_z1 = (4129175240772755346390531095019236292556679216644425 * _r ** 5
       - 2419264474466871444593945974852247509452968701438426 * _r ** 4 * _rh
       + 23942186657879314986406923695111115458099209438929019 * _r ** 3 * _rh ** 2
       - 15489702931362934051353960889290571370446686457698338 * _r ** 2 * _rh ** 3
       + 25491571207833688105663124958526350757089950850589160 * _r * _rh ** 4
       - 4366380088284333123737827412366515654073268093718080 * _rh ** 5)
_z2 = (5665908544651409241 * _r
       * (39646131244569649 * _r ** 2 - 24556525364789942 * _r * _rh + 156809140779977329 * _rh ** 2)
       * (6266529735540821295 * _r ** 2 - 3742896005107026923 * _r * _rh + 11207698915983181988 * _rh ** 2))
_grr_expr = 1 / ((1 - _rh / _r) * (1 - _k * (_rh / _r) * (_z1 / _z2)))

_dgtt_dr_expr = sp.diff(_gtt_expr, _r)
_dgrr_dr_expr = sp.diff(_grr_expr, _r)

_gtt_num = sp.lambdify((_r, _rh, _k), _gtt_expr, modules='numpy')
_grr_num = sp.lambdify((_r, _rh, _k), _grr_expr, modules='numpy')
_dgtt_dr_num = sp.lambdify((_r, _rh, _k), _dgtt_dr_expr, modules='numpy')
_dgrr_dr_num = sp.lambdify((_r, _rh, _k), _dgrr_dr_expr, modules='numpy')

# From the large-r expansion g_tt -> -1 + 2*M_eff(k)/r, 2*M_eff(k) = rh*(1+ALPHA_RH*k):
# rh(k) = 1/(1+ALPHA_RH*k) keeps M_eff = 0.5 for every k (verified symbolically).
ALPHA_RH = 6857795 / 2337860877  # = 0.0029333632...


def compute_rh(k):
    return 1.0 / (1.0 + ALPHA_RH * k)


def gtt(r, R_s, k):
    return _gtt_num(r, compute_rh(k), k)


def grr(r, R_s, k):
    return _grr_num(r, compute_rh(k), k)


def dgtt_dr(r, R_s, k):
    return _dgtt_dr_num(r, compute_rh(k), k)


def dgrr_dr(r, R_s, k):
    return _dgrr_dr_num(r, compute_rh(k), k)


metric = MetricSpec(
    name="quadgrav",
    extra_param_names=("k",),
    gtt=gtt, grr=grr, dgtt_dr=dgtt_dr, dgrr_dr=dgrr_dr,
    horizon_radius=lambda R_s, k: compute_rh(k),
    keplerian_mass_guess=lambda R_s, k: compute_rh(k) / 2.0,
    description="Quadratic gravity: independent g_tt(r,k), g_rr(r,k) deformation",
)

# ---------------------------------------------------------------------
# PARAMETER PRIORS -- edit bounds / (mean, std) here to change the fit.
# Values below follow the Gillessen-style prior table (M/D/a/e/i/omega/
# Omega flat; k flat over its full deformation range). `prior_mode`
# (passed to main()) decides flat vs mixed vs all-Gaussian, exactly as
# in snope/metrics/sds.py.
# ---------------------------------------------------------------------
param_priors = {
    'M_bh':     ParamPrior(bounds=(3.7e6, 4.9e6),      gaussian=(4.3e6, 0.2e6)),
    'distance': ParamPrior(bounds=(5.33, 11.33),        gaussian=(8.33, 0.5)),
    'a':        ParamPrior(bounds=(112.0, 139.0),       gaussian=(125.5, 2.0)),
    'e':        ParamPrior(bounds=(0.855, 0.912),       gaussian=(0.884, 0.005)),
    'i':        ParamPrior(bounds=(128.17, 140.20),     gaussian=(134.18, 1.5)),
    'omega':    ParamPrior(bounds=(56.95, 74.03),       gaussian=(65.49, 2.0)),
    'Omega':    ParamPrior(bounds=(217.95, 235.94),     gaussian=(226.95, 2.0)),
    't_peri':   ParamPrior(bounds=(2002.07, 2002.57),   gaussian=(2002.32, 0.05)),
    'x0':       ParamPrior(bounds=(-0.001, 0.001),      gaussian=(0.0, 0.0002)),
    'y0':       ParamPrior(bounds=(-0.001, 0.001),      gaussian=(0.0, 0.0002)),
    'vx0':      ParamPrior(bounds=(-0.0005, 0.0005),    gaussian=(0.0, 0.0001)),
    'vy0':      ParamPrior(bounds=(-0.0005, 0.0005),    gaussian=(0.0, 0.0001)),
    'vz0':      ParamPrior(bounds=(-25.0, 25.0),        gaussian=(0.0, 5.0)),
    'k':        ParamPrior(bounds=(-100.0, 0.0),        gaussian=(-50.0, 20.0), init=-1.0, init_std=5.0),
}

main = make_main(
    metric, param_priors, default_prior_mode=PriorMode.MIXED,
    data_path_pos="tab_gillessen_pos.csv", data_path_rv="tab_gillessen_vr.csv",
    n_walkers=40, n_steps=30000, burn_in=3000, n_points=1000,
)
