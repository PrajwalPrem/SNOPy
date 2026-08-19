"""
The interface every metric plug-in file implements.

A metric here is any static, diagonal, equatorial spacetime:

    ds^2 = g_tt(r) dt^2 + g_rr(r) dr^2 + r^2 dphi^2

with r measured in units of R_s = 2 G M_bh / c^2 (the horizon sits near
r=1). g_tt and g_rr don't need to satisfy g_rr = -1/g_tt -- the orbit
integrator uses the fully general diagonal-metric Christoffel symbols,
so any independent g_tt(r), g_rr(r) works (this is a standard GR
identity, not a special case; see
`snope.orbit_model.S2OrbitModel.geodesic_derivatives`).

To add a new metric, create one file under snope/metrics/ (use sds.py,
quadratic_gravity.py, or brans_dicke.py as a template) that defines:

  1. gtt(r, R_s, **extra), grr(r, R_s, **extra) and their r-derivatives.
     `R_s` (metres) is passed along in case your parameter has physical
     units that need converting into the dimensionless r used here --
     that's what SdS's cosmological constant needs. Ignore it otherwise.
  2. A `MetricSpec` bundling those functions with the name(s) of your
     new parameter(s).
  3. A `param_priors` dict (see snope.priors.ParamPrior) giving both a
     flat range and a Gaussian (mean, std) for every parameter -- the
     usual 13 orbital/offset parameters plus your new one(s).
  4. `main = make_main(metric, param_priors, ...)` from snope.pipeline,
     so people can just do:

         from snope.metrics.my_metric import main
         sampler, flat_samples, best_fit = main()

Nothing else in the package needs to change.
"""
from dataclasses import dataclass, field
from typing import Callable, Sequence


@dataclass
class MetricSpec:
    name: str                                     # short id -> output filename prefix
    extra_param_names: Sequence[str]               # e.g. ("k",) or ("log10_lambda",) or ()
    gtt: Callable                                   # gtt(r, R_s, **extra) -> float
    grr: Callable                                   # grr(r, R_s, **extra) -> float
    dgtt_dr: Callable                               # d(gtt)/dr (r, R_s, **extra) -> float
    dgrr_dr: Callable                               # d(grr)/dr (r, R_s, **extra) -> float
    horizon_radius: Callable = field(default=lambda R_s, **extra: 1.0)
    keplerian_mass_guess: Callable = field(default=lambda R_s, **extra: 0.5)
    description: str = ""
