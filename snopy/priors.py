"""Priors: flat, mixed (flat + Gaussian), and all-Gaussian modes.

Each parameter is described once, in the metric's config file, by a
`ParamPrior` that carries both a flat range and a Gaussian (mean, std).
Which one is actually used for a given parameter is decided entirely by
the `PriorMode` passed to `PriorSet` / `main()`:

    PriorMode.FLAT     - every parameter uses its flat `bounds`
    PriorMode.MIXED     - parameters in `gaussian_params` (by default
                          t_peri, x0, y0, vx0, vy0, vz0 -- the offset/
                          reference-epoch nuisance parameters) use their
                          Gaussian (mean, std); everything else is flat
    PriorMode.GAUSSIAN  - every parameter uses its Gaussian (mean, std)

The hard bounds handed to emcee/L-BFGS-B are always the flat `bounds`
when a parameter is flat, or mean +/- N_SIGMA_BOUND*std when it is
Gaussian (this keeps a technically-infinite Gaussian support finite
without ever letting the sampler get close to clipping it).
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

N_SIGMA_BOUND = 5.0

DEFAULT_GAUSSIAN_PARAMS = ('t_peri', 'x0', 'y0', 'vx0', 'vy0', 'vz0')


class PriorMode(str, Enum):
    FLAT = "flat"
    MIXED = "mixed"
    GAUSSIAN = "gaussian"


@dataclass
class ParamPrior:
    """One parameter's prior, in both flavors at once.

    bounds:   (lo, hi) flat / uniform range
    gaussian: (mean, std), or None if this parameter can never be
              sampled with a Gaussian prior
    init:     starting value for the optimizer; defaults to the bounds
              midpoint (flat) or the Gaussian mean
    init_std: spread used to scatter the initial walker ball around the
              optimizer's best point; defaults to (hi-lo)/6 (flat) or
              std (Gaussian)
    """
    bounds: Tuple[float, float]
    gaussian: Optional[Tuple[float, float]] = None
    init: Optional[float] = None
    init_std: Optional[float] = None

    def midpoint(self):
        return 0.5 * (self.bounds[0] + self.bounds[1])


class PriorSet:
    """Resolves {name: ParamPrior} + a PriorMode into the concrete
    per-parameter (use_gaussian, bounds, gaussian, init, init_std)
    tables consumed by S2OrbitMCMC."""

    def __init__(self, param_priors: Dict[str, ParamPrior], mode: PriorMode = PriorMode.MIXED,
                 gaussian_params: Iterable[str] = DEFAULT_GAUSSIAN_PARAMS):
        self.param_names = list(param_priors.keys())
        self.param_priors = param_priors
        self.mode = PriorMode(mode)
        self.gaussian_params = set(gaussian_params)

        self.use_gaussian = {}
        self.bounds = {}
        self.gaussian = {}
        self.init = {}
        self.init_std = {}

        for name, pp in param_priors.items():
            if self.mode == PriorMode.FLAT:
                use_g = False
            elif self.mode == PriorMode.GAUSSIAN:
                use_g = True
            else:  # MIXED
                use_g = name in self.gaussian_params

            if use_g and pp.gaussian is None:
                raise ValueError(
                    f"Parameter '{name}' has no Gaussian prior defined, but prior "
                    f"mode '{self.mode.value}' requires one for this parameter."
                )

            self.use_gaussian[name] = use_g

            if use_g:
                mean, std = pp.gaussian
                self.gaussian[name] = (mean, std)
                self.bounds[name] = (mean - N_SIGMA_BOUND * std, mean + N_SIGMA_BOUND * std)
                self.init[name] = pp.init if pp.init is not None else mean
                self.init_std[name] = pp.init_std if pp.init_std is not None else std
            else:
                self.bounds[name] = pp.bounds
                self.init[name] = pp.init if pp.init is not None else pp.midpoint()
                lo, hi = pp.bounds
                self.init_std[name] = pp.init_std if pp.init_std is not None else (hi - lo) / 6.0

    def log_prior(self, params):
        lp = 0.0
        for name, value in zip(self.param_names, params):
            lo, hi = self.bounds[name]
            if not (lo <= value <= hi):
                return -np.inf
            if self.use_gaussian[name]:
                mean, std = self.gaussian[name]
                lp += -0.5 * ((value - mean) / std) ** 2 - np.log(std * np.sqrt(2 * np.pi))
        return lp

    def initial_guess(self):
        return np.array([self.init[p] for p in self.param_names])

    def initial_spread(self):
        return np.array([self.init_std[p] for p in self.param_names])

    def bounds_array(self):
        return [self.bounds[p] for p in self.param_names]
