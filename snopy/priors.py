# We already have 3 kinds of Priors set up: flat, mixed (flat + Gaussian), and all-Gaussian modes
# PriorMode.FLAT: every parameter uses its flat bounds
# PriorMode.MIXED: parameters in gaussian_params (by default t_peri, x0, y0, vx0, vy0, vz0) use their Gaussian (mean, std); everything else is flat
# PriorMode.GAUSSIAN: every parameter uses its Gaussian (mean, std)

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
    
    bounds: Tuple[float, float]
    gaussian: Optional[Tuple[float, float]] = None
    init: Optional[float] = None
    init_std: Optional[float] = None

    def midpoint(self):
        return 0.5 * (self.bounds[0] + self.bounds[1])


class PriorSet:

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
