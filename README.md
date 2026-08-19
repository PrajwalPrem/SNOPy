# SNOPE

**S**pherical-symmetric spacetime **N**umerical **O**rbit and **P**arameter **E**stimation.

<p align="center">
  <img src="logo_SNOPE.png" width="700">
</p>

SNOPE is a simple Python framework for modelling stellar orbits in a static, spherically symmetric spacetime of choice and performing Bayesian parameter estimation using Markov Chain Monte Carlo (MCMC).

It provides a common geodesic integration and inference pipeline that is independent of the underlying gravity model. New spacetime metrics can be incorporated easily by specifying only the metric components ($tt$ and $rr$) and the associated model parameters. 
The numerical integration, likelihood evaluation, optimization, sampling, and plotting are handled automatically.

We include implementation examples for

- Schwarzschild spacetime
- Schwarzschild–de Sitter spacetime
- Quadratic Gravity *(coming soon)*
- Brans–Dicke Gravity *(coming soon)*

Although we currently use the S2 star as the reference dataset, the framework is designed to accommodate other stellar orbits with minimal modification.

## Installation

Clone the repository

```bash
git clone https://github.com/PrajwalPrem/SNOPE.git
cd SNOPE
pip install -r requirements.txt
```
Install the required packages

```bash
pip install -r requirements.txt
```
Needs Python 3.9+. The MCMC uses `emcee`'s together with multiprocessing `Pool`.
Standalone scripts should therefore be protected with `if __name__ == "__main__":` on platforms where multiprocessing requires it. 
Jupyter notebooks require no additional setup.

## Structure

```
snope/
├── constants.py          Physical constants
├── data.py               S2 astrometry/RV data loader
├── orbit_model.py        (Metric-independent) Physics engine that includes the Geodesic equation (Christoffel symbols etc)
├── priors.py             Priors 
├── mcmc.py               MCMC 
├── plotting.py           Corner, convergence plots
├── pipeline.py           Fitting pipeline
│
├── metrics/
│   ├── __init__.py
│   ├── schwarzschild.py
│   ├── sds.py
│   ├── quadratic_gravity.py
│   └── brans_dicke.py
│
├── notebooks/            Example notebooks
├── data/                 Observational data
└── tests/                Unit tests
```
## Data

The repository includes the astrometric and radial-velocity measurements of the S2 star from

> Gillessen et al. (2017),
> *An Update on Monitoring Stellar Orbits in the Galactic Center*,
> ApJ **837**, 30.
   
The corresponding VizieR catalogue is also available at

https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/ApJ/837/30

If this dataset is used in scientific work, please cite both the original publication and the associated VizieR catalogue.

## Quickstart

Running an MCMC analysis for the Schwarzschild–de Sitter metric requires only a few lines of code.

```python
from snope.metrics.sds import main

sampler, flat_samples, best_fit = main(
    data_path_pos="data/tab_gillessen_pos.csv",
    data_path_rv="data/tab_gillessen_vr.csv",
)
```
The pipeline automatically

The orbit modelling already takes into account - we are numerically evaluating the full geodesic equation and hence no approximations at the physics level. It also includes additional effects like delays etc. 

1. numerically integrates the full geodesic equations for the chosen spacetime
2. includes relativistic effects, example, light-travel (Roemer) delay, doppler delay
3. optimizes the initial parameter vector using L-BFGS-B
4. initializes the MCMC walkers, performs Bayesian parameter estimation with `emcee`
5. computes the best-fitting parameters and reports χ², AIC, and BIC,
6. produces diagnostic plots.

To fit another spacetime, simply import the corresponding metric module.

```python
from snope.metrics.brans_dicke import main
```

or

```python
from snope.metrics.quadratic_gravity import main
```
No further changes to the analysis pipeline are required.

Pipeline parameters may be overridden directly.

```python
from snope.priors import PriorMode

sampler, flat_samples, best_fit = main(
    prior_mode=PriorMode.FLAT,
    n_walkers=64,
    n_steps=50000,
    burn_in=5000,
    n_points=1000,
    n_cores=8,
)
```
Example notebooks are provided in the `notebooks/` directory.

## An overview of the physics

Each spacetime is assumed to be **static and spherically symmetric** with line element written as,

```text
ds² = g_tt(r) dt² + g_rr(r) dr² + r² dφ²,
```

where the radial coordinate is measured in units of

```text
R_s = 2GM/c².
```

The orbit model `S2OrbitModel` integrates the geodesic equations using the Christoffel symbols of a general diagonal metric.

```
Gamma^t_tr     =  g_tt'/(2 g_tt)
Gamma^r_tt     = -g_tt'/(2 g_rr)
Gamma^r_rr     =  g_rr'/(2 g_rr)
Gamma^r_phiphi = -r/g_rr
Gamma^phi_rphi =  1/r
```
We do not assume any further gauge conditions like `g_rr = -1/g_tt`. The conserved energy and angular momentum are computed from the orbital turning-point conditions. So, it is advisable to verify that the effective potential of the chosen metric,

```text
V_eff(r) = -g_tt(r) (1 + L²/r²),
```
admits a bound orbit. A bound orbit requires two turning points. 

## Adding a new metric

Adding a new gravity model requires only a single metric file. The new module must provide

- `g_tt(r)`
- `g_rr(r)`
- `dg_tt/dr`
- `dg_rr/dr`
- the model parameter,
- parameter priors,
- horizon location (if required).

Here is an example to create `snope/metrics/my_metric.py`:

```python
from . import MetricSpec
from ..priors import ParamPrior
from ..pipeline import make_main
from ..priors import PriorMode

M_VAL = 0.5  # Schwarzschild-convention geometric mass (r in units of R_s)

def gtt(r, R_s, my_param):
    ...   # return g_tt(r)

def grr(r, R_s, my_param):
    ...

def dgtt_dr(r, R_s, my_param):
    ...

def dgrr_dr(r, R_s, my_param):
    ...

metric = MetricSpec(
    name="my_metric",
    extra_param_names=("my_param",),
    gtt=gtt, grr=grr, dgtt_dr=dgtt_dr, dgrr_dr=dgrr_dr,
    horizon_radius=lambda R_s, **extra: 1.0,
    keplerian_mass_guess=lambda R_s, **extra: M_VAL,
    description="One-line description of the metric",
)

param_priors = {
    'M_bh':     ParamPrior(bounds=(3.7e6, 4.9e6), gaussian=(4.3e6, 0.2e6)),
    'distance': ParamPrior(bounds=(5.33, 11.33),  gaussian=(8.33, 0.5)),
    'a':        ParamPrior(bounds=(112.0, 139.0), gaussian=(125.5, 2.0)),
    'e':        ParamPrior(bounds=(0.855, 0.912), gaussian=(0.884, 0.005)),
    'i':        ParamPrior(bounds=(128.17, 140.20), gaussian=(134.18, 1.5)),
    'omega':    ParamPrior(bounds=(56.95, 74.03), gaussian=(65.49, 2.0)),
    'Omega':    ParamPrior(bounds=(217.95, 235.94), gaussian=(226.95, 2.0)),
    't_peri':   ParamPrior(bounds=(2002.07, 2002.57), gaussian=(2002.32, 0.05)),
    'x0':       ParamPrior(bounds=(-0.001, 0.001), gaussian=(0.0, 0.0002)),
    'y0':       ParamPrior(bounds=(-0.001, 0.001), gaussian=(0.0, 0.0002)),
    'vx0':      ParamPrior(bounds=(-0.0005, 0.0005), gaussian=(0.0, 0.0001)),
    'vy0':      ParamPrior(bounds=(-0.0005, 0.0005), gaussian=(0.0, 0.0001)),
    'vz0':      ParamPrior(bounds=(-25.0, 25.0), gaussian=(0.0, 5.0)),
    'my_param': ParamPrior(bounds=(lo, hi), gaussian=(mean, std)),
}

main = make_main(metric, param_priors, default_prior_mode=PriorMode.MIXED,
                  n_walkers=40, n_steps=30000, burn_in=3000, n_points=1000)
```

With this, `from snope.metrics.my_metric import main` now works just
like the other built-in examples. 

## Prior modes

Every parameter stores both

- flat bounds
- Gaussian prior information

The active prior is selected globally.

| mode | orbital elements + new-physics parameter | t_peri, x0, y0, vx0, vy0, vz0 |
|---|:---:|:---:|
| `PriorMode.FLAT`             | flat | flat |
| `PriorMode.MIXED` (default)  | flat | Gaussian |
| `PriorMode.GAUSSIAN`         | Gaussian | Gaussian |

```python
from snope.priors import PriorMode
main(prior_mode=PriorMode.FLAT)
main(prior_mode=PriorMode.MIXED)      # default
main(prior_mode=PriorMode.GAUSSIAN)
```
## Outputs

Each run generates:

```
<prefix>_flat_samples.npy       flattened post-burn-in chain
<prefix>_samples.npy             full chain (n_steps, n_walkers, n_params)
<prefix>_log_likelihoods.npy     post-burn-in log-likelihoods
<prefix>_statistical_info.txt    chi2/AIC/BIC + best-fit params + autocorr times
<prefix>_corner.png
<prefix>_convergence.png
<prefix>_orbit.png               sky orbit (RA vs Dec), RV(t), RA(t), Dec(t)
```

## Performance

The default settings are

```
n_walkers = 40
n_steps = 30000
n_points = 1000
```
It is recommended to test with a smaller number of steps for testing.
Each likelihood evaluation integrates the geodesic equations using SciPy's **DOP853** integrator with `rtol=atol=1e-12`. 

# Citation

If you use **SNOPE** in published research, please cite the original observational dataset used in your analysis (e.g. Gillessen et al. 2017 for the S2 data):

Also consider citing the methodological paper describing the SNOPE framework:
> Hassan Puttasiddappa et.al. (2026),
> *Bounds on Λ at the Galactic Center*,
> xxxxx.

```bibtex
@article{HassanPuttasiddappa:2026cuh,
    author = {Hassan Puttasiddappa, Prajwal and Mushtaq, Muzammil and Ramirez, Willian and Mota, David F.},
    title = {Bounds on {$\Lambda$} at the Galactic Center},
    eprint = {2606.13356},
    archivePrefix = {arXiv},
    primaryClass = {gr-qc},
    year = {2026},
    month = jun
}
```
### Related codes:
1. [*PyGro*](https://github.com/rdellamonica/pygro)- PyGRO: a Python Integrator for General Relativistic Orbits ([arXiv](https://arxiv.org/abs/2504.20152))
2. [*GC-OrbitFit*](https://github.com/pmplewa/GC-OrbitFit) - Uses nested sampling for inference
