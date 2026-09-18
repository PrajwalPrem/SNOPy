# In summary - build a model and set priors for the metric, run the MCMC, produce all diagnostic plots, and return the sampler / samples / best-fit dict.

import multiprocessing

from .orbit_model import S2OrbitModel
from .mcmc import S2OrbitMCMC
from .priors import PriorSet, PriorMode
from . import plotting


def run_fit(metric, param_priors, prior_mode=PriorMode.MIXED,
            data_path_pos="tab_gillessen_pos.csv", data_path_rv="tab_gillessen_vr.csv",
            n_walkers=40, n_steps=30000, burn_in=3000, n_points=1000,
            n_cores=None, out_prefix=None, gaussian_params=None, seed=42,
            refit_n_points=20000, make_plots=True):
    
    out_prefix = out_prefix or metric.name

    model = S2OrbitModel(metric, data_path_pos, data_path_rv, verbose=False)

    prior_kwargs = {}
    if gaussian_params is not None:
        prior_kwargs['gaussian_params'] = gaussian_params
    priors = PriorSet(param_priors, mode=prior_mode, **prior_kwargs)

    mcmc = S2OrbitMCMC(model, priors)

    n_cores = n_cores or multiprocessing.cpu_count()
    print(f"Running on {n_cores} CPU cores | metric = '{metric.name}' | prior mode = '{priors.mode.value}'")

    sampler, flat_samples, best_fit, best_fit_params, _ = mcmc.run_mcmc(
        n_walkers=n_walkers, n_steps=n_steps, burn_in=burn_in, n_points=n_points,
        n_cores=n_cores, seed=seed, out_prefix=out_prefix)

    if make_plots:
        samples = sampler.get_chain()
        plotting.chi2_summary(model, mcmc, best_fit_params, sampler, out_prefix=out_prefix)

        truths = [priors.init[p] for p in mcmc.param_names]
        plotting.corner_plot(mcmc, flat_samples, truths,
                              f'Corner Plot - {metric.name}', out_prefix=out_prefix)
        plotting.convergence_plot(mcmc, samples,
                                   f'Walker Traces - {metric.name}', out_prefix=out_prefix)
        plotting.orbit_plot(model, best_fit, metric.name, out_prefix=out_prefix,
                             n_points=refit_n_points)

    return sampler, flat_samples, best_fit


def make_main(metric, param_priors, default_prior_mode=PriorMode.MIXED, **fixed_defaults):
    
    def main(prior_mode=default_prior_mode, **overrides):
        params = dict(fixed_defaults)
        params.update(overrides)
        return run_fit(metric, param_priors, prior_mode=prior_mode, **params)

    main.__doc__ = (f"Run the full MCMC fit + diagnostics for the '{metric.name}' metric "
                     f"({metric.description}).")
    return main
