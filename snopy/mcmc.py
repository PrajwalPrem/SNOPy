"""The MCMC fitter. Works with any S2OrbitModel + PriorSet -- it doesn't
know or care which metric it's fitting, or how many extra parameters
that metric has."""
import sys
import time

import numpy as np
from scipy.optimize import minimize
import emcee
from multiprocessing import Pool


class S2OrbitMCMC:
    def __init__(self, model, prior_set):
        self.model = model
        self.priors = prior_set
        self.param_names = prior_set.param_names
        self.n_params = len(self.param_names)

    def log_prior(self, params):
        return self.priors.log_prior(params)

    def log_likelihood(self, params, n_points=1000):
        try:
            kwargs = dict(zip(self.param_names, params))
            self.model.set_parameters(n_points=n_points, verbose=False, **kwargs)
            astro = self.model.compute_observables(self.model.data['t_pos'])
            vel = self.model.compute_observables(self.model.data['t_rv'])

            ra_model = np.array(astro['ra'])
            dec_model = np.array(astro['dec'])
            rv_model = np.array(vel['rv'])

            if not (np.all(np.isfinite(ra_model)) and np.all(np.isfinite(dec_model))
                    and np.all(np.isfinite(rv_model))):
                return -np.inf

            d = self.model.data
            ll = -0.5 * np.sum(((ra_model - d['alpha_obs']) / d['alpha_err']) ** 2
                                + np.log(2 * np.pi * d['alpha_err'] ** 2))
            ll += -0.5 * np.sum(((dec_model - d['delta_obs']) / d['delta_err']) ** 2
                                 + np.log(2 * np.pi * d['delta_err'] ** 2))
            ll += -0.5 * np.sum(((rv_model - d['vlos_obs']) / d['vlos_err']) ** 2
                                 + np.log(2 * np.pi * d['vlos_err'] ** 2))
            return ll
        except Exception:
            return -np.inf

    def log_probability(self, params, n_points=1000):
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(params, n_points=n_points)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    def optimize_initial_params(self, n_points=1000):
        print("\nOptimizing initial parameters...")

        def neg_log_prob(params):
            return -self.log_probability(params, n_points=n_points)

        result = minimize(neg_log_prob, self.priors.initial_guess(), method='L-BFGS-B',
                           bounds=self.priors.bounds_array(), options={'maxiter': 100, 'disp': False})
        print(f"Optimization complete. Log-probability: {-result.fun:.1f}")
        return result.x

    def run_mcmc(self, n_walkers=40, n_steps=20000, burn_in=5000, n_points=1000,
                 n_cores=None, seed=42, out_prefix="snope"):
        np.random.seed(seed)

        best_start = self.optimize_initial_params(n_points=n_points)
        spread = self.priors.initial_spread() * 0.1
        pos = best_start + spread * np.random.randn(n_walkers, self.n_params)

        for i in range(n_walkers):
            for j, p in enumerate(self.param_names):
                lo, hi = self.priors.bounds[p]
                pos[i, j] = np.clip(pos[i, j], lo, hi)

        with Pool(processes=n_cores) as pool:
            sampler = emcee.EnsembleSampler(
                n_walkers, self.n_params, self.log_probability,
                kwargs={'n_points': n_points}, pool=pool)

            print(f"\nRunning MCMC with {n_walkers} walkers for {n_steps} steps...")
            print("-" * 50)
            start_time = time.time()
            last_print = 0
            for i, _ in enumerate(sampler.sample(pos, iterations=n_steps)):
                progress = (i + 1) / n_steps * 100
                pct = int(progress)
                if pct > last_print:
                    elapsed = time.time() - start_time
                    sps = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (n_steps - i - 1) / sps if sps > 0 else 0
                    if eta < 60:
                        eta_str = f"{eta:.0f} sec"
                    elif eta < 3600:
                        eta_str = f"{eta / 60:.1f} min"
                    else:
                        eta_str = f"{eta / 3600:.1f} hr"
                    acc = np.mean(sampler.acceptance_fraction)
                    print(f"Progress: {progress:.1f}% | Step {i + 1}/{n_steps} | "
                          f"Accept: {acc:.3f} | ETA: {eta_str}")
                    last_print = pct
                    sys.stdout.flush()
            print("-" * 50)
            print(f"MCMC complete! Runtime: {time.time() - start_time:.1f} seconds")

            log_prob_samples = sampler.get_log_prob()
            flat_log_probs = log_prob_samples[burn_in:].reshape(-1)

        samples = sampler.get_chain()
        flat_samples = sampler.get_chain(discard=burn_in, flat=True)

        np.save(f'{out_prefix}_flat_samples.npy', flat_samples)
        np.save(f'{out_prefix}_samples.npy', samples)
        np.save(f'{out_prefix}_log_likelihoods.npy', flat_log_probs)
        print(f"\nSaved MCMC samples with prefix '{out_prefix}_*.npy'")

        print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")

        try:
            tau_all = sampler.get_autocorr_time(quiet=True)
            print("Autocorrelation times per parameter:")
            for name, tau in zip(self.param_names, tau_all):
                print(f"  {name}: {tau:.1f} steps")
        except emcee.autocorr.AutocorrError:
            print("Warning: Autocorrelation time estimation failed for some parameters.")

        best_fit = {}
        for i, name in enumerate(self.param_names):
            lo, med, hi = np.percentile(flat_samples[:, i], [16, 50, 84])
            best_fit[name] = {'median': med, 'lower': med - lo, 'upper': hi - med}
            print(f"{name}: {med:.4g} (+{hi - med:.4g}, -{med - lo:.4g})")

        log_priors = np.array([self.log_prior(s) for s in flat_samples])
        log_likelihoods = flat_log_probs - log_priors
        best_idx = np.argmax(log_likelihoods)
        best_fit_params = flat_samples[best_idx]

        return sampler, flat_samples, best_fit, best_fit_params, flat_log_probs
