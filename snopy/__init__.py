"""
SNOPE -- Spherical-symmetric spacetime Numerical Orbit and Parameter Estimation.

One integrator, one MCMC engine, one plotting module, shared across every
spacetime you want to test against the S2 star's orbit around Sgr A*.
Each metric (Schwarzschild-de Sitter, quadratic gravity, Brans-Dicke, or
whatever you add next) is just a small file under `snope.metrics` that
defines g_tt(r), g_rr(r), the new-physics parameter(s), and their priors.

    from snope.metrics.sds import main
    sampler, flat_samples, best_fit = main()

See the README for the full guide.
"""

__version__ = "0.1.0"
