# SNOPy: Spherical-symmetric spacetime Numerical Orbit and Parameter inference in Python

# We have a single integrator, MCMC engine and plotting module that is used across any spherical symmetric spacetime you want to test using the S2 star's orbit.
# Each metric is just a small file under `snope.metrics` that defines g_tt(r), g_rr(r), the new-physics parameter(s), and their priors. This has to be defined. 
# We provide examples of Schwarzschild and Schwarzschild de Sitter (used in arXiv: 2606.13356)
# For example,

# from snope.metrics.sds import main
# sampler, flat_samples, best_fit = main()

__version__ = "0.1.0"
