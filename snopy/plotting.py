"""Diagnostics shared by every metric: chi^2/AIC/BIC summary, corner plot,
walker-trace (convergence) plot, the 4-panel orbit/RV fit plot, and a
quick effective-potential check."""
import numpy as np
import matplotlib.pyplot as plt
import corner


def chi2_summary(model, mcmc, best_fit_params, sampler, out_prefix="snope"):
    kwargs = dict(zip(mcmc.param_names, best_fit_params))
    model.set_parameters(n_points=1000, verbose=False, **kwargs)

    astro = model.compute_observables(model.data['t_pos'])
    vel = model.compute_observables(model.data['t_rv'])

    ra_model = np.array(astro['ra'])
    dec_model = np.array(astro['dec'])
    rv_model = np.array(vel['rv'])
    d = model.data

    chi2_ra = np.sum(((ra_model - d['alpha_obs']) / d['alpha_err']) ** 2)
    chi2_dec = np.sum(((dec_model - d['delta_obs']) / d['delta_err']) ** 2)
    chi2_rv = np.sum(((rv_model - d['vlos_obs']) / d['vlos_err']) ** 2)
    chi2_total = chi2_ra + chi2_dec + chi2_rv

    n_data = len(d['alpha_obs']) + len(d['delta_obs']) + len(d['vlos_obs'])
    k_npar = mcmc.n_params
    dof = n_data - k_npar
    chi2_reduced = chi2_total / dof if dof > 0 else np.inf
    AIC = chi2_total + 2 * k_npar
    BIC = chi2_total + k_npar * np.log(n_data)

    print("\nChi-squared analysis:")
    print(f"  chi2_RA: {chi2_ra:.2f}  chi2_Dec: {chi2_dec:.2f}  chi2_RV: {chi2_rv:.2f}")
    print(f"  chi2_total: {chi2_total:.2f}  dof: {dof}  reduced chi2: {chi2_reduced:.3f}")
    print(f"  AIC: {AIC:.2f}  BIC: {BIC:.2f}")

    with open(f'{out_prefix}_statistical_info.txt', 'w') as f:
        f.write("STATISTICAL INFORMATION CRITERIA\n" + "=" * 50 + "\n")
        f.write(f"chi2_total: {chi2_total:.2f}\n")
        f.write(f"Reduced chi2: {chi2_reduced:.3f}\n")
        f.write(f"Degrees of freedom: {dof}\n")
        f.write(f"AIC: {AIC:.2f}\nBIC: {BIC:.2f}\n")
        f.write("\nBest-fit parameters (maximum likelihood):\n")
        for name, val in zip(mcmc.param_names, best_fit_params):
            f.write(f"  {name}: {val:.6g}\n")
        try:
            tau_all = sampler.get_autocorr_time(quiet=True)
            f.write("\nAutocorrelation times:\n")
            for name, tau in zip(mcmc.param_names, tau_all):
                f.write(f"  {name}: {tau:.1f} steps\n")
        except Exception:
            f.write("\nAutocorrelation time estimation failed.\n")

    print(f"Saved statistics to '{out_prefix}_statistical_info.txt'")
    return dict(chi2_total=chi2_total, chi2_reduced=chi2_reduced, dof=dof, AIC=AIC, BIC=BIC)


def corner_plot(mcmc, flat_samples, truths, title, out_prefix="snope"):
    fig = corner.corner(flat_samples, labels=mcmc.param_names, truths=truths,
                         show_titles=True, title_fmt='.4f',
                         quantiles=[0.16, 0.84], title_quantiles=[0.16, 0.5, 0.84])
    plt.suptitle(title, fontsize=16)
    plt.savefig(f'{out_prefix}_corner.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved corner plot to '{out_prefix}_corner.png'")


def convergence_plot(mcmc, samples, title, out_prefix="snope"):
    n = mcmc.n_params
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for i, (ax, name) in enumerate(zip(axes, mcmc.param_names)):
        ax.plot(samples[:, :, i], 'k-', alpha=0.3)
        ax.set_ylabel(name, fontsize=12)
        ax.grid(True, alpha=0.3)
        if i == n - 1:
            ax.set_xlabel('Step Number', fontsize=12)
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{out_prefix}_convergence.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved convergence plot to '{out_prefix}_convergence.png'")


def orbit_plot(model, best_fit, title, out_prefix="snope", n_points=20000):
    kwargs = {name: v['median'] for name, v in best_fit.items()}
    model.set_parameters(n_points=n_points, verbose=True, **kwargs)

    astro = model.compute_observables(model.data['t_pos'])
    vel = model.compute_observables(model.data['t_rv'])
    t_high = np.linspace(model.t_min, model.t_max, 1000)
    high = model.compute_observables(t_high)

    plt.style.use('default')
    fig = plt.figure(figsize=(18, 6))

    ax1 = plt.subplot(141)
    ax1.plot(high['ra'], high['dec'], 'b-', linewidth=2, alpha=0.8, label='Model')
    ax1.plot(astro['ra'], astro['dec'], 'bo', markersize=5, alpha=0.7, label='Model @ obs times')
    ax1.errorbar(model.data['alpha_obs'], model.data['delta_obs'],
                 xerr=model.data['alpha_err'], yerr=model.data['delta_err'],
                 fmt='ro', markersize=5, alpha=0.7, capsize=3, label='Data')
    ax1.invert_xaxis()
    ax1.set_xlabel('RA (arcsec)')
    ax1.set_ylabel('Dec (arcsec)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.set_title(f'S2 Orbit - {title}')

    ax2 = plt.subplot(142)
    ax2.plot(t_high, high['rv'], 'b-', linewidth=2, alpha=0.8, label='Model')
    ax2.plot(model.data['t_rv'], vel['rv'], 'bo', markersize=5, alpha=0.7, label='Model @ obs times')
    ax2.errorbar(model.data['t_rv'], model.data['vlos_obs'], yerr=model.data['vlos_err'],
                 fmt='ro', markersize=5, alpha=0.7, capsize=3, label='Data')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Radial Velocity (km/s)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Radial Velocity Curve')

    ax3 = plt.subplot(143)
    ax3.plot(t_high, high['ra'], 'b-', linewidth=2, alpha=0.8, label='Model')
    ax3.plot(model.data['t_pos'], astro['ra'], 'bo', markersize=5, alpha=0.7, label='Model @ obs times')
    ax3.errorbar(model.data['t_pos'], model.data['alpha_obs'], yerr=model.data['alpha_err'],
                 fmt='ro', markersize=5, alpha=0.7, capsize=3, label='Data')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('RA (arcsec)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_title('RA vs Time')

    ax4 = plt.subplot(144)
    ax4.plot(t_high, high['dec'], 'b-', linewidth=2, alpha=0.8, label='Model')
    ax4.plot(model.data['t_pos'], astro['dec'], 'bo', markersize=5, alpha=0.7, label='Model @ obs times')
    ax4.errorbar(model.data['t_pos'], model.data['delta_obs'], yerr=model.data['delta_err'],
                 fmt='ro', markersize=5, alpha=0.7, capsize=3, label='Data')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Dec (arcsec)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Dec vs Time')

    plt.tight_layout()
    plt.savefig(f'{out_prefix}_orbit.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved orbit plot to '{out_prefix}_orbit.png'")


def effective_potential_plot(preview, title, out_prefix=None, show=True):
    """Plot V_eff(r) against E^2 from `S2OrbitModel.preview_effective_potential`.

    Two crossings of V_eff and the E^2 line -- at r_p and r_a -- mean the
    chosen metric and parameters give a bound orbit. If the curves don't
    cross twice inside the plotted range, the orbit isn't bound as
    configured, and the full integrator will fail or return nonsense for
    these parameters.
    """
    r, V, E2 = preview['r'], preview['V_eff'], preview['E2']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r, V, color='C0', linewidth=2, label=r'$V_{\rm eff}(r)$')
    ax.axhline(E2, color='C3', linestyle='--', linewidth=1.5, label=r'$E^2$')
    ax.axvline(preview['r_p'], color='gray', linestyle=':', linewidth=1, label='periapsis / apoapsis')
    ax.axvline(preview['r_a'], color='gray', linestyle=':', linewidth=1)
    ax.axvline(preview['rh'], color='k', linestyle='-', linewidth=1, alpha=0.5, label='horizon')

    bound = np.any(V > E2) and (V[0] < E2 or V[-1] < E2)
    status = "bound orbit (turning points found)" if bound else "check parameters -- no clear turning points"
    ax.set_title(f'Effective potential - {title}\n{status}')
    ax.set_xlabel(r'$r$  (units of $R_s$)')
    ax.set_ylabel(r'$V_{\rm eff}(r)$, $E^2$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if out_prefix:
        plt.savefig(f'{out_prefix}_effective_potential.png', dpi=200, bbox_inches='tight')
        print(f"Saved effective-potential plot to '{out_prefix}_effective_potential.png'")
    if show:
        plt.show()
    else:
        plt.close(fig)
