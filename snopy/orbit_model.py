"""The relativistic S2 orbit model -- one engine, any metric.

Works for any diagonal, static, equatorial metric
ds^2 = g_tt(r) dt^2 + g_rr(r) dr^2 + r^2 dphi^2, described by a
`MetricSpec` (see snope/metrics/__init__.py). Only the metric functions
change between Schwarzschild-de Sitter, quadratic gravity, Brans-Dicke,
and so on -- geodesic integration, the Roemer delay, relativistic
Doppler + gravitational redshift, and sky projection are the same for
all of them, so they live here exactly once. Every metric gets the same
high-fidelity integrator (DOP853, rtol=atol=1e-12), with no shortcuts.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import astropy.units as u
from scipy.interpolate import interp1d

from .constants import G_SI, C_SI, SEC_PER_YEAR, ARCSEC_PER_RAD
from .data import load_orbit_data

STANDARD_PARAM_NAMES = ('M_bh', 'distance', 'a', 'e', 'i', 'omega', 'Omega',
                         't_peri', 'x0', 'y0', 'vx0', 'vy0', 'vz0')


class S2OrbitModel:
    """Relativistic S2 orbit model parametrized by a pluggable `MetricSpec`."""

    def __init__(self, metric, data_path_pos="tab_gillessen_pos.csv",
                 data_path_rv="tab_gillessen_vr.csv", verbose=True):
        self.metric = metric
        self.G = G_SI
        self.c = C_SI
        self.sec_per_year = SEC_PER_YEAR
        self.arcsec_per_rad = ARCSEC_PER_RAD
        self.verbose = verbose

        self.data = load_orbit_data(data_path_pos, data_path_rv, verbose=verbose)

        t_all = np.concatenate([self.data['t_pos'], self.data['t_rv']])
        self.t_min = float(np.min(t_all) - 2)
        self.t_max = float(np.max(t_all) + 2)

        self.R_s = None
        self.interp_functions_set = False

    @property
    def param_names(self):
        return STANDARD_PARAM_NAMES + tuple(self.metric.extra_param_names)

    def to_geometric(self, value):
        return value / self.R_s

    def from_geometric(self, value):
        return value * self.R_s

    def set_parameters(self, M_bh=4.1e6, distance=8.1, a=125.0, e=0.884, i=134.18,
                        omega=66.1, Omega=228.07, t_peri=2002.32,
                        x0=0.0, y0=0.0, vx0=0.0, vy0=0.0, vz0=0.0,
                        n_points=1000, verbose=None, **extra_params):
        """`a` is the semi-major axis in milliarcseconds. Any metric-specific
        parameters (e.g. k=..., log10_lambda=..., omega_bd=...) are passed
        as keyword arguments and forwarded straight through to the metric
        functions."""
        if verbose is None:
            verbose = self.verbose

        missing = set(self.metric.extra_param_names) - set(extra_params)
        if missing:
            raise TypeError(f"Missing required metric parameter(s): {sorted(missing)}")
        self.extra = {name: extra_params[name] for name in self.metric.extra_param_names}

        self.M_bh_kg = (M_bh * u.M_sun).to(u.kg).value
        self.R_s = 2 * self.G * self.M_bh_kg / self.c ** 2
        self.distance_m = (distance * u.kpc).to(u.m).value

        self.a_mas = a
        a_arcsec = a / 1000.0
        distance_pc = distance * 1000
        self.a_au = a_arcsec * distance_pc
        self.a_m = self.a_au * u.au.to(u.m)
        self.a_arcsec = a_arcsec

        self.e = e
        self.i = np.radians(i)
        self.omega = np.radians(omega)
        self.Omega = np.radians(Omega)
        self.t_peri = t_peri
        self.x0, self.y0 = x0, y0
        self.vx0, self.vy0, self.vz0 = vx0, vy0, vz0

        self.rh = self.metric.horizon_radius(self.R_s, **self.extra)
        self.M_kepler = self.metric.keplerian_mass_guess(self.R_s, **self.extra)

        if verbose:
            print(f"Semi-major axis: {a} mas = {self.a_arcsec:.3f} arcsec = {self.a_au:.1f} AU")
            print(f"Periastron: {t_peri}, Distance: {distance} kpc")
            if self.extra:
                print(f"Metric parameters: {self.extra}")

        self.integrate_orbit(self.t_min, self.t_max, n_points=n_points)

    # ------------------------------------------------------------------
    # Metric access (thin wrappers so the rest of the class is
    # completely metric-agnostic)
    # ------------------------------------------------------------------
    def gtt(self, r):
        return self.metric.gtt(r, self.R_s, **self.extra)

    def grr(self, r):
        return self.metric.grr(r, self.R_s, **self.extra)

    def dgtt_dr(self, r):
        return self.metric.dgtt_dr(r, self.R_s, **self.extra)

    def dgrr_dr(self, r):
        return self.metric.dgrr_dr(r, self.R_s, **self.extra)

    def V_eff(self, r, L):
        """Effective potential for the radial turning points:
        E^2 = V_eff(r) = -g_tt(r) * (1 + L^2/r^2). This only depends on
        g_tt, not g_rr -- true for any diagonal static metric of this
        form -- so this (and find_EL below) works unchanged for every
        metric."""
        return -self.gtt(r) * (1 + L ** 2 / r ** 2)

    def find_EL(self, a_geo, e):
        r_p = a_geo * (1 - e)
        r_a = a_geo * (1 + e)

        M_guess = self.M_kepler
        keplerian_T = np.sqrt(4 * np.pi ** 2 * a_geo ** 3 / M_guess)
        keplerian_u_phi_0 = 2 * np.pi / keplerian_T * a_geo ** 2 / r_p ** 2 * np.sqrt(1 - e ** 2)
        gtt_p = self.gtt(r_p)
        keplerian_u_t_0 = np.sqrt((1 + r_p ** 2 * keplerian_u_phi_0 ** 2) / (-gtt_p))
        E_guess = -gtt_p * keplerian_u_t_0
        L_guess = r_p ** 2 * keplerian_u_phi_0

        def eq(EL):
            E, L = EL
            return [E ** 2 - self.V_eff(r_p, L), E ** 2 - self.V_eff(r_a, L)]

        sol = root(eq, [E_guess, L_guess], tol=1e-12, method='lm')
        if not sol.success:
            if self.verbose:
                print("Warning: Root finding failed; using Keplerian guesses.")
            return [E_guess, L_guess]
        return sol.x

    def preview_effective_potential(self, M_bh=4.1e6, distance=8.1, a=125.0, e=0.884,
                                     n_r=500, r_range=None, **extra_params):
        """Sample V_eff(r) and the conserved E^2 without integrating the
        full orbit -- useful for checking, in a fraction of a second,
        whether a parameter combination gives a bound orbit (two turning
        points) before committing to a slow MCMC run.

        Returns a dict with r, V_eff, E2, r_p (periapsis), r_a (apoapsis),
        and rh (horizon radius), all in geometric units (r in R_s).
        """
        missing = set(self.metric.extra_param_names) - set(extra_params)
        if missing:
            raise TypeError(f"Missing required metric parameter(s): {sorted(missing)}")
        self.extra = {name: extra_params[name] for name in self.metric.extra_param_names}

        self.M_bh_kg = (M_bh * u.M_sun).to(u.kg).value
        self.R_s = 2 * self.G * self.M_bh_kg / self.c ** 2
        distance_pc = distance * 1000
        a_au = (a / 1000.0) * distance_pc
        a_m = a_au * u.au.to(u.m)

        self.rh = self.metric.horizon_radius(self.R_s, **self.extra)
        self.M_kepler = self.metric.keplerian_mass_guess(self.R_s, **self.extra)

        a_geo = self.to_geometric(a_m)
        E, L = self.find_EL(a_geo, e)

        r_p, r_a = a_geo * (1 - e), a_geo * (1 + e)
        if r_range is None:
            r_lo = max(1.01 * self.rh, 0.5 * r_p)
            r_hi = 1.5 * r_a
        else:
            r_lo, r_hi = r_range
        r = np.linspace(r_lo, r_hi, n_r)
        V = np.array([self.V_eff(ri, L) for ri in r])

        return {'r': r, 'V_eff': V, 'E2': E ** 2, 'E': E, 'L': L,
                'r_p': r_p, 'r_a': r_a, 'rh': self.rh}

    def geodesic_derivatives(self, tau, y):
        """Equatorial geodesic equations for a general diagonal metric
        ds^2 = g_tt(r) dt^2 + g_rr(r) dr^2 + r^2 dphi^2 -- valid for any
        g_tt(r), g_rr(r), including cases where g_rr != -1/g_tt. These
        are the standard Christoffel symbols for a diagonal metric that
        only depends on r (from Gamma^rho_mu nu = 1/2 g^rho rho (d_mu
        g_rho nu + d_nu g_rho mu - d_rho g_mu nu)), and they collapse to
        the textbook Schwarzschild/SdS Christoffels when g_rr = 1/(-g_tt):

            Gamma^t_tr     =  g_tt'/(2 g_tt)
            Gamma^r_tt     = -g_tt'/(2 g_rr)
            Gamma^r_rr     =  g_rr'/(2 g_rr)
            Gamma^r_phiphi = -r/g_rr
            Gamma^phi_rphi =  1/r
        """
        t, r, phi, ut, ur, uphi = y

        if r <= 1.001 * self.rh:
            return [0, 0, 0, 0, 0, 0]

        g_tt = self.gtt(r)
        g_rr = self.grr(r)

        if g_tt >= 0 or not np.isfinite(g_tt) or not np.isfinite(g_rr) or g_rr <= 0:
            return [0, 0, 0, 0, 0, 0]

        dgtt = self.dgtt_dr(r)
        dgrr = self.dgrr_dr(r)

        Gamma_t_tr = 0.5 * dgtt / g_tt
        Gamma_r_tt = -0.5 * dgtt / g_rr
        Gamma_r_rr = 0.5 * dgrr / g_rr
        Gamma_r_phiphi = -r / g_rr
        Gamma_phi_rphi = 1 / r

        return [
            ut,
            ur,
            uphi,
            -2 * Gamma_t_tr * ut * ur,
            -(Gamma_r_tt * ut ** 2 + Gamma_r_rr * ur ** 2 + Gamma_r_phiphi * uphi ** 2),
            -2 * Gamma_phi_rphi * ur * uphi,
        ]

    def rotate_to_sky_frame(self, x_orb, y_orb):
        A = (np.cos(self.omega) * np.cos(self.Omega) - np.sin(self.omega) * np.sin(self.Omega) * np.cos(self.i))
        B = (np.cos(self.omega) * np.sin(self.Omega) + np.sin(self.omega) * np.cos(self.Omega) * np.cos(self.i))
        F = (-np.sin(self.omega) * np.cos(self.Omega) - np.cos(self.omega) * np.sin(self.Omega) * np.cos(self.i))
        Gc = (-np.sin(self.omega) * np.sin(self.Omega) + np.cos(self.omega) * np.cos(self.Omega) * np.cos(self.i))

        x_sky = A * x_orb + F * y_orb
        y_sky = B * x_orb + Gc * y_orb
        z_sky = -(np.sin(self.omega) * np.sin(self.i) * x_orb + np.cos(self.omega) * np.sin(self.i) * y_orb)
        return x_sky, y_sky, z_sky

    def integrate_orbit(self, t_min, t_max, n_points=1000):
        a_geo = self.to_geometric(self.a_m)
        E, L = self.find_EL(a_geo, self.e)
        if self.verbose:
            print(f"Energy: {E:.8f}, Angular Momentum: {L:.8f}")

        r0 = a_geo * (1 - self.e)
        gtt_r0 = self.gtt(r0)
        u_t = -E / gtt_r0
        u_phi = L / r0 ** 2
        u_r = 0.0

        tau_max_fwd = self.to_geometric((t_max - self.t_peri) * self.sec_per_year * self.c)
        tau_max_bwd = self.to_geometric((self.t_peri - t_min) * self.sec_per_year * self.c)
        max_step = min(tau_max_fwd, tau_max_bwd) / n_points

        def rhs(tau, y):
            return self.geodesic_derivatives(tau, y)

        sol_fwd = solve_ivp(rhs, [0, tau_max_fwd], [0, r0, 0, u_t, u_r, u_phi],
                             method='DOP853', rtol=1e-12, atol=1e-12,
                             max_step=max_step, dense_output=True)
        sol_bwd = solve_ivp(rhs, [0, -tau_max_bwd], [0, r0, 0, u_t, u_r, u_phi],
                             method='DOP853', rtol=1e-12, atol=1e-12,
                             max_step=max_step, dense_output=True)

        n_half = n_points // 2
        tau_fwd = np.linspace(0, tau_max_fwd, n_half)
        tau_bwd = np.linspace(-tau_max_bwd, 0, n_half, endpoint=False)

        y_fwd = sol_fwd.sol(tau_fwd)
        y_bwd = sol_bwd.sol(tau_bwd)

        t_geo = np.concatenate([y_bwd[0][::-1], y_fwd[0]])
        r = np.concatenate([y_bwd[1][::-1], y_fwd[1]])
        phi = np.concatenate([y_bwd[2][::-1], y_fwd[2]])
        u_t = np.concatenate([y_bwd[3][::-1], y_fwd[3]])
        u_r = np.concatenate([y_bwd[4][::-1], y_fwd[4]])
        u_phi = np.concatenate([y_bwd[5][::-1], y_fwd[5]])

        t_phys = self.from_geometric(t_geo) / self.c
        t_years = self.t_peri + t_phys / self.sec_per_year

        uniq_idx = np.unique(t_years, return_index=True)[1]
        t_years_u = t_years[uniq_idx]
        order = np.argsort(t_years_u)
        idx = uniq_idx[order]

        t_years, r, phi, u_t, u_r, u_phi = (arr[idx] for arr in (t_years, r, phi, u_t, u_r, u_phi))

        if not np.all(np.diff(t_years) > 0):
            raise ValueError("Time array is not strictly increasing after processing.")

        x_orb = r * np.cos(phi)
        y_orb = r * np.sin(phi)
        x_sky, y_sky, z_sky = self.rotate_to_sky_frame(x_orb, y_orb)
        x_m = self.from_geometric(x_sky)
        y_m = self.from_geometric(y_sky)
        z_m = self.from_geometric(z_sky)

        v_r = u_r / u_t
        v_phi = (r * u_phi) / u_t
        v_x_orb = v_r * np.cos(phi) - v_phi * np.sin(phi)
        v_y_orb = v_r * np.sin(phi) + v_phi * np.cos(phi)
        v_x_sky, v_y_sky, v_z_sky = self.rotate_to_sky_frame(v_x_orb, v_y_orb)

        vx = v_x_sky * self.c
        vy = v_y_sky * self.c
        vz = v_z_sky * self.c

        if self.verbose:
            v_mag = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
            print(f"Max velocity magnitude: {np.max(v_mag) / 1000:.2f} km/s "
                  f"(expected ~7000-8000 km/s at periastron)")

        self.t_interp = t_years
        try:
            self.x_interp = interp1d(t_years, x_m, kind='cubic', bounds_error=False, fill_value='extrapolate')
            self.y_interp = interp1d(t_years, y_m, kind='cubic', bounds_error=False, fill_value='extrapolate')
            self.z_interp = interp1d(t_years, z_m, kind='cubic', bounds_error=False, fill_value='extrapolate')
            self.vx_interp = interp1d(t_years, vx, kind='cubic', bounds_error=False, fill_value='extrapolate')
            self.vy_interp = interp1d(t_years, vy, kind='cubic', bounds_error=False, fill_value='extrapolate')
            self.vz_interp = interp1d(t_years, vz, kind='cubic', bounds_error=False, fill_value='extrapolate')
            self.interp_functions_set = True
        except ValueError as e:
            print(f"Interpolation setup failed: {e}")
            raise

        self.E, self.L = E, L
        return t_years, x_m, y_m, z_m, vx, vy, vz, E, L

    def compute_observables(self, observation_times):
        if not self.interp_functions_set:
            raise ValueError("Interpolation functions not set. Call set_parameters first.")

        results = {'ra': [], 'dec': [], 'rv': [], 'roemer': []}

        for t_obs in observation_times:
            t_emit = t_obs
            for _ in range(5):
                z = self.z_interp(t_emit)
                roemer_delay_sec = z / self.c
                t_emit = t_obs - roemer_delay_sec / self.sec_per_year

            x = self.x_interp(t_emit)
            y = self.y_interp(t_emit)
            z = self.z_interp(t_emit)
            vx = self.vx_interp(t_emit)
            vy = self.vy_interp(t_emit)
            vz = self.vz_interp(t_emit)

            RA = np.arctan2(y, self.distance_m - z)
            Dec = np.arctan2(x, np.sqrt((self.distance_m - z) ** 2 + y ** 2))
            RA_arcsec = np.degrees(RA) * 3600 + self.x0
            Dec_arcsec = np.degrees(Dec) * 3600 + self.y0

            dt_yrs = t_obs - self.t_peri
            RA_arcsec += self.vx0 * dt_yrs
            Dec_arcsec += self.vy0 * dt_yrs

            r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            beta2 = (vx ** 2 + vy ** 2 + vz ** 2) / self.c ** 2
            beta2 = min(max(float(beta2), 0.0), 0.999999)
            beta_z = vz / self.c

            r_geo = self.to_geometric(r)
            g_tt_r = self.gtt(r_geo)
            # z_total = (1+z_grav)(1+z_doppler) - 1, metric-agnostic since it
            # only needs g_tt(r) at the emission point.
            gravitational_factor = 1.0 / np.sqrt(-g_tt_r)
            doppler_factor = np.sqrt(1 - beta2) / (1 - beta_z)
            z_total = gravitational_factor * doppler_factor - 1

            Vz = z_total * self.c / 1000.0 + self.vz0

            results['ra'].append(RA_arcsec)
            results['dec'].append(Dec_arcsec)
            results['rv'].append(Vz)
            results['roemer'].append(roemer_delay_sec)

        return results
