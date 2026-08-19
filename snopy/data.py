"""Load S2's astrometry (RA/Dec offsets) and radial-velocity data.

Both files are plain, headerless CSVs:

    pos_file:  t[yr], alpha[mas], alpha_err[mas], delta[mas], delta_err[mas]
    rv_file:   t[yr], v_los[km/s], vlos_err[km/s]

(the public Gillessen et al. astrometry/RV tables for S2 are one example.)
"""
import pandas as pd


def load_orbit_data(pos_file, rv_file, verbose=True):
    try:
        pos = pd.read_csv(pos_file, header=None,
                           names=["t", "alpha", "alpha_err", "delta", "delta_err"])
        rv = pd.read_csv(rv_file, header=None,
                          names=["t", "v_los", "vlos_err"])
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Data files not found. Please ensure '{pos_file}' and '{rv_file}' exist."
        ) from exc

    data = {
        't_pos': pos.t.values,
        'alpha_obs': pos.alpha.values / 1000.0,   # mas -> arcsec
        'delta_obs': pos.delta.values / 1000.0,
        'alpha_err': pos.alpha_err.values / 1000.0,
        'delta_err': pos.delta_err.values / 1000.0,
        't_rv': rv.t.values,
        'vlos_obs': rv.v_los.values,               # km/s
        'vlos_err': rv.vlos_err.values,
    }

    if verbose:
        print(f"Loaded {len(data['t_pos'])} positional data points")
        print(f"Loaded {len(data['t_rv'])} radial velocity data points")
        print(f"Time range - Positions: {data['t_pos'].min():.2f} to {data['t_pos'].max():.2f} years")
        print(f"Time range - RV: {data['t_rv'].min():.2f} to {data['t_rv'].max():.2f} years")
        print(f"RA range: {data['alpha_obs'].min():.3f} to {data['alpha_obs'].max():.3f} arcsec")
        print(f"Dec range: {data['delta_obs'].min():.3f} to {data['delta_obs'].max():.3f} arcsec")
        print(f"RV range: {data['vlos_obs'].min():.1f} to {data['vlos_obs'].max():.1f} km/s")

    return data
