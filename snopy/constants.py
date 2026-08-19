"""Physical constants and unit conversions, shared by every metric."""
from astropy.constants import G, c
import astropy.units as u

G_SI = G.value                       # m^3 kg^-1 s^-2
C_SI = c.value                       # m/s
SEC_PER_YEAR = (1 * u.yr).to(u.s).value
ARCSEC_PER_RAD = 206264.806
