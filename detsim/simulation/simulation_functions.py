#
# Authors: G. Martinez, J.A. Hernando


import time

import numpy             as np
import scipy             as sc
import scipy.stats       as st
import tables            as tb

from typing    import    Tuple
from typing    import     List
from typing    import Callable
from typing    import  Mapping


import invisible_cities.core    .system_of_units_c as system_of_units
#import invisible_cities.core    .fit_functions     as fitf
#import invisible_cities.database.load_db           as db

units = system_of_units.SystemOfUnits()

FANO_FACTOR    = 0.15                            # Fano Factor
DRIFT_VELOCITY = 1.00 * units.mm / units.mus     # default drift velocity
W_I            = 22.4 * units.eV                 # energy per secondary electrons
W_S            = 60.0 * units.eV                 # energy  per scintillation photon
LIFETIME       = 8.e3 * units.mus                # default lifetime value


def secondary_electrons_generate(deposits        : Tuple,
                                 drift_velocity  : float = DRIFT_VELOCITY,
                                 wi              : float = W_I,
                                 fano_factor     : float = NANO_FACTOR) -> Tuple:
    """
    """
    xs, ys, zs, es = deposits
    n_ie        = np.zeros_like(es/wi, dtype=int)
    lowE        = mean <= 10
    n_ie[ lowE] = np.random.poisson(mean[lowE])
    n_ie[~lowE] = np.round(np.random.normal(mean[~lowE], np.sqrt(mean[~lowE] * fano_factor)))
    ts          = zs/drift_velocity
    return (xs, ys, ts, n_ie)


def secondary_electrons_drift(secondary_electrons : Tuple,
                              lifetime            : float) -> Tuple:
    xs, ys, ts, nes = secondary_electrons
    def attach(x, y):
        return np.count_nonzero(-lifetime * np.log(np.random.rand(x)) > y)
    nes_ = np.array(list(map(attach, nes, tes))
    return (xs, ys, ts, nes_)


def secondary_electrons_diffusion(secondary_electrons: Tuple,
                                  l
)
# def simulate_diffusion(x, y, z):
#     sqrtz = z**0.5
#
#     x = np.random.normal(x, NEXT.  transverse_diffusion * sqrtz)
#     y = np.random.normal(y, NEXT.  transverse_diffusion * sqrtz)
#     z = np.random.normal(z, NEXT.longitudinal_diffusion * sqrtz)
#     return x, y, np.clip(z, 0, NEXT.max_drift_length)
#
#
# def create_ionization_electrons(x, y, z, E):
#     mean_n_ie = E / NEXT.w_i
#     n_ie      = poisson_fluctuate_fano(mean_n_ie)
#     n_ie      = simulate_attachment(n_ie, z)
#
#     x = np.repeat(x, n_ie)
#     y = np.repeat(y, n_ie)
#     z = np.repeat(z, n_ie)
#
#     x, y, z = simulate_diffusion(x, y, z)
#     return x, y, z / NEXT.drift_velocity
