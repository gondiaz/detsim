import numpy as np

from numpy.polynomial.polynomial import polyval

from functools import  partial

from typing    import Callable
from typing    import    Tuple

from invisible_cities.core.system_of_units_c import   units
from invisible_cities.database               import load_db as DB


def WS_() -> float:
    return 39.2 * units.eV


def scintillation_time(x        : np.ndarray,
                       fast_amp :      float,
                       fast_tau :      float,
                       slow_amp :      float,
                       slow_tau :      float) -> float:
    """
    Scintillation relaxation time according to
    fast and slow components with tau in ns
    """

    return fast_amp * np.exp(-x / fast_tau) + slow_amp * np.exp(-x / slow_tau)


def xenon_scintillation_time(time_range : np.ndarray = np.arange(500)) -> Callable:
    xenon_params = 0.1, 4.5, 0.9, 100
    raw_dist     = scintillation_time(time_range, *xenon_params)
    return partial(np.random.choice            ,
                   a = time_range              ,
                   p = raw_dist / sum(raw_dist))


def photon_times(hit_times : np.ndarray,
                 sampler   : Callable = xenon_scintillation_time()) -> np.ndarray:
    return hit_times + sampler(size = len(hit_times))


def num_photons(hit_energies : np.ndarray) -> np.ndarray:
    n_ws = hit_energies / WS_()
    return np.random.poisson(n_ws)


## Will need to be generalised
def relative_coordinates(detector_db : str,
                         run_number  : int) -> Callable:
    pmt_xyphi = DB.DataPMT(detector_db, run_number)[['X', 'Y']]
    pmt_xyphi['phi'] = np.arctan2(pmt_xyphi.loc[:, 'Y'],
                                  pmt_xyphi.loc[:, 'X'])

    def get_relative_coords(x : np.ndarray,
                            y : np.ndarray) -> Tuple:

        rel_r   = np.sqrt((x - pmt_xyphi.X)**2 + (y - pmt_xyphi.Y)**2)

        rel_phi = np.abs(np.arctan2(y, x) - pmt_xyphi.phi)
        rel_phi[rel_phi > np.pi] = 2 * np.pi - rel_phi
        return rel_r, rel_phi
    return get_relative_coords


def scint_prob(r      :      float,
               z      :      float,
               params : np.ndarray) -> float:
    """
    Returns the probability of detecting a
    photon given a relative r and z position
    and parameters from the parametrization
    appropriate for the sensor being considered

    r      : straight line distance between deposit
             and sensor in X,Y plane
    z      : z position of deposit
    params : num z powers x num. r powers array
             polyval expects columns of parameters
    """
    return polyval(r, polyval(z, params))


def s1_detection_probability(n_bins_phi :        int,
                             parameters : np.ndarray) -> Callable:

    phi_bins = np.linspace(0, np.pi, n_bins_phi)
    def probability(r    : np.ndarray,
                    phi  : np.ndarray,
                    z    : np.ndarray,
                    ring : np.ndarray) -> np.ndarray:
        """
        Returns the probability of detection
        by each of the sensors in the order
        of the lists given.

        r    : np.ndarray of floats
               relative radius for the sensors in order
        phi  : np.ndarray of floats
               relative azimuthal angle between point
               and each sensor
        z    : np.ndarray of floats
               Z position of deposition
        ring : np.ndarray of ints
               ring in the plane 0-1 for new

        returns
            np.ndarray of floats
            probability of detecting a photon at each sensor
        """
        bin_gen    = (np.argmax(p < phibins) - 1 for p in phi)
        phi_bin    = np.fromiter(bin_gen, np.int)

        prob_tuple = map(scint_prob, r, z, parameters[ring, phi_bin])

        return np.array(prob_tuple).clip(0)
    return probability
