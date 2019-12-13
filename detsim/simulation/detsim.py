#
#   DetSim Functionality: From MC-hits to raw-waveforms (adc)
#
#   GM, JAH, JR, GD Dec 2019


import time

import numpy             as np
import scipy             as sc
import scipy.stats       as st
import tables            as tb

from typing    import Callable
from typing    import Tuple
from typing    import List

import invisible_cities.core    .system_of_units_c as system_of_units
import invisible_cities.core    .fit_functions     as fitf
import invisible_cities.database.load_db           as db

import matplotlib.pyplot as plt

units = system_of_units.SystemOfUnits()

#
# Utilities
#----------------

def bincounter(xs, dx = 1., x0 = 0.):
    ixs    = ((xs + x0) // dx).astype(int)
    return np.unique(ixs, return_counts=True)


def bincounterdd(xxs, dxs = 1., x0s = 0., n = 1000):

    xs = np.array(xxs, dtype = float).T
    dxs = np.ones_like(xs) * dxs
    x0s = np.ones_like(xs) * x0s

    ixs = ((xs - x0s) // dxs).astype(int)
    ids, ccs =  np.unique(ixs, axis = 0, return_counts = True)
    return ids.T, ccs


#
# Configuration
#-------------------


class Singleton(object):

    def __new__(cls, *args, **kargs):

        if not hasattr(cls, '_instance'):
            orig = super(Singleton, cls)
            cls._instance = orig.__new__(cls, *args, **kargs)
        return cls._instance


class DetSimParameters(Singleton):
    """ Singleton with the configuration and NEXT detector parameters
    control parameters are: pmt and SiPM wf time bins,
    and re-voxelization size (mm) of the diffused electrons
    """

    def __init__(self, detector = 'new', run_number = -1):

        self.ws                     = 60.0 * units.eV
        self.wi                     = 22.4 * units.eV
        self.fano_factor            = 0.15
        self.conde_policarpo_factor = 0.10

        self.drift_velocity         = 1.00 * units.mm / units.mus
        self.lifetime               = 5.e3 * units.ms

        self.transverse_diffusion   = 1.00 * units.mm / units.cm**0.5
        self.longitudinal_diffusion = 0.30 * units.mm / units.cm**0.5
        self.voxel_sizes            = np.array((2.00 * units.mm, 2.00 * units.mm, 0.2 * units.mm), dtype = float)

        self.el_gain                = 1e3
        self.el_gain_sigma          = np.sqrt(self.el_gain * self.conde_policarpo_factor)

        self.wf_buffer_time         = 1000 * units.mus
        self.wf_pmt_bin_time        =   25 * units.ns
        self.wf_sipm_bin_time       =    1 * units.mus

        self.EP_z                   = 632 * units.mm
        self.EL_dz                  =  5  * units.mm
        self.TP_z                   = -self.EL_dz
        self.EL_dtime               =  self.EL_dz / self.drift_velocity
        self.EL_pmt_time_sample     =  self.wf_pmt_bin_time
        self.EL_sipm_time_sample    =  self.wf_sipm_bin_time

        if (detector == 'dummy'):
            self.dummy_detector()
        else:
            self.next_detector(detector, run_number)

    def dummy_detector(self):

        self.x_pmts   = np.zeros(self.npmts)
        self.y_pmts   = np.zeros(self.npmts)
        self.adc_to_pes_pmts = np.ones(self.npmts)

        self.sipm_pich = 10 * units.mm
        indices       = np.arange(-self.nsipms_side, self.nsipms_side + 1)
        self.sipms    = [(self.sipm_pitch * i, self.sipm_pitch * j) for i in indices for j in indices]
        self.nsipms   = len(self.sipms)

        self.nsipms   = len(self.sipms)

        self.x_sipms  = np.array([sipm[0] for sipm in self.sipms])
        self.y_sipms  = np.array([sipm[1] for sipm in self.sipms])
        self.adc_to_pes_sipms = np.ones(self.nsipms)


    def next_detector(self, detector = 'new', run_number = -1):

        db_pmts  = db.DataPMT (detector, run_number)
        self.x_pmts          = db_pmts['X']         .values
        self.y_pmts          = db_pmts['Y']         .values
        self.adc_to_pes_pmts = db_pmts['adc_to_pes'].values
        self.npmts           = len(self.x_pmts)

        db_sipms = db.DataSiPM(detector, run_number)
        self.x_sipms          = db_sipms['X']         .values
        self.y_sipms          = db_sipms['Y']         .values
        self.adc_to_pes_sipms = db_sipms['adc_to_pes'].values
        self.nsipms           = len(self.x_sipms)


    def configure(self, **kargs):
        for key, val in kargs.items():
            setattr(obj, key, val)

#        self.xybins   = vx.bins_edges(np.sort(detsimparams.x_sipms), 0.5 * sipm_pitch), vx.bins_edges(np.sort(detsimparams.y_sipms), 0.5 * sipm_pitch)

detsimparams = DetSimParameters()

#
#  Secondary electrons
#------------------------

def get_deposits(hits):

    xs   = hits['x']     .values
    ys   = hits['y']     .values
    zs   = hits['z']     .values
    enes = hits['energy'].values

    return xs, ys, zs, enes


def generate_deposits(x0 = 0., y0 = 0., z0 = 300. * units.mm,
                      xsigma = 1.00 * units.mm, size = 1853, dx = 1.00 * units.mm):
    """ generate energy deposits, dummy funciton
    """

    xs = np.random.normal(x0, xsigma, size)
    ys = np.random.normal(y0, xsigma, size)
    zs = np.random.normal(z0, xsigma, size)

    xpos, vnes = bincounterdd((xs, ys, zs), dx)
    vxs, vys, vzs = dx * xpos
    vnes = detsimparams.wi * vnes

    return (vxs, vys, vzs, vnes)


def generate_electrons(energies    : np.array,
                       wi          : float = detsimparams.wi,
                       fano_factor : float = detsimparams.fano_factor) -> np.array:
    """ generate number of electrons starting from energy deposits
    """
    nes  = np.array(energies/wi, dtype = int)
    pois = nes < 10
    nes[ pois] = np.random.poisson(nes[pois])
    nes[~pois] = np.round(np.random.normal(nes[~pois], np.sqrt(nes[~pois] * fano_factor)))
    return nes


def drift_electrons(zs             : np.array,
                    electrons      : np.array,
                    lifetime       : float = detsimparams.lifetime,
                    drift_velocity : float = detsimparams.drift_velocity) -> np.array:
    """ returns number of electrons due to lifetime loses from initial secondary electrons
    """
    ts  = zs / drift_velocity
    nes = np.copy(electrons -np.random.poisson(electrons * (1. - np.exp(-ts/lifetime))))
    nes[nes < 0] = 0
    return nes


def diffuse_electrons(xs                     : np.array,
                      ys                     : np.array,
                      zs                     : np.array,
                      electrons              : np.array,
                      transverse_diffusion   : float = detsimparams.transverse_diffusion,
                      longitudinal_diffusion : float = detsimparams.longitudinal_diffusion,
                      voxel_sizes            : np.array = detsimparams.voxel_sizes) -> Tuple:
    """
    starting from a voxelized electrons with positions xs, ys, zs, and number of electrons, electrons,
    apply diffusion and return voxelixed electrons with positions xs, ys, zs, an electrons
    the voxel_size arguement controls the size of the voxels for the diffused electrons
    """
    nes = electrons
    xs = np.repeat(xs, nes); ys = np.repeat(ys, nes); zs = np.repeat(zs, nes)
    #zzs = np.concatenate([np.array((zi,)*ni) for zi, ni in zip(zs, ns)])
    sqrtz = zs ** 0.5
    vxs  = np.random.normal(xs, sqrtz * transverse_diffusion)
    vys  = np.random.normal(ys, sqrtz * transverse_diffusion)
    vzs  = np.random.normal(zs, sqrtz * longitudinal_diffusion)
    vnes = np.ones(vxs.size)
    vpos, vnes = bincounterdd((vxs, vys, vzs), voxel_sizes)
    vxs, vys, vzs = voxel_sizes[:, np.newaxis] * vpos
    return (vxs, vys, vzs, vnes)

#
#  EL photons and pes at sensors
#---------------------------------

def _psf(dx, dy, dz, factor = 1.):
    """ generic analytic PSF function
    """
    return factor * np.abs(dz) / (2 * np.pi) / (dx**2 + dy**2 + dz**2)**1.5


psf_pmt  = lambda dx, dy : _psf(dx, dy, detsimparams.EP_z, factor = 25e3)

psf_sipm = lambda dx, dy : _psf(dx, dy, detsimparams.TP_z, factor = 10.)


def generate_EL_photons(electrons      : np.array,
                        el_gain        : float = detsimparams.el_gain,
                        el_gain_sigma  : float = detsimparams.el_gain_sigma):
    """ generate number of EL-photons produced by secondary electrons that reach
    the EL (after drift and diffusion)
    """
    nphs      = electrons * np.random.normal(el_gain, el_gain_sigma, size = electrons.size)
    return nphs


def estimate_pes_at_sensors(xs        : np.array,  # nelectrons
                            ys        : np.array,  # nelectrons
                            photons   : np.array,  # nelectrons
                            x_sensors : np.array,  # nsensors
                            y_sensors : np.array,  # nsensors
                            psf       : Callable) -> np.array:
    """ estimate the number of pes at sensors from the number of photons
    produced by electrons that reach the EL at xs, ys positions.
    It returns an array of pes with nelectrons x nsensors shape.
    """
    dxs     = xs[:, np.newaxis] - x_sensors
    dys     = ys[:, np.newaxis] - y_sensors
    photons = photons[:, np.newaxis] + np.zeros_like(x_sensors)
    pes     = photons * psf(dxs, dys)
    #pes = np.random.poisson(pes)
    return pes


def estimate_pes_at_pmts(xs      : np.array,
                         ys      : np.array,
                         photons : np.array) -> np.array:
    return estimate_pes_at_sensors(xs, ys, photons,
                                   detsimparams.x_pmts, detsimparams.y_pmts, psf_pmt)


def estimate_pes_at_sipms(xs      : np.array,
                          ys      : np.array,
                          photons : np.array) -> np.array:
    return estimate_pes_at_sensors(xs, ys, photons,
                                   detsimparams.x_sipms, detsimparams.y_sipms, psf_sipm)

#
#  WFs
#-----------


def create_wfs(ts             : np.array, # nelectrons
               pes            : np.array, # (nelectrons, nsensors)
               wf_buffer_time : float = detsimparams.wf_buffer_time,
               wf_bin_time    : float = detsimparams.wf_pmt_bin_time,
               el_time_sample : float = detsimparams.EL_pmt_time_sample,
               el_time        : float = detsimparams.EL_dtime,
               adc_to_pes     : np.array = detsimparams.adc_to_pes_pmts): # nsensors
    """ create the wfs starting drom the pes produced per electrons and each sensor
    the control parameters are the wf_bin_time and the el_time_sample,
    by default they are the same parameters.
    The number of pes photos per sensor are spread along the EL-time depending
    on the wf_bin_time
    Returns: time bins of the wf (nbins), and wfs contents (nbins, nsensors)
    """

    nsize = int(el_time        // el_time_sample)
    nbins = int(wf_buffer_time // wf_bin_time )

    wftimes = wf_bin_time    * np.arange(nbins)
    dts     = el_time_sample * np.arange(nsize)

    nsensors  = 1 if len(pes.shape) == 1 else int(pes.shape[1])
    wfs     = np.zeros((nbins, nsensors), dtype = int)

    def _wf(its, ipes, iwf):
        # for each sensor, check if has pes, sample pes in EL times and fill wf
        if (np.sum(ipes) <= 0): return iwf

        nts        = np.repeat(its, ipes.astype(int))
        sits, spes = bincounter(nts, wf_bin_time)

        spesn       = np.random.poisson(spes/nsize, size = (nsize, spes.size))
        for k, kpes in enumerate(spesn):
            kk =  int(k * el_time_sample / wf_bin_time)
            iwf[sits + kk] = iwf[sits + kk] + kpes
        return iwf

    [_wf(ts, ipes, iwf) for ipes, iwf in zip(pes.T, wfs.T)]

    return wftimes, wfs * adc_to_pes


create_wfs_pmts= create_wfs


def create_wfs_sipms(ts, pes,
                         wf_buffer_time : float = detsimparams.wf_buffer_time,
                         wf_bin_time    : float = detsimparams.wf_sipm_bin_time,
                         el_time_sample : float = detsimparams.EL_sipm_time_sample,
                         el_time        : float = detsimparams.EL_dtime,
                         adc_to_pes     : np.array = detsimparams.adc_to_pes_sipms):
    return create_wfs(ts, pes, wf_buffer_time, wf_bin_time,
                      el_time_sample, el_time, adc_to_pes)


#
# IC - function
#--------------------

def get_function_generate_wfs(detector         : str = 'new',
                              run_number       : int = -1,
                              wf_buffer_time   : float = detsimparams.wf_buffer_time,
                              wf_pmt_bin_time  : float = detsimparams.wf_pmt_bin_time,
                              wf_sipm_bin_time : float = detsimparams.wf_sipm_bin_time,
                              voxel_xy_size    : float = detsimparams.voxel_sizes[0],
                              voxel_z_size     : float = detsimparams.voxel_sizes[2]
                             ) -> Tuple[np.array, np.array]:

    dsim                  = DetSimParameters(detector, run_number)
    dsim.voxel_sizes      = np.array((voxel_xy_size, voxel_xy_size, voxel_z_size))
    dsim.wf_buffer_time   = wf_buffer_time
    dsim.wf_pmt_bin_time  = wf_pmt_bin_time
    dsim.wf_sipm_bin_time = wf_sipm_bin_time
    
    def _generate_wfs(hits):

        xs, ys, zs, enes       = get_deposits(hits)

        nes                    = generate_electrons(enes)
        nes                    = drift_electrons(zs, nes)
        dxs, dys, dzs, dnes    = diffuse_electrons(xs, ys, zs, nes)

        dts                    = dzs / dsim.drift_velocity
        photons                = generate_EL_photons(dnes)

        pes_pmts               = estimate_pes_at_pmts (dxs, dys, photons)
        pes_sipms              = estimate_pes_at_sipms(dxs, dys, photons)

        times_pmts, wfs_pmts   = create_wfs_pmts (dts, pes_pmts)
        times_sipms, wfs_sipms = create_wfs_sipms(dts, pes_sipms)

        return wfs_pmts, wfs_sipms

    return _generate_wfs
