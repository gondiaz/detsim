#
#   DetSim Functionality: From MC-hits to raw-waveforms (adc)
#
#   GM, JAH, JR, GD Dec 2019


import time

import numpy             as np
import scipy             as sc
import scipy.stats       as st
import tables            as tb

from functools import partial

from typing    import Callable
from typing    import Tuple
from typing    import List

import invisible_cities.core    .system_of_units_c as system_of_units
import invisible_cities.core    .fit_functions     as fitf
import invisible_cities.database.load_db           as db

from invisible_cities.reco.corrections_new import read_maps

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

def _xybins(xxs, yys):
    def _bins(xxs):
        xs = np.sort(np.unique(xxs))
        dx   = 0.5*(xs[1] - xs[0])
        bins = np.append(xs - dx, xs[-1] + dx)
        return bins
    return _bins(xxs), _bins(yys)


def _xyfun(xaxis, yaxis, xymap, norma = 1.):

    xmin, xmax, dx = xaxis
    ymin, ymax, dy = yaxis

    xmax = xmax - 0.1 * dx # to avoid problems with clip in xmax
    ymax = ymax - 0.1 * dy

    xymap  = np.array(xymap) / norma

    def _fun(x, y):
        x = np.clip(x, xmin, xmax)
        y = np.clip(y, ymin, ymax)
        ix = ((x  - xmin) // dx).astype(int)
        iy = ((y  - ymin) // dy).astype(int)
        return xymap[ix, iy]

    return _fun


#
#  PSFs
#---------------------------------

def _psf(dx, dy, dz, factor = 1.):
    """ generic analytic PSF function
    """
    return factor * np.abs(dz) / (2 * np.pi) / (dx**2 + dy**2 + dz**2)**1.5


def get_psf_pmt_from_krmap(filename, factor = 1.):

    maps = read_maps(filename)

    xmin = maps.mapinfo.xmin
    xmax = maps.mapinfo.xmax
    ymin = maps.mapinfo.ymin
    ymax = maps.mapinfo.ymax
    nx   = maps.mapinfo.nx
    ny   = maps.mapinfo.ny

    dx   = (xmax - xmin)/ float(nx)
    dy   = (ymax - ymin)/ float(ny)

    xymap  = factor * np.nan_to_num(np.array(maps.e0), 0.)

    return _xyfun((xmin, xmax, dx), (ymin, ymax, dy), xymap)


def get_psf_sipm_from_file(filename, factor = 1.):

    h5file = tb.open_file(filename)
    dat = h5file.root.PSF.PSFs.read()
    h5file.close()

    xr, yr, zr, fr = dat['xr'], dat['yr'], dat['z'], dat['factor']
    sel = (zr == 12.5)
    xs, ys, fs = xr[sel], yr[sel], fr[sel]

    xybins   = _xybins(xs, ys)
    xymap, _ = np.histogramdd((xs, ys), bins = xybins, weights = fs)
    xyaxis   = [(np.min(ixs), np.max(ixs), ixs[1] - ixs[0]) for ixs in xybins]

    return _xyfun(*xyaxis, factor * xymap)


#
# Configuration
#-------------------


class DetSimParameters:
    """ parameters for detsim
    """

    def __init__(self, detector = 'new', run_number = -1,
                      krmap_filename = '', psfsipm_filename = '', **kargs):

        self.ws                     = 60.0 * units.eV
        self.wi                     = 22.4 * units.eV
        self.fano_factor            = 0.15
        self.conde_policarpo_factor = 0.10

        self.drift_velocity         = 1.00 * units.mm / units.mus
        self.lifetime               = 8.00 * units.ms

        self.transverse_diffusion   = 1.00 * units.mm / units.cm**0.5
        self.longitudinal_diffusion = 0.30 * units.mm / units.cm**0.5
        self.voxel_sizes            = np.array((2.00 * units.mm, 2.00 * units.mm, 0.2 * units.mm), dtype = float)

        self.el_gain                = 1e3

        self.wf_buffer_time         = 1300 * units.mus
        self.wf_pmt_bin_time        =   25 * units.ns
        self.wf_sipm_bin_time       =    1 * units.mus

        self.EP_z                   = 632 * units.mm
        self.EL_dz                  =  5  * units.mm
        self.TP_z                   = -self.EL_dz
        self.EL_dtime               =  self.EL_dz / self.drift_velocity
        self.EL_pmt_time_sample     =  self.wf_pmt_bin_time
        self.EL_sipm_time_sample    =  self.wf_sipm_bin_time

        self.s1_dtime               =  200 * units.ns

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

        ## TODO
        nphotons = (41.5 * units.keV / self.wi) * self.el_gain * self.npmts
        if (krmap_filename != ''):
            print('load psf pmt  from file : ', krmap_filename)
        if (psfsipm_filename != ''):
            print('load psf sipm from file : ', psfsipm_filename)
        psf_pmt  = partial(_psf, dz = self.EP_z, factor = 25e3)
        psf_sipm = partial(_psf, dz = self.TP_z, factor = 0.15)
        psf_s1   = partial(_psf, factor = 1e3)
        factor = 1./nphotons
        self.psf_pmt    = psf_pmt  if krmap_filename   == '' else get_psf_pmt_from_krmap(krmap_filename  , factor)
        self.psf_sipm   = psf_sipm if psfsipm_filename == '' else get_psf_sipm_from_file(psfsipm_filename, factor / 0.7e-4)
        self.psf_s1     = psf_s1

        self._configure(**kargs)
        self._update()


    def _configure(self, **kargs):
        for key, val in kargs.items():
            setattr(obj, key, val)


    def _update(self):

        self.el_gain_sigma          = np.sqrt(self.el_gain * self.conde_policarpo_factor)
        self.TP_z                   = -self.EL_dz
        self.EL_dtime               =  self.EL_dz / self.drift_velocity

        self.trigger_time           =  self.wf_buffer_time / 2

        self.wf_pmt_nbins           = int(self.wf_buffer_time // self.wf_pmt_bin_time)
        self.wf_sipm_nbins          = int(self.wf_buffer_time // self.wf_sipm_bin_time)

        self.s1_nsamples            = np.max((int(self.s1_dtime // self.wf_pmt_bin_time ), 1))
        self.s2_pmt_nsamples        = np.max((int(self.EL_dtime // self.wf_pmt_bin_time ), 1))
        self.s2_sipm_nsamples       = np.max((int(self.EL_dtime // self.wf_sipm_bin_time), 1))


dsim = DetSimParameters()


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
                      xsigma = 1.00 * units.mm, size = 1853,
                      dx = 1.00 * units.mm, wi = dsim.wi):
    """ generate energy deposits, dummy funciton
    """

    xs = np.random.normal(x0, xsigma, size)
    ys = np.random.normal(y0, xsigma, size)
    zs = np.random.normal(z0, xsigma, size)

    xpos, vnes = bincounterdd((xs, ys, zs), dx)
    vxs, vys, vzs = dx * xpos
    vnes = wi * vnes

    return (vxs, vys, vzs, vnes)


def generate_electrons(energies    : np.array,
                       wi          : float = dsim.wi,
                       fano_factor : float = dsim.fano_factor) -> np.array:
    """ generate number of electrons starting from energy deposits
    """
    nes  = np.array(energies/wi, dtype = int)
    pois = nes < 10
    nes[ pois] = np.random.poisson(nes[pois])
    nes[~pois] = np.round(np.random.normal(nes[~pois], np.sqrt(nes[~pois] * fano_factor)))
    return nes


def drift_electrons(zs             : np.array,
                    electrons      : np.array,
                    lifetime       : float = dsim.lifetime,
                    drift_velocity : float = dsim.drift_velocity) -> np.array:
    """ returns number of electrons due to lifetime loses from initial secondary electrons
    """
    ts  = zs / drift_velocity
    nes = electrons - np.random.poisson(electrons * (1. - np.exp(-ts/lifetime)))
    nes[nes < 0] = 0
    return nes


def diffuse_electrons(xs                     : np.array,
                      ys                     : np.array,
                      zs                     : np.array,
                      electrons              : np.array,
                      transverse_diffusion   : float = dsim.transverse_diffusion,
                      longitudinal_diffusion : float = dsim.longitudinal_diffusion,
                      voxel_sizes            : np.array = dsim.voxel_sizes) -> Tuple:
    """
    starting from a voxelized electrons with positions xs, ys, zs, and number of electrons, electrons,
    apply diffusion and return voxelixed electrons with positions xs, ys, zs, an electrons
    the voxel_size arguement controls the size of the voxels for the diffused electrons
    """
    nes = electrons.astype(int)
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

def generate_s1_photons(enes : np.array,
                        ws: float = dsim.ws) -> np.array:
    """ generate s1 photons,
    """
    return np.random.poisson(enes / ws)

def estimate_s1_pes(xs        : np.array,
                    ys        : np.array,
                    zs        : np.array,
                    photons   : np.array,
                    x_sensors : np.array = dsim.x_pmts,
                    y_sensors : np.array = dsim.y_pmts,
                    z_sensors : np.array = dsim.EP_z,
                    psf = dsim.psf_s1) -> np.array :
    """ Estimate pes at pmts for s1-photons given a psf
    """
    dxs = xs[:, np.newaxis] - x_sensors
    dys = ys[:, np.newaxis] - y_sensors
    dzs = zs[:, np.newaxis] - z_sensors
    photons = photons[:, np.newaxis] + np.zeros_like(x_sensors)
    pes = photons * psf(dxs, dys, dzs)
    pes = np.random.poisson(pes)
    return pes

def generate_s2_photons(electrons      : np.array,
                        el_gain        : float = dsim.el_gain,
                        el_gain_sigma  : float = dsim.el_gain_sigma) -> np.array:
    """ generate number of EL-photons produced by secondary electrons that reach
    the EL (after drift and diffusion)
    """
    nphs      = electrons * np.random.normal(el_gain, el_gain_sigma, size = electrons.size)
    return nphs


def estimate_s2_pes_at_sensors(xs        : np.array,  # nelectrons
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
    pes     = np.random.poisson(pes)
    return pes

estimate_s2_pes_at_pmts  = partial(estimate_s2_pes_at_sensors,
                                  x_sensors = np.zeros_like(dsim.x_pmts),
                                  y_sensors = np.zeros_like(dsim.y_pmts),
                                  psf       = dsim.psf_pmt)


estimate_s2_pes_at_sipms = partial(estimate_s2_pes_at_sensors,
                                   x_sensors = dsim.x_sipms,
                                   y_sensors = dsim.y_sipms,
                                   psf       = dsim.psf_sipm)

#
# Trigger
#------------------------

def trigger_time(ts       : np.array,
                 pes      : np.array,
                 ttrigger : float = dsim.trigger_time):
    return ttrigger - np.min(ts)

#
#  WFs
#-----------


def sample_pes_and_fill_wfs(ts          : np.array, # nelectrons
                            pes         : np.array, # (nelectrons, nsensors)
                            wfs         : np.array,  # (nwfsamples, nsensors)
                            wf_bin_time : float = dsim.wf_pmt_bin_time,
                            nsamples    : int = 1):
    """ create the wfs starting drom the pes produced per electrons and each sensor
    the control parameters are the wf_bin_time and the el_time_sample,
    by default they are the same parameters.
    The number of pes photos per sensor are spread along the EL-time depending
    on the wf_bin_time
    Returns: time bins of the wf (nbins), and wfs contents (nbins, nsensors)
    """

    def _wf(its, ipes, iwf):
        # for each sensor, check if has pes, sample pes in EL times and fill wf
        if (np.sum(ipes) <= 0): return iwf
        isel       = ipes > 0
        nts        = np.repeat(its[isel], ipes[isel])
        #nts        = np.repeat(its, ipes)
        sits, spes = bincounter(nts, wf_bin_time)

        spesn       = np.random.poisson(spes/nsamples, size = (nsamples, spes.size))
        for kk, kpes in enumerate(spesn):
            iwf[sits + kk] = iwf[sits + kk] + kpes
        return iwf

    [_wf(ts, ipes, iwf) for ipes, iwf in zip(pes.T, wfs.T)]

    return wfs


fill_wfs_s1       = partial(sample_pes_and_fill_wfs,
                            wf_bin_time = dsim.wf_pmt_bin_time,
                            nsamples    = dsim.s1_nsamples)


fill_wfs_s2_pmts  = partial(sample_pes_and_fill_wfs,
                           wf_bin_time = dsim.wf_pmt_bin_time,
                           nsamples    = dsim.s2_pmt_nsamples)


fill_wfs_s2_sipms = partial(sample_pes_and_fill_wfs,
                            wf_bin_time = dsim.wf_sipm_bin_time,
                            nsamples    = dsim.s2_sipm_nsamples)


#
# IC - functions
#--------------------


def get_detsim_functions(dsim : DetSimParameters = dsim) -> Tuple[Callable]:

    gen_electrons_     = partial(generate_electrons,
                                wi          = dsim.wi,
                                fano_factor = dsim.fano_factor)

    drift_electrons_   = partial(drift_electrons,
                                lifetime       = dsim.lifetime,
                                drift_velocity = dsim.drift_velocity)

    diffuse_electrons_ = partial(diffuse_electrons,
                                 transverse_diffusion   = dsim.transverse_diffusion,
                                 longitudinal_diffusion = dsim.longitudinal_diffusion,
                                 voxel_sizes            = dsim.voxel_sizes)

    gen_s1_photons_    = partial(generate_s1_photons,
                                 ws = dsim.ws)

    s1_pes_            = partial(estimate_s1_pes,
                                x_sensors = dsim.x_pmts,
                                y_sensors = dsim.y_pmts,
                                z_sensors = dsim.EP_z,
                                psf       = dsim.psf_s1)

    gen_s2_photons_    = partial(generate_s2_photons,
                                 el_gain       = dsim.el_gain,
                                 el_gain_sigma = dsim.el_gain_sigma)


    s2_pes_at_pmts_    = partial(estimate_s2_pes_at_sensors,
                                  x_sensors = np.zeros_like(dsim.x_pmts),
                                  y_sensors = np.zeros_like(dsim.y_pmts),
                                  psf       = dsim.psf_pmt)

    s2_pes_at_sipms_  = partial(estimate_s2_pes_at_sensors,
                                x_sensors = dsim.x_sipms,
                                y_sensors = dsim.y_sipms,
                                psf       = dsim.psf_sipm)


    trigger_time_     = partial(trigger_time,
                                ttrigger = dsim.trigger_time)

    fill_wfs_s1_      = partial(sample_pes_and_fill_wfs,
                                wf_bin_time = dsim.wf_pmt_bin_time,
                                nsamples    = dsim.s1_nsamples)


    fill_wfs_s2_pmts_ = partial(sample_pes_and_fill_wfs,
                                wf_bin_time = dsim.wf_pmt_bin_time,
                                nsamples    = dsim.s2_pmt_nsamples)


    fill_wfs_s2_sipms_ = partial(sample_pes_and_fill_wfs,
                                 wf_bin_time = dsim.wf_sipm_bin_time,
                                 nsamples    = dsim.s2_sipm_nsamples)

    functions = (gen_electrons_ , drift_electrons_ , diffuse_electrons_,
                 gen_s1_photons_, s1_pes_          ,
                 gen_s2_photons_, s2_pes_at_pmts_  , s2_pes_at_sipms_,
                 trigger_time_  ,
                 fill_wfs_s1_   , fill_wfs_s2_pmts , fill_wfs_s2_sipms_)

    return functions


# IC main function
#-------------------


def get_function_generate_wfs(detector         : str  = 'new',
                              run_number       : int  = -1,
                              krmap_filename   : str  = '',
                              psfsipm_filename : str  = '',
                              conf             : dict  = {}
                             ) -> Tuple[np.array, np.array]:

    dsim     = DetSimParameters(detector, run_number, krmap_filename, psfsipm_filename, **conf)

    generate_electrons_ , drift_electrons_ , diffuse_electrons_,  \
    generate_s1_photons_, s1_pes_          ,                      \
    generate_s2_photons_, s2_pes_at_pmts_  , s2_pes_at_sipms_,   \
    trigger_time_       ,                                        \
    fill_wfs_s1_        , fill_wfs_s2_pmts_, fill_wfs_s2_sipms_  = get_detsim_functions(dsim)


    def _generate_wfs(hits):

        xs, ys, zs, enes    = get_deposits(hits)

        nes                 = generate_electrons_(enes)
        nes                 = drift_electrons_(zs, nes)
        dxs, dys, dzs, dnes = diffuse_electrons_(xs, ys, zs, nes)

        dts                 = dzs / dsim.drift_velocity
        s2photons           = generate_s2_photons_(dnes)
        pes_pmts            = s2_pes_at_pmts_ (dxs, dys, s2photons)
        pes_sipms           = s2_pes_at_sipms_(dxs, dys, s2photons)

        s1photons           = generate_s1_photons_(enes)
        pes_s1              = s1_pes_(xs, ys, zs, s1photons)

        t0                  = trigger_time_(dts, pes_pmts)
        t0s                 = t0 * np.ones_like(zs)
        dts                 = t0 + dts

        wfs_pmts            = np.zeros((dsim.wf_pmt_nbins, dsim.npmts), dtype = int)
        fill_wfs_s1     (t0s, pes_s1  , wfs_pmts)
        fill_wfs_s2_pmts(dts, pes_pmts, wfs_pmts)

        wfs_sipms           = np.zeros((dsim.wf_sipm_nbins, dsim.nsipms), dtype = int)
        fill_wfs_s2_sipms(dts, pes_sipms, wfs_sipms)

        wfs_pmts            = wfs_pmts  * dsim.adc_to_pes_pmts
        sipms_pmts          = wfs_sipms * dsim.adc_to_pes_sipms

        return wfs_pmts, wfs_sipms

    return _generate_wfs
