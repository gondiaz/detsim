import numpy as np

from pytest import fixture
from pytest import    mark

from invisible_cities.database                   import                   load_db as DB
from invisible_cities.io      .mcinfo_io         import load_mcsensor_response_df
from invisible_cities.core    .system_of_units_c import                     units

from . buffer_functions           import         calculate_binning
from . buffer_functions           import         calculate_buffers
from . buffer_functions           import            trigger_finder


@fixture(scope="module")
def mc_waveforms(fullsim_data):
    return load_mcsensor_response_df(fullsim_data, 'new', -6400)


## !! to-do: generalise for all detector configurations
@fixture(scope="module")
def pmt_ids():
    return DB.DataPMT('new', 6400).SensorID.values


@fixture(scope="module")
def sipm_ids():
    return DB.DataSiPM('new', 6400).SensorID.values


@fixture(scope="module")
def binned_waveforms(mc_waveforms, pmt_ids, sipm_ids):
    max_buffer = 10 * units.minute
    wf_binner  = calculate_binning(max_buffer)

    evts, pmt_binwid, sipm_binwid, all_wfs = mc_waveforms

    evt   = evts[0]
    wfs   = all_wfs.loc[evt]

    pmts  = wfs.loc[ pmt_ids]
    sipms = wfs.loc[sipm_ids]

    ## Assumes pmts the triggering sensors as in new/next-100
    pmt_bins ,  pmt_wf = wf_binner(pmts ,  pmt_binwid)
    sipm_bins, sipm_wf = wf_binner(sipms, sipm_binwid,
                                   pmt_bins[0],
                                   pmt_bins[-1] + np.diff(pmt_bins)[-1])
    return pmt_bins, pmt_wf, sipm_bins, sipm_wf


def test_calculate_binning(mc_waveforms, pmt_ids, sipm_ids, binned_waveforms):

    pmt_bins, pmt_wf, sipm_bins, sipm_wf = binned_waveforms

    evts, pmt_binwid, sipm_binwid, all_wfs = mc_waveforms

    evt   = evts[0]
    wfs   = all_wfs.loc[evt]

    pmts  = wfs.loc[ pmt_ids]
    sipms = wfs.loc[sipm_ids]

    assert np.all(np.diff( pmt_bins) ==  pmt_binwid)
    assert np.all(np.diff(sipm_bins) == sipm_binwid)
    assert pmt_bins[ 0] >= sipm_bins[ 0]
    assert pmt_bins[-1] >= sipm_bins[-1]

    pmt_sum  = pmts .charge.sum()
    sipm_sum = sipms.charge.sum()
    assert pmt_wf .sum().sum() ==  pmt_sum
    assert sipm_wf.sum().sum() == sipm_sum


@mark.parametrize("trigger_thresh", (2, 10))
def test_trigger_finder(binned_waveforms, trigger_thresh):

    pmt_bins, pmt_wf, *_ = binned_waveforms

    buffer_length = 800
    bin_width     = np.diff(pmt_bins)[0]

    trg_finder    = trigger_finder(buffer_length ,
                                   bin_width     ,
                                   trigger_thresh)

    pmt_sum       = pmt_wf.sum()
    triggers      = trg_finder(pmt_wf)

    assert np.all(pmt_sum[triggers] > trigger_thresh)


@mark.parametrize("pre_trigger trigger_thresh".split(),
                  ((100,  2),
                   (400, 10)))
def test_calculate_buffers(mc_waveforms, binned_waveforms,
                           pre_trigger, trigger_thresh):

    _, pmt_binwid, sipm_binwid, _ = mc_waveforms

    pmt_bins, pmt_wf, sipm_bins, sipm_wf = binned_waveforms

    buffer_length     = 800

    trg_finder        = trigger_finder(buffer_length ,
                                       pmt_binwid    ,
                                       trigger_thresh)

    triggers          = trg_finder(pmt_wf)

    buffer_calculator = calculate_buffers(buffer_length,
                                          pre_trigger  ,
                                          pmt_binwid   ,
                                          sipm_binwid  )

    buffers           = buffer_calculator(triggers, *binned_waveforms)
    pmt_sum           = pmt_wf.sum()

    assert len(buffers) == len(triggers)
    for i, evt in enumerate(buffers):
        sipm_trg_bin = np.where(sipm_bins <= pmt_bins[triggers[i]])[0][-1]
        diff_binedge = pmt_bins[triggers[i]] - sipm_bins[sipm_trg_bin]
        pre_trg_samp = int(pre_trigger * units.mus / pmt_binwid + diff_binedge)

        assert pmt_wf .shape[0] == evt[0].shape[0]
        assert sipm_wf.shape[0] == evt[1].shape[0]
        assert evt[0] .shape[1] == int(buffer_length * units.mus / pmt_binwid)
        assert np.sum(evt[0], axis=0)[pre_trg_samp] == pmt_sum[triggers[i]]
