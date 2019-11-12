import numpy  as np
import pandas as pd

from invisible_cities.reco.peak_functions    import indices_and_wf_above_threshold
from invisible_cities.reco.peak_functions    import                 split_in_peaks
from invisible_cities.evm .event_model       import                       Waveform
from invisible_cities.core.system_of_units_c import                          units

from typing    import    Tuple
from typing    import     List
from typing    import Callable
from typing    import  Mapping

from functools import    wraps
from functools import  partial


@wraps(np.histogram)
def weighted_histogram(data : pd.DataFrame, bins : np.ndarray) -> np.ndarray:
    return np.histogram(data.time, weights=data.charge, bins=bins)[0]


def padder(sensors : np.ndarray, padding : Tuple) -> np.ndarray:
    return np.apply_along_axis(np.pad, 1, sensors, padding, "constant")


def calculate_buffers(buffer_len : float, pre_trigger : float,
                      pmt_binwid : float, sipm_binwid : float) -> Callable:

    def buffer_samples(bin_width : float) -> int:
        return int(buffer_len * units.mus /  bin_width)

    def pre_samples(bin_width  : float,
                    correction : float = 0) -> int:
        correct_bins = int(correction / bin_width)
        return correct_bins + int(pre_trigger * units.mus / bin_width)

    def post_samples(bin_width : float,
                     pre_samp  :   int) -> int:
        return int(buffer_samples(bin_width) - pre_samp)

    def sipm_trg_bin(sipm_bins : np.ndarray,
                     pmt_bins  : np.ndarray) -> Callable[[int], int]:
        def get_sipm_bin(trigger : int) -> int:
            return np.where(sipm_bins <= pmt_bins[trigger])[0][-1]
        return get_sipm_bin

    def slice_generator(pmt_bins    : np.ndarray,
                        pmt_charge  : np.ndarray,
                        sipm_bins   : np.ndarray,
                        sipm_charge : np.ndarray) -> Callable[[List], Tuple]:

        npmt_bin    = len(pmt_bins)
        nsipm_bin   = len(sipm_bins)
        pmt_buffer  = buffer_samples( pmt_binwid)
        sipm_buffer = buffer_samples(sipm_binwid)

        _pmt_pretrg = partial( pre_samples, pmt_binwid)
        _pmt_postrg = partial(post_samples, pmt_binwid)
        sipm_pretrg = pre_samples(sipm_binwid)
        sipm_postrg = post_samples(sipm_binwid, sipm_pretrg)

        sipm_trg    = sipm_trg_bin(sipm_bins, pmt_bins)

        def generate_slices(triggers : List) -> Tuple:

            for trg in triggers:
                sipm_trg_bin = sipm_trg(trg)
                pmt_pretrg   = _pmt_pretrg(pmt_bins[trg] - sipm_bins[sipm_trg_bin])
                pmt_postrg   = _pmt_postrg(pmt_pretrg)

                pmt_pre = 0, trg - pmt_pretrg
                pmt_sl  = slice(max(pmt_pre),
                                min(npmt_bin, trg + pmt_postrg))
                sipm_sl = slice(max(0, sipm_trg_bin - sipm_pretrg),
                                min(nsipm_bin, sipm_trg_bin + sipm_postrg))
                pmt_pd  = (int(-min(pmt_pre)),
                           int( max(0, trg + pmt_postrg - npmt_bin + 1)))
                sipm_pd = (int(-min(0, sipm_trg_bin - sipm_pretrg)),
                           int( max(0, sipm_trg_bin + sipm_postrg - nsipm_bin + 1)))
                yield (pmt_charge[:, pmt_sl], pmt_pd), (sipm_charge[:, sipm_sl], sipm_pd)
        return generate_slices


    def position_signal(triggers, pmt_bins, pmt_charge,
                        sipm_bins, sipm_charge):

        slice_and_pad = slice_generator(pmt_bins                             ,
                                        np.array(pmt_charge.values.tolist()) ,
                                        sipm_bins                            ,
                                        np.array(sipm_charge.values.tolist()))

        return [(padder(*pmts), padder(*sipms))
                for pmts, sipms in slice_and_pad(triggers)]
    return position_signal


def calculate_binning(max_buffer : int) -> Callable:
    """
    Returns a function to be used to convert the raw
    input Waveforms into data binned according to
    the bin width stored in the Waveforms, effectively
    padding with zeros inbetween the separate signals.

    max_buffer : float
        Maximum event time to be considered in nanoseconds
    """
    def bin_data(sensors   : pd.Series   ,
                 bin_width : float       ,
                 t_min     : float = None,
                 t_max     : float = None) -> Tuple:
        """
        Raw data binning function.

        sensors : List of Waveforms
            Should be sorted into one type/binning
        t_min : float
            Minimum time to be used to define bins.
            Should be used only if the binning is defined
            by one type of sensor to be applied to another
            as in the case of NEW with PMTs and SiPMs
        t_max : float
            As t_min but the maximum to be used
        """
        if t_min is None or t_max is None:
            min_time = sensors.time.min()
            max_time = min(sensors.time.max()  ,
                           min_time + max_buffer)
            min_bin  = np.floor(min_time / bin_width) * bin_width
            max_bin  = np.floor(max_time / bin_width) * bin_width
            max_bin += bin_width
        else:
            ## Adjust according to bin_width
            min_bin  = np.floor(t_min / bin_width) * bin_width
            max_bin  = np.ceil (t_max / bin_width) * bin_width

        bins = np.arange(min_bin, max_bin, bin_width)

        bin_sensors = sensors.groupby('sensor_id').apply(weighted_histogram,
                                                         bins              )
        return bins, bin_sensors
    return bin_data


## !! to-do: clarify for non-pmt versions of next
def trigger_finder(buffer_len    : float,
                   bin_width     : float,
                   bin_threshold :   int) -> Callable:
    """
    Decides where possible triggers could be
    based on the PMT sum in order to give
    a useful position for buffer selection
    """

    stand_off = int(buffer_len * units.mus / bin_width)
    def find_triggers(pmt_wfs : pd.Series) -> List[int]:

        pmt_sum = pmt_wfs.sum(0)
        indices = indices_and_wf_above_threshold(pmt_sum,
                                                 bin_threshold).indices
        ## Just using this and the stand_off for now
        ## taking first above sum threshold.
        ## !! To-do: make more robust with min int? or similar
        all_indx = split_in_peaks(indices, stand_off)
        return [trg[0] for trg in all_indx]
    return find_triggers
