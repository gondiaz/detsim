"""
Module taking in nexus full simulation and
positioning the signal according to a
given buffer size, pre-trigger and simple
trigger conditions.
"""

import  sys
import json

import numpy  as np
import tables as tb

from functools import wraps
from typing    import Tuple

from .io.hdf5_io import      buffer_writer
from .io.hdf5_io import get_sensor_binning
from .io.hdf5_io import       load_sensors

from invisible_cities.core    .configure        import         configure
from invisible_cities.database.load_db          import           DataPMT
from invisible_cities.database.load_db          import          DataSiPM
from invisible_cities.dataflow                  import          dataflow as  fl
from invisible_cities.dataflow.dataflow         import              push
from invisible_cities.dataflow.dataflow         import              pipe
from invisible_cities.detsim  .buffer_functions import calculate_binning
from invisible_cities.detsim  .buffer_functions import calculate_buffers
from invisible_cities.detsim  .buffer_functions import    trigger_finder
from invisible_cities.reco                      import     tbl_functions as tbl


def bin_minmax(sensor_binning : np.ndarray) -> Tuple:
    min_bin = sensor_binning[ 0]
    max_bin = sensor_binning[-1] + np.diff(sensor_binning)[-1]
    return min_bin, max_bin


@wraps(np.sum)
def sensor_sum(sensors : np.ndarray) -> np.ndarray:
    return np.sum(sensors, axis = 0)


def get_no_sensors(detector_db : str, run_number : int) ->Tuple:
    npmt  = DataPMT (detector_db, run_number).shape[0]
    nsipm = DataSiPM(detector_db, run_number).shape[0]
    return npmt, nsipm


def position_signal(conf):

    files_in      = os.path.expandvars(conf.files_in)
    file_out      = os.path.expandvars(conf.file_out)
    detector_db   =                    conf.detector_db
    run_number    =                int(conf.run_number)
    max_time      =                int(conf.max_time)
    buffer_length =              float(conf.buffer_length)
    pre_trigger   =              float(conf.pre_trigger)
    trg_threshold =              float(conf.trg_threshold)
    compression   =                    conf.compression

    npmt, nsipm        = get_no_sensors(detector_db, run_number)
    pmt_wid, sipm_wid  = get_sensor_binning(files_in[0],
                                            detector_db,
                                             run_number)
    nsamp_pmt          = buffer_length * units.mus /  pmt_wid
    nsamp_sipm         = buffer_length * units.mus / sipm_wid

    bin_calculation    = calculate_binning(max_time)
    pmt_binning        = fl.map(bin_calculation,
                             args = "pmt_wfs",
                             out  = ("pmt_bins", "pmt_bin_wfs"))

    extract_minmax     = fl.map(bin_minmax,
                                args = "pmt_bins",
                                out  = ("min_bin", "max-bin"))

    sipm_binning       = fl.map(bin_calculation,
                                args = ("sipm_wfs", "min_bin", "max_bin"),
                                out  = ("sipm_bins", "sipm_bin_wfs"))

    pmt_summer         = fl.map(sensor_sum,
                                args = "pmt_bin_wfs",
                                out  = "pmt_sum")

    trigger_finder_    = fl.map(trigger_finder(buffer_length,
                                               pmt_wid, trg_threshold),
                                args = "pmt_sum",
                                out  = "triggers")

    calculate_buffers_ = fl.map(calculate_buffers(buffer_length, pre_trigger),
                                args = ("triggers",
                                        "pmt_bins" ,  "pmt_bin_wfs",
                                        "sipm_bins", "sipm_bin_wfs"),
                                out  = "buffers")

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:

        buffer_writer_ = fl.sink(buffer_writer(h5out,
                                               n_sens_eng = npmt   , n_sens_trk = nsipm   ,
                                               length_eng = pmt_wid, length_trk = sipm_wid),
                                 args = ("evt", "pmt_ord", "sipm_ord", "buffers"))

        return push(source = load_sensors(files_in, detector_db, run_number),
                    pipe   = pipe(pmt_binning       ,
                                  extract_minmax    ,
                                  sipm_binning      ,
                                  pmt_summer        ,
                                  trigger_finder_   ,
                                  calculate_buffers_,
                                  buffer_writer_    ))



if __name__ == "__main__":
    conf = configure(sys.argv).as_namespace
    position_signal(conf)
