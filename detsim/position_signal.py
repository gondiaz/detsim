"""
Module taking in nexus full simulation and
positioning the signal according to a
given buffer size, pre-trigger and simple
trigger conditions.
"""

import   os
import json
import  sys

import numpy  as np
import pandas as pd
import tables as tb

from glob      import    glob
from functools import   wraps
from functools import partial
from typing    import   Tuple

from detsim.io        .hdf5_io          import      buffer_writer
#from detsim.io        .hdf5_io          import get_sensor_binning
from detsim.io        .hdf5_io          import       load_sensors
from detsim.io        .hdf5_io          import      save_run_info
from detsim.util      .util             import      trigger_times
from detsim.simulation.buffer_functions import  calculate_binning
from detsim.simulation.buffer_functions import  calculate_buffers
from detsim.simulation.buffer_functions import     trigger_finder

from invisible_cities.core    .configure         import          configure
from invisible_cities.core    .system_of_units_c import              units
from invisible_cities.database.load_db           import            DataPMT
from invisible_cities.database.load_db           import           DataSiPM
#from invisible_cities.detsim  .buffer_functions  import calculate_binning
#from invisible_cities.detsim  .buffer_functions  import calculate_buffers
#from invisible_cities.detsim  .buffer_functions  import    trigger_finder
from invisible_cities.io      .mcinfo_io         import     mc_info_writer
from invisible_cities.io      .mcinfo_io         import get_sensor_binning
from invisible_cities.reco                       import      tbl_functions as tbl

from invisible_cities.dataflow          import dataflow as fl
from invisible_cities.dataflow.dataflow import     fork
from invisible_cities.dataflow.dataflow import     push
from invisible_cities.dataflow.dataflow import     pipe


def bin_minmax(sensor_binning : np.ndarray) -> Tuple:
    min_bin = sensor_binning[ 0]
    max_bin = sensor_binning[-1] + np.diff(sensor_binning)[-1]
    return min_bin, max_bin


def sensor_order(pmt_sr      : pd.Series,
                 sipm_sr     : pd.Series,
                 detector_db : str      ,
                 run_number  : int      ) -> Tuple:
    pmts     = DataPMT (detector_db, run_number).SensorID
    sipms    = DataSiPM(detector_db, run_number).SensorID
    pmt_ord  = pmts [ pmts.isin( pmt_sr.index.tolist())].index
    sipm_ord = sipms[sipms.isin(sipm_sr.index.tolist())].index
    return pmt_ord, sipm_ord


def get_no_sensors(detector_db : str, run_number : int) -> Tuple:
    npmt  = DataPMT (detector_db, run_number).shape[0]
    nsipm = DataSiPM(detector_db, run_number).shape[0]
    return npmt, nsipm


def position_signal(conf):

    files_in      = glob(os.path.expandvars(conf.files_in))
    file_out      =      os.path.expandvars(conf.file_out)
    detector_db   =                         conf.detector_db
    run_number    =                     int(conf.run_number)
    max_time      =                     int(conf.max_time)
    buffer_length =                   float(conf.buffer_length)
    pre_trigger   =                   float(conf.pre_trigger)
    trg_threshold =                   float(conf.trg_threshold)
    compression   =                         conf.compression

    npmt, nsipm        = get_no_sensors(detector_db, run_number)
    pmt_wid, sipm_wid  = get_sensor_binning(files_in[0])#,
                                            #detector_db,
                                            # run_number)
    nsamp_pmt          = int(buffer_length * units.mus /  pmt_wid)
    nsamp_sipm         = int(buffer_length * units.mus / sipm_wid)

    bin_calculation    = calculate_binning(max_time)
    pmt_binning        = fl.map(bin_calculation,
                                args = ("pmt_wfs" ,  "pmt_binwid"),
                                out  = ("pmt_bins", "pmt_bin_wfs"))

    extract_minmax     = fl.map(bin_minmax,
                                args = "pmt_bins",
                                out  = ("min_bin", "max_bin"))

    sipm_binning       = fl.map(bin_calculation,
                                args = ("sipm_wfs", "sipm_binwid",
                                        "min_bin" ,     "max_bin") ,
                                out  = ("sipm_bins", "sipm_bin_wfs"))

    sensor_order_      = fl.map(partial(sensor_order,
                                        detector_db = detector_db,
                                        run_number  =  run_number),
                                args = ("pmt_bin_wfs", "sipm_bin_wfs"),
                                out  = ("pmt_ord", "sipm_ord"))

    trigger_finder_    = fl.map(trigger_finder(buffer_length,
                                               pmt_wid, trg_threshold),
                                args = "pmt_wfs",
                                out  = "triggers")

    event_times        = fl.map(trigger_times,
                                args = ("triggers", "timestamp", "pmt_bins"),
                                out  = "evt_times")

    calculate_buffers_ = fl.map(calculate_buffers(buffer_length, pre_trigger,
                                                  pmt_wid      ,    sipm_wid),
                                args = ("triggers",
                                        "pmt_bins" ,  "pmt_bin_wfs",
                                        "sipm_bins", "sipm_bin_wfs"),
                                out  = "buffers")

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:

        write_mc         = fl.sink(mc_info_writer(h5out),
                                   args = ("mc", "evt"))
        buffer_writer_   = fl.sink(buffer_writer(h5out                  ,
                                                 n_sens_eng = npmt      ,
                                                 n_sens_trk = nsipm     ,
                                                 length_eng = nsamp_pmt ,
                                                 length_trk = nsamp_sipm),
                                   args = ("evt", "pmt_ord", "sipm_ord",
                                           "evt_times", "buffers"))

        save_run_info(h5out, run_number)
        return push(source = load_sensors(files_in, detector_db, run_number),
                    pipe   = pipe(pmt_binning         ,
                                  extract_minmax      ,
                                  sipm_binning        ,
                                  sensor_order_       ,
                                  trigger_finder_     ,
                                  event_times         ,
                                  calculate_buffers_  ,
                                  fork(buffer_writer_,
                                       write_mc      )))



if __name__ == "__main__":
    conf = configure(sys.argv).as_namespace
    position_signal(conf)
