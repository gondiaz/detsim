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

from glob      import      glob
from functools import     wraps
from functools import   partial
from typing    import     Tuple
from typing    import  Callable
from typing    import Generator
from typing    import      List

from detsim.io        .hdf5_io          import                 wf_writer
from detsim.io        .hdf5_io          import                 load_hits
from detsim.io        .hdf5_io          import             save_run_info
from detsim.util      .util             import             trigger_times
from detsim.simulation.buffer_functions import         calculate_binning
from detsim.simulation.buffer_functions import         calculate_buffers
from detsim.simulation.buffer_functions import            trigger_finder
from detsim.simulation.detsim_functions import get_function_generate_wfs
from detsim.simulation.detsim_functions import              detsimparams

from invisible_cities.core    .configure         import          configure
from invisible_cities.core    .system_of_units_c import              units
from invisible_cities.database.load_db           import            DataPMT
from invisible_cities.database.load_db           import           DataSiPM
from invisible_cities.io      .mcinfo_io         import     mc_info_writer
from invisible_cities.io      .mcinfo_io         import get_sensor_binning
from invisible_cities.io      .mcinfo_io         import        load_mchits
from invisible_cities.reco                       import      tbl_functions as tbl

from invisible_cities.dataflow          import dataflow as fl
from invisible_cities.dataflow.dataflow import     fork
from invisible_cities.dataflow.dataflow import     push
from invisible_cities.dataflow.dataflow import     pipe

from invisible_cities.cities.components import city

@city
def detsim(files_in, file_out, compression, event_range,
           detector_db, run_number):

    npmt, nsipm        = detsimparams.npmts, detsimparams.nsipms
    pmt_wid, sipm_wid  = detsimparams.wf_pmt_bin_time, detsimparams.wf_sipm_bin_time
    nsamp_pmt          = int(detsimparams.wf_buffer_time / pmt_wid)
    nsamp_sipm         = int(detsimparams.wf_buffer_time / sipm_wid)

    generate_wfs_      = fl.map(get_function_generate_wfs(),
                                args = ("hits"),
                                out = ("wfs"))

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:

        write_mc         = fl.sink(mc_info_writer(h5out),
                                   args = ("mc", "evt"))
        wf_writer_   = fl.sink(wf_writer(h5out                       ,
                                                 n_sens_eng = npmt       ,
                                                 n_sens_trk = nsipm      ,
                                                 length_eng = nsamp_pmt  ,
                                                 length_trk = nsamp_sipm),
                                   args = ("evt", "wfs"))

        save_run_info(h5out, run_number)
        return push(source = load_hits(files_in),
                    pipe   = pipe(generate_wfs_        ,
                                  fork(wf_writer_,
                                       write_mc      )))

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


if __name__ == "__main__":
    conf = configure(sys.argv).as_namespace
    position_signal(conf)
