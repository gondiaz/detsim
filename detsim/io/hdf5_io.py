import numpy  as np
import pandas as pd
import tables as tb

from functools import     wraps
from typing    import  Callable
from typing    import Generator
from typing    import     Tuple
from typing    import      List

from invisible_cities.database           import                   load_db as  DB
from invisible_cities.io      .mcinfo_io import load_mcsensor_response_df
from invisible_cities.io      .mcinfo_io import    load_mcsensor_response
from invisible_cities.io      .mcinfo_io import       read_mcsns_response
from invisible_cities.io      .rwf_io    import                rwf_writer
from invisible_cities.reco               import             tbl_functions as tbl


class EventInfo(tb.IsDescription):
    """
    For the runInfo table to save event
    number and timestamp.
    Additionally, saves the original nexus
    event number
    """
    event_number = tb. Int32Col(shape=(), pos=0)
    timestamp    = tb.UInt64Col(shape=(), pos=1)
    nexus_evt    = tb. Int32Col(shape=(), pos=2)


class RunInfo(tb.IsDescription):
    """
    Saves the run number in its own table
    This is expected by diomira.
    """
    run_number = tb.Int32Col(shape=(), pos=0)


def save_run_info(h5out      : tb.file.File,
                  run_number :          int) -> None:
    try:
        run_table = getattr(h5out.root.Run, 'runInfo')
    except tb.NoSuchNodeError:
        try:
            run_group = getattr(h5out.root, 'Run')
            run_table = h5out.create_table(run_group, "runInfo", RunInfo,
                                           "Run number used in detsim")
        except tb.NoSuchNodeError:
            run_group = h5out.create_group(h5out.root, 'Run')
            run_table = h5out.create_table(run_group, "runInfo", RunInfo,
                                           "Run number used in detsim")
    row = run_table.row
    row["run_number"] = run_number
    row.append()


def get_sensor_binning(file_in     : str,
                       db_detector : str,
                       run_number  : int) -> Tuple:
    """
    Returns the bin width for the sensor information
    read from file using the widths given in the
    first recorded event.
    """

    first_pmt  = DB.DataPMT (db_detector, run_number).SensorID.values[0]
    first_sipm = DB.DataSiPM(db_detector, run_number).SensorID.values[0]
    first_evt  = next(iter(load_mcsensor_response(file_in, (0, 1)).values()))

    ## Assumes all PMTs present, valid?
    pmt_width  = first_evt[first_pmt].bin_width
    sipm_indx  = np.where(np.array(tuple(first_evt)) >= first_sipm)[0][0]
    sipm_width = first_evt[tuple(first_evt)[sipm_indx]].bin_width

    return pmt_width, sipm_width


def event_timestamp(file_in : tb.file.File) -> Callable:
    """
    Returns a function giving access to the next
    events first hit time.
    This is used as the event timestamp.
    Generally set to zero in nexus but here for
    completeness.
    """

    hit_inds   = (ext[2] + 1 for ext in file_in.root.MC.extents[:-1])
    first_hits = iter(np.concatenate(([0], np.fromiter(hit_inds, np.int))))
    max_iter = len(file_in.root.MC.extents)
    def get_evt_timestamp() -> float:
        get_evt_timestamp.counter += 1
        if get_evt_timestamp.counter > max_iter:
            raise IndexError('No more events')
        return file_in.root.MC.hits[next(first_hits)][2]
    get_evt_timestamp.counter = 0
    return get_evt_timestamp


@wraps(rwf_writer)
def buffer_writer(h5out, *,
                  group_name  : str = 'detsim',
                  compression : str =  'ZLIB4',
                  n_sens_eng  : int =       12,
                  n_sens_trk  : int =     1792,
                  length_eng  : int           ,
                  length_trk  : int           ) -> Callable[[int, List, List, List], None]:
    """
    Generalised buffer writer which defines a raw waveform writer
    for each type of sensor as well as a event info writer
    with written event, timestamp and a mapping to the
    nexus event number in case of event splitting.
    """

    eng_writer = rwf_writer(h5out,
                            group_name      =  group_name,
                            compression     = compression,
                            table_name      =     'pmtrd',
                            n_sensors       =  n_sens_eng,
                            waveform_length =  length_eng)

    trk_writer = rwf_writer(h5out,
                            group_name      =  group_name,
                            compression     = compression,
                            table_name      =    'sipmrd',
                            n_sensors       =  n_sens_trk,
                            waveform_length =  length_trk)

    try:
        evt_group = getattr(h5out.root, 'Run')
    except tb.NoSuchNodeError:
        evt_group = h5out.create_group(h5out.root, 'Run')

    nexus_evt_tbl = h5out.create_table(evt_group, "events", EventInfo,
                                       "event, timestamp & nexus evt for each index",
                                       tbl.filters(compression))

    def write_buffers(nexus_evt      :          int ,
                      eng_sens_order :   List[  int],
                      trk_sens_order :   List[  int],
                      timestamps     :   List[  int],
                      events         :   List[Tuple]) -> None:

        for t_stamp, (eng, trk) in zip(timestamps, events):
            row = nexus_evt_tbl.row
            row["event_number"] = write_buffers.counter
            row["timestamp"]    = t_stamp
            row["nexus_evt"]    = nexus_evt
            row.append()

            e_sens = np.zeros((n_sens_eng, length_eng), np.int)
            t_sens = np.zeros((n_sens_trk, length_trk), np.int)

            e_sens[eng_sens_order] = eng
            eng_writer(e_sens)

            t_sens[trk_sens_order] = trk
            trk_writer(t_sens)

            write_buffers.counter += 1
    write_buffers.counter = 0
    return write_buffers


def load_sensors(file_names : List[str],
                 db_file    :      str ,
                 run_no     :      int ) -> Generator[Tuple, None, None]:

    pmt_ids  = DB.DataPMT (db_file, run_no).SensorID
    sipm_ids = DB.DataSiPM(db_file, run_no).SensorID

    for file_name in file_names:

        (all_evt    ,
         pmt_binwid ,
         sipm_binwid,
         all_wf     ) = load_mcsensor_response_df(file_name, db_file, run_no)

        with tb.open_file(file_name, 'r') as h5in:

            mc_info = tbl.get_mc_info(h5in)

            timestamps = event_timestamp(h5in)

            for evt in all_evt:

                pmt_wfs  = all_wf.loc[evt].loc[ pmt_ids]
                sipm_wfs = all_wf.loc[evt].loc[sipm_ids]

                yield dict(evt         = evt         ,
                           mc          = mc_info     ,
                           timestamp   = timestamps(),
                           pmt_binwid  = pmt_binwid  ,
                           sipm_binwid = sipm_binwid ,
                           pmt_wfs     = pmt_wfs     ,
                           sipm_wfs    = sipm_wfs    )


## !! This uses a copy of the fanal function load_mc_hits
## !! Will be imported and wrapped as necessary once
## !! moved to IC
def load_hits(file_names : List[str]) -> Generator:

    for file_name in file_names:
        with tb.open_file(file_name) as h5in:

            extents    = pd.read_hdf(file_name, 'MC/extents')

            event_ids  = extents.evt_number

            hits_df    = load_mc_hits(file_name, extents)

            mc_info    = tbl.get_mc_info(h5in)

            timestamps = event_timestamp(h5in)

            for evt in event_ids:
                yield dict(evt       = evt                ,
                           mc        = mc_info            ,
                           timestamp = timestamp()        ,
                           hits      = hits_df.loc[evt, :])


## !! This code temporarily copied/adapted from FANAL
def load_mc_hits(h5in    : tb.file.File,
                 extents : pd.DataFrame) -> pd.DataFrame:

    hits_tb  = h5in.root.MC.hits

    # Generating hits DataFrame
    hits = pd.DataFrame({'hit_id'      : hits_tb.col('hit_indx'),
                         'particle_id' : hits_tb.col('particle_indx'),
                         'label'       : hits_tb.col('label').astype('U13'),
                         'time'        : hits_tb.col('hit_time'),
                         'x'           : hits_tb.col('hit_position')[:, 0],
                         'y'           : hits_tb.col('hit_position')[:, 1],
                         'z'           : hits_tb.col('hit_position')[:, 2],
                         'E'           : hits_tb.col('hit_energy')})

    evt_hit_df = extents[['last_hit', 'evt_number']]
    evt_hit_df.set_index('last_hit', inplace = True)

    hits = hits.merge(evt_hit_df, left_index=True, right_index=True, how='left')
    hits.rename(columns={"evt_number": "event_id"}, inplace = True)
    hits.event_id.fillna(method='bfill', inplace = True)
    hits.event_id = hits.event_id.astype(int)

    # Setting the indexes
    hits.set_index(['event_id', 'particle_id', 'hit_id'], inplace=True)

    return hits
