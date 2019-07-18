import tables as tb

from functools import     wraps
from typing    import  Callable
from typing    import Generator
from typing    import     Tuple

from invisible_cities.database           import                load_db as  DB
from invisible_cities.io      .mcinfo_io import load_mcsensor_response
from invisible_cities.io      .rwf_io    import             rwf_writer
from invisible_cities.reco               import          tbl_functions as tbl


class event_map(tb.IsDescription):
    """
    Saves the mapping of the output events
    to the original nexus events.
    """
    nexus_evt = tb.Int32Col(shape=(), pos=0)


def get_sensor_binning(file_in     : str,
                       db_detector : str,
                       run_number  : int) -> Tuple:
    first_pmt  = DB.DataPMT (db_detector, run_number).values[0]
    first sipm = DB.DataSiPM(db_detector, run_number).values[0]
    first_evt  = next(iter(load_mcsensor_response(file_in, (0, 1)).values()))

    ## Assumes all PMTs present, valid?
    pmt_width  = first_evt[first_pmt].bin_width
    sipm_indx  = np.where(np.array(tuple(first_evt)) >= first_sipm)[0][0]
    sipm_width = first_evt[tuple(first_evt)[sipm_indx]].bin_width

    return pmt_width, sipm_width


@wraps(rwf_writer)
def buffer_writer(h5out, *,
                  group_name  : str = 'detsim',
                  compression : str =  'ZLIB4',
                  n_sens_eng  : int =       12,
                  n_sens_trk  : int =     1792,
                  length_eng  : int           ,
                  length_trk  : int           ) -> Callable[[int, List, List, List], None]:

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

    h5_group      = getattr(h5out.root, group_name)
    nexus_evt_tbl = h5out.create_table(h5_group, "nexus_evt", event_map,
                                       "nexus run numbers for each index",
                                       tbl.filters(compression))

    def write_buffers(nexus_evt      :         int,
                      eng_sens_order :   List[int],
                      trk_sens_order :   List[int],
                      events         : List[Tuple]) -> None:

        eng_sens = np.zeros((len(events), n_sens_eng, length_eng), np.int)
        trk_sens = np.zeros((len(events), n_sens_trk, length_trk), np.int)
        for e_sens, t_sens, (eng, trk) in zip(eng_sens, trk_sens, events):
            row = nexus_evt_tbl.row
            row["nexus_evt"] = nexus_evt
            row.append()

            e_sens[eng_sens_order] = eng
            eng_writer(e_sens)

            t_sens[trk_sens_order] = trk
            trk_writer(t_sens)
    return write_buffers


def load_sensors(file_names : str,
                 db_file    : str,
                 run_no     : int) -> Generator[Tuple, None, None]:

    all_evt = load_mcsensor_response(file_name)

    ## Generalisable for an all sipm detector?
    pmt_ids = DB.DataPMT(db_file, run_no).SensorID.values
    #sipm_ids = DB.DataSiPM(db_file, run_no).SensorID.values

    for evt, wfs in all_evt.items():
        pmt_ord  = []
        pmt_wfs  = []
        sipm_ord = []
        sipm_wfs = []
        for sens_id, wf in wfs.items():
            if sens_id in pmt_ids:
                pmt_ord .append(sens_id)
                pmt_wfs .append(     wf)
            else:
                sipm_ord.append(sens_id)
                sipm_wfs.append(     wf)
        yield dict(evt = evt,
                   pmt_ord  =  pmt_ord, pmt_wfs  =  pmt_wfs,
                   sipm_ord = sipm_ord, sipm_wfs = sipm_wfs)
