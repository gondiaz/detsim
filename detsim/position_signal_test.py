import os

import numpy  as np
import pandas as pd
import tables as tb

from pytest import fixture
from pytest import    mark

from invisible_cities.core.configure     import              configure
from invisible_cities.core.testing_utils import assert_tables_equality

from . position_signal import      bin_minmax
from . position_signal import position_signal
from . position_signal import    sensor_order


def test_bin_minmax():

    bin_min    = 22
    bin_max    = 40
    test_bins  = np.arange(bin_min, bin_max)

    bmin, bmax = bin_minmax(test_bins)

    assert np.round(bmin) == bin_min
    assert np.round(bmax) == bin_max


@fixture(scope = 'function')
def ids_and_orders():
    id_dict = {'new'    :{'pmt_ids' :    (2, 5, 7), 'pmt_ord' :  (2, 5, 7),
                          'sipm_ids': (1010, 5023), 'sipm_ord': (10, 279)},
               'next100':{'pmt_ids' :     (22, 50), 'pmt_ord' :   (22, 50),
                          'sipm_ids': (5044, 7001), 'sipm_ord': (300, 385)}}
    return id_dict
@mark.parametrize("detector", ('new', 'next100'))
def test_sensor_order(ids_and_orders, detector):

    npmt     = len(ids_and_orders[detector]['pmt_ids'])
    pmt_sig  = pd.Series([np.random.uniform(size = 3) for i in range(npmt)],
                         index = ids_and_orders[detector]['pmt_ids'])
    nsipm    = len(ids_and_orders[detector]['sipm_ids'])
    sipm_sig = pd.Series([np.random.uniform(size = 3) for i in range(nsipm)],
                         index = ids_and_orders[detector]['sipm_ids'])

    pmt_ord, sipm_ord = sensor_order(pmt_sig, sipm_sig, detector, -1000)

    assert len( pmt_ord) == npmt
    assert len(sipm_ord) == nsipm

    assert np.all(pmt_ord  == ids_and_orders[detector][ 'pmt_ord'])
    assert np.all(sipm_ord == ids_and_orders[detector]['sipm_ord'])


def test_position_signal_kr(config_tmpdir, fullsim_data, test_config):

    PATH_OUT = os.path.join(config_tmpdir, 'Kr_fullsim.buffers.h5')

    conf = configure(['dummy', test_config])
    conf.update(dict(files_in = fullsim_data,
                     file_out = PATH_OUT    ))


    cnt = position_signal(conf.as_namespace)

    with tb.open_file(fullsim_data, mode='r') as h5in, \
         tb.open_file(PATH_OUT    , mode='r') as h5out:

        assert hasattr(h5out.root   ,        'MC')
        assert hasattr(h5out.root   ,       'Run')
        assert hasattr(h5out.root   ,    'detsim')
        assert hasattr(h5out.root.MC,   'extents')
        assert hasattr(h5out.root.MC,      'hits')
        assert hasattr(h5out.root.MC, 'particles')

        assert_tables_equality(h5in .root.MC.particles,
                               h5out.root.MC.particles)


def test_position_signal_neut(config_tmpdir, neut_fullsim,
                              test_config  , neut_buffers):

    PATH_OUT = os.path.join(config_tmpdir, 'neut_fullsim.buffers.h5')

    conf = configure(['dummy', test_config])
    conf.update(dict(files_in = neut_fullsim,
                     file_out = PATH_OUT    ))


    cnt = position_signal(conf.as_namespace)

    with tb.open_file(neut_buffers, mode='r') as h5test, \
         tb.open_file(PATH_OUT    , mode='r') as h5out:

         pmt_out  = h5out .root.detsim.pmtrd
         pmt_test = h5test.root.detsim.pmtrd

         assert pmt_out.shape == pmt_test.shape
         assert_tables_equality(pmt_out, pmt_test)

         sipm_out  = h5out .root.detsim.sipmrd
         sipm_test = h5test.root.detsim.sipmrd

         assert sipm_out.shape == sipm_test.shape
         assert_tables_equality(sipm_out, sipm_test)
