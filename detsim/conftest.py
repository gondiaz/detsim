import os
import pytest


@pytest.fixture(scope = 'session')
def ICDIR():
    return os.environ['ICDIR']


@pytest.fixture(scope = 'session')
def ICDATADIR(ICDIR):
    return os.path.join(ICDIR, "database/test_data/")


@pytest.fixture(scope = 'session')
def config_tmpdir(tmpdir_factory):
    return tmpdir_factory.mktemp('configure_tests')


@pytest.fixture(scope = 'session')
def fullsim_data(ICDATADIR):
    return os.path.join(ICDATADIR,
                        'Kr83_full_nexus_v5_03_01_ACTIVE_7bar_1evt.sim.h5')
