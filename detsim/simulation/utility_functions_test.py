import numpy as np

from numpy.testing import assert_allclose
from pytest        import            mark

from .utility_functions import light_scale

@mark.parametrize("rel_scale",
                  (None, np.array([0.78, 1., 0.79, 0.70, 1.05, 1.01, 0.81, 0.81, 1.01, 0.88, 0.93, 0.81])))
def test_light_scale(rel_scale):

    fake_sensors = np.full((12, 10), 1)

    global_scale = 3

    relative_s = None if rel_scale is None else rel_scale[:, np.newaxis]

    scaled_sensors = light_scale(fake_sensors,
                                 global_scale,
                                   relative_s)

    assert scaled_sensors.shape == fake_sensors.shape
    if rel_scale is None:
        assert np.all(scaled_sensors == global_scale)
    else:
        assert_allclose(scaled_sensors[:, 0], global_scale * rel_scale)
