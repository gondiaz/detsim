import numpy as np


def light_scale(sensors        : np.ndarray,
                scale_factor   :   int  = 1,
                relative_scale : np.ndarray = None) -> np.ndarray:
    """
    Multiply the signal levels in the
    sensors array by a scaling factor

    """
    if relative_scale is None:
        return sensors * scale_factor

    return sensors * relative_scale * scale_factor
