import numpy as np

from typing import List


def trigger_times(trigger_indx : List[int] ,
                  event_time   :      float,
                  time_bins    : np.ndarray) -> List[int]:

    return [event_time + time_bins[trg] for trg in trigger_indx]
