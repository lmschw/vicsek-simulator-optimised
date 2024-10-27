from enum import Enum

"""
Indicates how the particles should be distributed spatially when selecting which particles are affected by 
an event.
"""
class ThresholdEvaluationMethod(Enum):
    LOCAL_ORDER = 0,
    ORIENTATION_DIFFERENCE_AVG = 1,
    ORIENTATION_DIFFERENCE_MIN = 2,
    ORIENTATION_DIFFERENCE_MAX = 3,
    NEIGHBOUR_DISTANCE_AVG = 4,
    NEIGHBOUR_DISTANCE_MIN = 5,
    NEIGHBOUR_DISTANCE_MAX = 6
