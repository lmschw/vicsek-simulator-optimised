from enum import Enum

"""
Contains the different hyperparameters that can be used to switch between behaviours.
"""
class SwitchType(str, Enum):
    NEIGHBOUR_SELECTION_MECHANISM = "NSM",
    K = "K",
    SPEED = "SPEED"