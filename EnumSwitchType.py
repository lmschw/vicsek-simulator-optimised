from enum import Enum

"""
Contains the different hyperparameters that can be used to switch between behaviours.
"""
class SwitchType(str, Enum):
    NOISE = "N",
    NEIGHBOUR_SELECTION_MODE = "MODE",
    K = "K"