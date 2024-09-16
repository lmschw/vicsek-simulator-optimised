from enum import Enum

"""
Contains all the different ways of selecting a subset of neighbours from all possible neighbours.
"""
class NeighbourSelectionMechanism(str, Enum):
    RANDOM = "R",
    NEAREST = "N",
    FARTHEST = "F",
    LEAST_ORIENTATION_DIFFERENCE = "LOD",
    HIGHEST_ORIENTATION_DIFFERENCE = "HOD",
    ALL = "A"