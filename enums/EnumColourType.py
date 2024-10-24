from enum import Enum

"""
Indicates how the particles should be distributed spatially when selecting which particles are affected by 
an event.
"""
class ColourType(str, Enum):
    EXAMPLE = "EX",
    AFFECTED = "AF"