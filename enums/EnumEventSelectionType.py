from enum import Enum

"""
Indicates how the particles should be distributed spatially when selecting which particles are affected by 
an event.
"""
class EventSelectionType(str, Enum):
    NEAREST_DISTANCE = "ND",
    RANDOM = "R"