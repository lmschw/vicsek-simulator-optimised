from enum import Enum

"""
Indicates how the particles should be selected for the event from all available candidates.
"""
class EventSelectionType(str, Enum):
    NEAREST_DISTANCE = "ND",
    RANDOM = "R"