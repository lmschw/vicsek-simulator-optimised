from enum import Enum

"""
Indicates how the particles should be distributed spatially when selecting which particles are affected by 
an event.
"""
class DistributionType(str, Enum):
    GLOBAL = "G",
    LOCAL_SINGLE_SITE = "LSS"