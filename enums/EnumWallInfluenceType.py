from enum import Enum

"""
Indicates where in its area of influence (the walls) an event will take effect
"""
class WallInfluenceType(str, Enum):
    FULL_AREA = "full",
    EXCEPT_NEAR_BORDER = "not_near",
    CLOSE_TO_BORDER = "close"
    