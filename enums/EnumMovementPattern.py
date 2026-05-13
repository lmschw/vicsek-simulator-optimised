from enum import Enum

"""
Indicates how the input element should move
"""
class MovementPattern(Enum):
    STATIC = "static", "STATIC", # no movement
    RANDOM = "random", "RANDOM", # moves randomly
    PURSUIT_NEAREST = "p_nearest", "PURSUE NEAREST", # pursues the nearest particle
    #TODO: PURSUIT_DENSITY = "p_dense", "PURSUE DENSITY", # moves in the direction with the highest particle density
    #TODO: FLIGHT_NEAREST   = "f_nearest", "FLEE FROM NEAREST", # flees from the nearest particle
    #TODO: FLIGHT_DENSITY = "f_dense", "FLEE FROM DENSITY" # turns away from the direction with the highest particle density

    def __init__(self, val, label):
        self.val = val
        self.label = label