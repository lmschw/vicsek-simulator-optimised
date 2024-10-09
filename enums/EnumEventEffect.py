from enum import Enum

"""
Indicates what should happen to the particles' orientation
"""
class EventEffect(Enum):
    ALIGN_TO_FIXED_ANGLE = "align_fixed", "DISTANT", # the same angle is imposed on all particles
    ALIGN_TO_FIXED_ANGLE_NOISE = "align_noise", "DISTANT WITH NOISE", # the same angle is imposed on all particles but subject to noise
    AWAY_FROM_ORIGIN = "origin_away", "PREDATOR", # turn away from the point of origin of the event
    RANDOM = "random", "RANDOM" # sets the orientations to a random value, intended for baseline

    def __init__(self, val, label):
        self.val = val
        self.label = label

class InternalEventEffect(Enum):
    ALIGN_TO_FIXED_ANGLE = "align_fixed", "DISTANT"
    REINFORCE_RANDOM_ANGLE = "reinforce_random", "REINFORCE RANDOM"

    def __init__(self, val, label):
        self.val = val
        self.label = label
