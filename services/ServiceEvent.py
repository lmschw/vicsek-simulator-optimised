import numpy as np

from enums.EnumEventSelectionType import EventSelectionType
import services.ServiceVicsekHelper as svh


def findEventCandidates(totalNumberOfParticles, positions, originPoint, domainSize, radius):
    posWithCenter = np.zeros((totalNumberOfParticles+1, 2))
    posWithCenter[:-1] = positions
    posWithCenter[-1] = originPoint
    rij2 = svh.getDifferences(posWithCenter, domainSize)
    relevantDistances = rij2[-1][:-1] # only the comps to the origin and without the origin point
    candidates = (relevantDistances <= radius**2)
    return candidates

def selectAffected(eventSelectionType, totalNumberOfParticles, positions, originPoint, domainSize, radius, numberOfAffected=None):
    """
    Determines which particles are affected by the event.

    Params:
        - candidates (array of boolean): which particles are within range, i.e. within the event radius
        - rij2 (array of floats): the distance squared of every particle to the event focus point

    Returns:
        Array of booleans representing which particles are affected by the event.
    """
    rij2 = svh.getDifferences(positions, domainSize)
    candidates = findEventCandidates(totalNumberOfParticles=totalNumberOfParticles,
                                     positions=positions,
                                     originPoint=originPoint,
                                     domainSize=domainSize,
                                     radius=radius)
    if numberOfAffected == None:
        numberOfAffected = len(candidates.nonzero()[0])
    else:
        numberOfAffected = numberOfAffected

    preselection = candidates # default case, we take all the candidates
    match eventSelectionType:
        case EventSelectionType.NEAREST_DISTANCE:
            indices = np.argsort(rij2)[:numberOfAffected]
            preselection = np.full(len(candidates), False)
            preselection[indices] = True
        case EventSelectionType.RANDOM:
            indices = candidates.nonzero()[0]
            selectedIndices = np.random.choice(indices, numberOfAffected, replace=False)
            preselection = np.full(len(candidates), False)
            preselection[selectedIndices] = True
    return candidates & preselection