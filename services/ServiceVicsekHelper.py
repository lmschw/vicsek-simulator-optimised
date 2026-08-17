import numpy as np
import random

import services.ServiceVision as ServiceVision

def getDifferences(array, domainSize):
    """
    Computes the differences between all individuals for the values provided by the array.

    Params:
        - array (array of floats): the values to be compared
        - domainSize (array of floats): the size of the domain, used to apply the minimum image
          convention (periodic wraparound) so that distances reflect the shortest path across the
          periodic boundary, consistent with how positions themselves wrap around it.

    Returns:
        An array of arrays of floats containing the difference between each pair of values.
    """
    rij=array[:,np.newaxis,:]-array
    rij = rij - domainSize*np.rint(rij/domainSize) #minimum image convention
    return np.sum(rij**2,axis=2)

def getOrientationDifferences(orientations, domainSize):
    """
    Helper method to gloss over identical differences implementation for position and orientation. 
    """
    return getDifferences(orientations, domainSize)

def getPositionDifferences(positions, domainSize):
    """
    Helper method to gloss over identical differences implementation for position and orientation. 
    """
    return getDifferences(positions, domainSize)

def getNeighbours(positions, domainSize, radius):
    """
    Determines all the neighbours for each individual.

    Params:
        - positions (array of floats): the position of every individual at the current timestep

    Returns:
        An array of arrays of booleans representing whether or not any two individuals are neighbours
    """
    rij2 = getPositionDifferences(positions, domainSize)
    return (rij2 <= radius**2)

def getNeighboursWithLimitedVision(positions, orientations, domainSize, radius, degreesOfVision, posDiff=None):
    """
    posDiff (array of arrays of floats) [optional]: the squared position differences between every
    pair of individuals, i.e. the result of getPositionDifferences(positions, domainSize). Callers
    that already have this on hand (e.g. because they need it again right after for neighbour
    selection) can pass it in to avoid recomputing the same O(n^2) distance matrix twice. If omitted,
    it is computed internally exactly as before.
    """
    if posDiff is None:
        posDiff = getPositionDifferences(positions, domainSize)
    candidates = (posDiff <= radius**2)
    if degreesOfVision >= 2*np.pi:
        # full circle: every candidate is in view, so skip the min/max-angle vision check entirely.
        # Computing it anyway is not just wasted work: determineMinMaxAngleOfVision derives minAngle
        # and maxAngle from the same current angle via two independent (and independently rounded)
        # normaliseAngles() calls, so for a 2*pi field of view they are mathematically but not always
        # bit-identical. isInFieldOfVision() treats any minAngle != maxAngle as a real (if tiny)
        # excluded slice rather than the intended "everything is visible", which silently dropped
        # genuine in-radius neighbours whenever that rounding mismatch occurred.
        combined = candidates
    else:
        minAngles, maxAngles = ServiceVision.determineMinMaxAngleOfVision(orientations=orientations, degreesOfVision=degreesOfVision)
        inFieldOfVision = ServiceVision.isInFieldOfVision(positions=positions, minAngles=minAngles, maxAngles=maxAngles)
        combined = candidates & inFieldOfVision
    np.fill_diagonal(combined, True)
    return combined

def padArray(a, n, kMin, kMax, paddingValue=-1):
    if kMax > len(a[0]):
        minusDiff = np.full((n,kMax-kMin), paddingValue)
        return np.concatenate((a, minusDiff), axis=1)
    return a

def padWithRepetition(vector, pad_width, iaxis, kwargs):
    if pad_width == (0, 0):
        values = vector
    else:
        values = list(vector[pad_width[0]:-pad_width[1]])
    randomValues = np.random.choice(values, len(vector))
    #c =  np.where((vector == 0), randomValues, vector)
    #vector = c
    vector[:pad_width[0]] = randomValues[:pad_width[0]]
    vector[-pad_width[1]:] = randomValues[-pad_width[1]:]

def getIndicesForTrueValues(a, paddingType='constant', paddingValue=-1):
    indices = np.transpose(np.nonzero(a))
    perRow = np.full(len(a), None)
    maxLength = 0
    for idx in indices:
        if perRow[idx[0]] == None:
            perRow[idx[0]] = [idx[1]]
        else:
            perRow[idx[0]].append(idx[1])
        if len(perRow[idx[0]]) > maxLength:
            maxLength = len(perRow[idx[0]])
    result = []
    for rowIdx in range(len(a)):
        if perRow[rowIdx] == None:
            result.append(np.full(maxLength, paddingValue))
        else:
            #result.append(np.pad(perRow[rowIdx], maxLength, ))
            pr = np.array(perRow[rowIdx])
            if paddingType == 'constant':
                result.append(np.pad(pr, ((0, maxLength-pr.shape[0])), 'constant', constant_values=paddingValue))
            elif paddingType == 'repetition':
                result.append(np.pad(pr, (0, maxLength-pr.shape[0]), padWithRepetition))
    
    return np.array(result)

def revertTimeDelayedChanges(t, oldValues, newValues, activationTimeDelays):
    vals = np.where((np.array([t % activationTimeDelays == 0, t % activationTimeDelays == 0]).T), newValues, oldValues)
    return vals