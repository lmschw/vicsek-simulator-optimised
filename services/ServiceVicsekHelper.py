import numpy as np
import random

import services.ServiceVision as ServiceVision

def getDifferences(array, domainSize):
    """
    Computes the differences between all individuals for the values provided by the array.

    Params:
        - array (array of floats): the values to be compared

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

def getNeighboursWithLimitedVision(positions, orientations, domainSize, radius, degreesOfVision):
    candidates = getNeighbours(positions=positions, domainSize=domainSize, radius=radius)
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