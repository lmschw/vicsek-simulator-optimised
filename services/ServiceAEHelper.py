import numpy as np
import random

import services.ServiceVision as ServiceVision

def getDifferences(array):
    """
    Computes the differences between all individuals for the values provided by the array.

    Params:
        - array (array of floats): the values to be compared

    Returns:
        An array of arrays of floats containing the difference between each pair of values.
    """
    rij=array[:,np.newaxis,:]-array   
    return np.sum(rij**2,axis=2)

def getOrientationDifferences(orientations):
    """
    Helper method to gloss over identical differences implementation for position and orientation. 
    """
    return getDifferences(orientations)

def getPositionDifferences(positions):
    """
    Helper method to gloss over identical differences implementation for position and orientation. 
    """
    return getDifferences(positions)

def getNeighbours(positions, radius):
    """
    Determines all the neighbours for each individual.

    Params:
        - positions (array of floats): the position of every individual at the current timestep

    Returns:
        An array of arrays of booleans representing whether or not any two individuals are neighbours
    """
    rij2 = getPositionDifferences(positions)
    return (rij2 <= radius**2)

def getNeighboursWithLimitedVision(positions, orientations, radius, degreesOfVision):
    candidates = getNeighbours(positions=positions, radius=radius)
    minAngles, maxAngles = ServiceVision.determineMinMaxAngleOfVision(orientations=orientations, degreesOfVision=degreesOfVision)
    inFieldOfVision = ServiceVision.isInFieldOfVision(positions=positions, minAngles=minAngles, maxAngles=maxAngles)

    combined = candidates & inFieldOfVision
    np.fill_diagonal(combined, True)
    return combined