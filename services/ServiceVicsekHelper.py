import numpy as np

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