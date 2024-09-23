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