import numpy as np

"""
Service that includes general methods used by the Vicsek model.
"""

def normalizeOrientations(orientations):
    """
    Normalises the orientations of all particles for the current time step

    Parameters:
        - orientations (array): The current orientations of all particles

    Returns:
        The normalised orientations of all particles as an array.
    """
    return orientations/(np.sqrt(np.sum(orientations**2,axis=1))[:,np.newaxis])

def computeUvCoordinates(angle):
    """
    Computes the (u,v)-coordinates based on the angle.

    Params:
        - angle (float): the angle in radians

    Returns:
        An array containing the [u, v]-coordinates corresponding to the angle.
    """
    # compute the uv-coordinates
    U = np.cos(angle)
    V = np.sin(angle)
    
    return [U,V]