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

def computeUvCoordinatesForList(angles):
    """
    Computes the (u,v)-coordinates based on the angle.

    Params:
        - angle (float): the angle in radians

    Returns:
        An array containing the [u, v]-coordinates corresponding to the angle.
    """
    # compute the uv-coordinates
    U = np.cos(angles)
    V = np.sin(angles)
    
    return np.column_stack((U,V))

def computeAngleForOrientation(orientation):
    """
    Computes the angle in radians based on the (u,v)-coordinates of the current orientation.

    Params:
        - orientation (array of floats): the current orientation in (u,v)-coordinates

    Returns:
        A float representin the angle in radians.
    """
    return np.arctan2(orientation[1], orientation[0])

def computeAnglesForOrientations(orientations):
    """
    Computes the angle in radians based on the (u,v)-coordinates of the current orientation.

    Params:
        - orientation (array of floats): the current orientation in (u,v)-coordinates

    Returns:
        A float representin the angle in radians.
    """
    return np.arctan2(orientations[:, 1], orientations[:, 0])

def computeAnglesWithRespectToFocusPoint(positions, focusPoint):
    return np.arctan2(positions[:, 1]-focusPoint[1], positions[:, 0]-focusPoint[0])