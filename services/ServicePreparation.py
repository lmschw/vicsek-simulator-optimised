import numpy as np
import random
import math

import services.ServiceOrientations as ServiceOrientations

"""
Service containing static methods to prepare simulations.
"""

def getDomainSizeForConstantDensity(density, numberOfParticles):
    """
    Computes the domain size to keep the density constant for the supplied number of particles.
    Density formula: "density" = "number of particles" / "domain area"

    Parameters:
        - density (float): the desired constant density of the domain
        - numberOfParticles (int): the number of particles to be placed in the domain

    Returns:
        A tuple containing the x and y dimensions of the domain size that corresponds to the density.
    """
    area = numberOfParticles / density
    return (np.sqrt(area), np.sqrt(area))

def getNumberOfParticlesForConstantDensity(density, domainSize):
    """
    Computes the number of particles to keep the density constant for the supplied domain size.
    Density formula: "density" = "number of particles" / "domain area"

    Parameters:
        - density (float): the desired constant density of the domain
        - domainSize (tuple): tuple containing the x and y dimensions of the domain size

    Returns:
        The number of particles to be placed in the domain that corresponds to the density.
    """
    return int(density * (domainSize[0] * domainSize[1])) # density * area

def getDensity(domainSize, numberOfParticles):
    """
    Computes the density of a given system.
    Density formula: "density" = "number of particles" / "domain area"

    Parameters:
        - domainSize (tuple): tuple containing the x and y dimensions of the domain size
        - numberOfParticles (int): the number of particles to be placed in the domain

    Returns:
        The density of the system as a float.
    """
    return numberOfParticles / (domainSize[0] * domainSize[1]) # n / area

def getNoiseAmplitudeValueForPercentage(percentage):
    """
    Paramters:
        - percentage (int, 1-100)
    """
    return 2 * np.pi * (percentage/100)

def getRadiusToSeeOnAverageNNeighbours(n, density):
    """
    Computes the radius that will ensure that every particle sees at least n other particles
    if the density is equally distributed in the whole domain.

    Params:
        - n (int): the number of neighbours that the particle should be able to see
        - density (float): the domain density (assumed to be equally distributed)

    Returns:
        An integer representing the perception radius of each particle
    """
    area = n/density
    return np.ceil(np.sqrt(area))

def createOrderedInitialDistributionEquidistancedIndividual(startSwitchTypeValue, domainSize, numberOfParticles, angleX=None, angleY=None):
    """
    Creates an ordered, equidistanced initial distribution of particles in a domain ready for use in individual decision scenarios. 
    The particles are placed in a grid-like fashion. The orientation of the particles is random unless specified
    but always the same for all particles.

    Parameters:
        - domainSize (tuple): tuple containing the x and y dimensions of the domain size
        - numberOfParticles (int): the number of particles to be placed in the domain
        - angleX (float [0,1)): first angle component to specify the orientation of all particles
        - angleY (float [0,1)): second angle component to specify the orientation of all particles

    Returns:
        Positions and orientations for all particles within the domain. Can be used as the initial state of a Vicsek simulation.
    """
    positions, orientations = createOrderedInitialDistributionEquidistanced(domainSize, numberOfParticles, angleX, angleY)
    switchTypeValues = numberOfParticles * [startSwitchTypeValue]
    return positions, orientations, switchTypeValues

def createOrderedInitialDistributionEquidistanced(domainSize, numberOfParticles, angleX=None, angleY=None):
    """
    Creates an ordered, equidistanced initial distribution of particles in a domain. 
    The particles are placed in a grid-like fashion. The orientation of the particles is random unless specified
    but always the same for all particles.

    Parameters:
        - domainSize (tuple): tuple containing the x and y dimensions of the domain size
        - numberOfParticles (int): the number of particles to be placed in the domain
        - angleX (float [0,1)): first angle component to specify the orientation of all particles
        - angleY (float [0,1)): second angle component to specify the orientation of all particles

    Returns:
        Positions and orientations for all particles within the domain. Can be used as the initial state of a Vicsek simulation.
    """
    # choose random angle for orientations
    if angleX is None:
        angleX = random.random()
    if angleY is None:
        angleY = random.random()

    # prepare the distribution for the positions
    xLength = domainSize[0]
    yLength = domainSize[1]
    
    area = xLength * yLength
    pointArea = area / numberOfParticles
    length = np.sqrt(pointArea)

    # initialise the initialState components
    positions = np.zeros((numberOfParticles, 2))
    orientations = np.zeros((numberOfParticles, 2))

    # set the orientation for all particles
    orientations[:, 0] = angleX
    orientations[:, 1] = angleY

    counter = 0
    # set the position of every particle
    for x in np.arange(length/2, xLength, length):
        for y in np.arange(length/2, yLength, length):
            if counter < numberOfParticles:
                positions[counter] = [x,y]
            counter += 1

    return positions, orientations


def createOrderedInitialDistributionEquidistancedForLowNumbers(domainSize, numberOfParticles, angleX=None, angleY=None):
    """
    Creates an ordered, equidistanced initial distribution of particles in a domain. 
    The particles are placed in a grid-like fashion. The orientation of the particles is random unless specified
    but always the same for all particles.

    Parameters:
        - domainSize (tuple): tuple containing the x and y dimensions of the domain size
        - numberOfParticles (int): the number of particles to be placed in the domain
        - angleX (float [0,1)): first angle component to specify the orientation of all particles
        - angleY (float [0,1)): second angle component to specify the orientation of all particles

    Returns:
        Positions and orientations for all particles within the domain. Can be used as the initial state of a Vicsek simulation.

    """
    # choose random angle for orientations
    if angleX is None:
        angleX = random.random()
    if angleY is None:
        angleY = random.random()

    # prepare the distribution for the positions
    xLength = domainSize[0]
    yLength = domainSize[1]
    
    area = xLength * yLength
    pointArea = area / numberOfParticles
    length = np.sqrt(pointArea)

    # initialise the initialState components
    positions = np.zeros((numberOfParticles, 2))
    orientations = np.zeros((numberOfParticles, 2))

    # set the orientation for all particles
    orientations[:, 0] = angleX
    orientations[:, 1] = angleY

    counter = 0
    # set the position of every particle
    for x in np.arange(length/2, xLength, length):
        positions[counter] = [x,x]
        counter += 1

    return positions, orientations


def createInitialStateInCircle(domainSize, center, radius, numberOfParticles, isOrdered, startSwitchTypeValue=None):
    positions = []
    for pos in range(numberOfParticles):
        """
        a = random.randint(center[0], center[1]) * 2 * math.pi
        r = 1 * math.sqrt(random.randint(center[0],center[1]))
        x = r * math.cos(a) + center[0]
        y = r * math.sin(a) + center[1]
        """
        r_squared, theta = [random.randint(0,radius**2), 2*math.pi*random.random()]
        x = center[0] + math.sqrt(r_squared)*math.cos(theta) 
        y = center[1] + math.sqrt(r_squared)*math.sin(theta)
        positions.append(np.array([x,y]))
    positions = np.array(positions)
    if isOrdered:
        baseOrientation = np.random.rand(1, len(domainSize))-0.5
        orientations = numberOfParticles * baseOrientation
    else:
        orientations = np.random.rand(numberOfParticles, len(domainSize))-0.5
    orientations = ServiceOrientations.normalizeOrientations(orientations)
    switchTypeValues = numberOfParticles * [startSwitchTypeValue]

    return positions, orientations, switchTypeValues