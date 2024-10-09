import random
import math
import numpy as np
from scipy.spatial.transform import Rotation as R


from enums.EnumDistributionType import DistributionType
from enums.EnumEventEffect import EventEffect

import events.BaseEvent as BaseEvent

import DefaultValues as dv
import services.ServiceOrientations as ServiceOrientations
import services.ServicePreparation as ServicePreparation
import services.ServiceVicsekHelper as ServiceVicsekHelper

class ExternalStimulusOrientationChangeEvent(BaseEvent.BaseEvent):
    # TODO refactor to allow areas with a radius bigger than the radius of a particle, i.e. remove neighbourCells and determine all affected cells here
    """
    Representation of an event occurring at a specified time and place within the domain and affecting 
    a specified percentage of particles. After creation, the check()-method takes care of everything.
    """
    def __init__(self, startTimestep, duration, domainSize, eventEffect, distributionType, areas=None, radius=None, angle=None, noisePercentage=None):
        """
        Creates an external stimulus event that affects part of the swarm at a given timestep.

        Params:
            - timestep (int): the timestep at which the stimulus is presented and affects the swarm
            - percentage (float, range: 0-100): how many percent of the swarm is directly affected by the event
            - angle (int, range: 1-359): how much the orientation of the affected particles is changed in a counterclockwise manner
            - eventEffect (EnumEventEffect): how the orientations should be affected
            - distributionType (EnumDistributionType) [optional]: how the directly affected particles are distributed, i.e. if the event occurs globally or locally
            - areas ([(centerXCoordinate, centerYCoordinate, radius)]) [optional]: list of areas in which the event takes effect. Should be specified if the distributionType is not GLOBAL and match the DistributionType
            - domainSize (tuple of floats) [optional]: the size of the domain
            - targetSwitchValue (switchTypeValue) [optional]: the value that every affected particle should select
            
        Returns:
            No return.
        """
        super().__init__(startTimestep=startTimestep, duration=duration, domainSize=domainSize, eventEffect=eventEffect, noisePercentage=noisePercentage)
        self.angle = angle
        self.distributionType = distributionType
        self.areas = areas

        match self.distributionType:
            case DistributionType.GLOBAL:
                self.radius = (domainSize[0] * domainSize[1]) /np.pi
            case DistributionType.LOCAL_SINGLE_SITE:
                self.radius = self.areas[0][2]

        if radius:
            self.radius = radius

        if self.distributionType != DistributionType.GLOBAL and self.areas == None:
            raise Exception("Local effects require the area to be specified")
        
    def getShortPrintVersion(self):
        return f"t{self.startTimestep}d{self.duration}e{self.eventEffect.val}a{self.angle}dt{self.distributionType.value}a{self.areas}"

    def getParameterSummary(self):
        summary = {"startTimestep": self.startTimestep,
            "duration": self.duration,
            "angle": self.angle,
            "eventEffect": self.eventEffect.name,
            "distributionType": self.distributionType.name,
            "areas": self.areas,
            "domainSize": self.domainSize.tolist(),
            }
        return summary
    
    def executeEvent(self, totalNumberOfParticles, positions, orientations):
        """
        Executes the event.

        Params:
            - totalNumberOfParticles (int): the total number of particles within the domain. Used to compute the number of affected particles
            - positions (array of tuples (x,y)): the position of every particle in the domain at the current timestep
            - orientations (array of tuples (u,v)): the orientation of every particle in the domain at the current timestep
            - switchValues (array of switchTypeValues): the switch type value of every particle in the domain at the current timestep
            - cells (array: [(minX, minY), (maxX, maxY)]): the cells within the cellbased domain
            - cellDims (tuple of floats): the dimensions of a cell (the same for all cells)
            - cellToParticleDistribution (dictionary {cellIdx: array of indices of all particles within the cell}): A dictionary containing the indices of all particles within each cell

        Returns:
            The orientations, switchTypeValues of all particles after the event has been executed as well as a list containing the indices of all affected particles.
        """
        posWithCenter = np.copy(positions)
        np.append(posWithCenter, self.getOriginPoint())
        rij2 = ServiceVicsekHelper.getDifferences(posWithCenter, self.domainSize)
        affected = (rij2 <= self.radius**2)[-1]

        match self.eventEffect:
            case EventEffect.ALIGN_TO_FIXED_ANGLE:
                orientations[affected] = ServiceOrientations.computeUvCoordinates(self.angle)
            case EventEffect.ALIGN_TO_FIXED_ANGLE_NOISE:
                orientations[affected] = ServiceOrientations.computeUvCoordinates(self.angle)
                orientations[affected] = self.__applyNoiseDistribution(orientations[affected])
            case EventEffect.AWAY_FROM_ORIGIN:
                orientations[affected] = self.computeAwayFromOrigin(positions[affected])
            case EventEffect.RANDOM:
                orientations[affected] = self.__getRandomOrientations(np.count_nonzero(affected))
        orientations = ServiceOrientations.normalizeOrientations(orientations)
        return orientations
    
    def __applyNoiseDistribution(self, orientations):
        return orientations + np.random.normal(scale=self.noise, size=(len(orientations), len(self.domainSize)))


    def computeAwayFromOrigin(self, positions):
        """
        Computes the (u,v)-coordinates for the orientation after turning away from the point of origin.

        Params:
            - position ([X,Y]): the position of the current particle that should turn away from the point of origin

        Returns:
            [U,V]-coordinates representing the new orientation of the current particle.
        """
        angles = self.__computeAngleWithRegardToOrigin(positions)
        #angles = ServiceOrientations.normaliseAngles(angles)
        return ServiceOrientations.computeUvCoordinatesForList(angles)

    def __computeAngleWithRegardToOrigin(self, positions):
        """
        Computes the angle between the position of the current particle and the point of origin of the event.

        Params:
            - position ([X,Y]): the position of the current particle that should turn towards the point of origin

        Returns:
            The angle in radians between the two points.
        """
        orientationFromOrigin = positions - self.getOriginPoint()
        anglesRadian = ServiceOrientations.computeAnglesForOrientations(orientationFromOrigin)
        return anglesRadian

    def getOriginPoint(self):
        """
        Determines the point of origin of the event.

        Returns:
            The point of origin of the event in [X,Y]-coordinates.
        """
        match self.distributionType:
            case DistributionType.GLOBAL:
                origin = (self.domainSize[0]/2, self.domainSize[1]/2)
            case DistributionType.LOCAL_SINGLE_SITE:
                origin = self.areas[0][:2]
        return origin

    
    def __getRandomOrientations(self, numAffectedParticles):
        """
        Selects a random orientation.

        Returns:
            A random orientation in [U,V]-coordinates.
        """
        return ServiceOrientations.normalizeOrientations(np.random.rand(numAffectedParticles, len(self.domainSize))-0.5)