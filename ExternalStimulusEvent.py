import random
import math
import numpy as np
from scipy.spatial.transform import Rotation as R


from EnumDistributionType import DistributionType
from EnumEventEffect import EventEffect

import DefaultValues as dv
import ServiceOrientations
import ServicePreparation
import ServiceViscekHelper

class ExternalStimulusOrientationChangeEvent:
    # TODO refactor to allow all distributionTypes
    # TODO refactor to allow areas with a radius bigger than the radius of a particle, i.e. remove neighbourCells and determine all affected cells here
    """
    Representation of an event occurring at a specified time and place within the domain and affecting 
    a specified percentage of particles. After creation, the check()-method takes care of everything.
    """
    def __init__(self, timestep, domainSize, eventEffect, distributionType, areas=None, angle=None, noisePercentage=None):
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
        self.timestep = timestep
        self.angle = angle
        self.eventEffect = eventEffect
        self.distributionType = distributionType
        self.areas = areas
        self.domainSize = np.asarray(domainSize)
        self.noisePercentage = noisePercentage
        if self.noisePercentage != None:
            self.noise = ServicePreparation.getNoiseAmplitudeValueForPercentage(self.noisePercentage)

        if self.distributionType != DistributionType.GLOBAL and self.areas == None:
            raise Exception("Local effects require the area to be specified")
        
    def getShortPrintVersion(self):
        return f"t{self.timestep}e{self.eventEffect.val}p{self.percentage}a{self.angle}dt{self.distributionType.value}a{self.areas}"

    def getParameterSummary(self):
        summary = {"timestep": self.timestep,
            "percentage": self.percentage,
            "angle": self.angle,
            "eventEffect": self.eventEffect.name,
            "distributionType": self.distributionType.name,
            "areas": self.areas,
            "domainSize": self.domainSize.tolist(),
            }
        return summary

    def check(self, totalNumberOfParticles, currentTimestep, positions, orientations):
        """
        Checks if the event is triggered at the current timestep and executes it if relevant.

        Params:
            - totalNumberOfParticles (int): the total number of particles within the domain. Used to compute the number of affected particles
            - currentTimestep (int): the timestep within the experiment run to see if the event should be triggered
            - positions (array of tuples (x,y)): the position of every particle in the domain at the current timestep
            - orientations (array of tuples (u,v)): the orientation of every particle in the domain at the current timestep
            - switchValues (array of switchTypeValues): the switch type value of every particle in the domain at the current timestep
            - cells (array: [(minX, minY), (maxX, maxY)]): the cells within the cellbased domain
            - cellDims (tuple of floats): the dimensions of a cell (the same for all cells)
            - cellToParticleDistribution (dictionary {cellIdx: array of indices of all particles within the cell}): A dictionary containing the indices of all particles within each cell

        Returns:
            The orientations of all particles - altered if the event has taken place, unaltered otherwise.
        """
        if self.checkTimestep(currentTimestep):
            print(f"executing event at timestep {currentTimestep}")
            orientations = self.executeEvent(totalNumberOfParticles, positions, orientations)
        return orientations

    def checkTimestep(self, currentTimestep):
        """
        Checks if the event should be triggered.

        Params:
            - currentTimestep (int): the timestep within the experiment run to see if the event should be triggered

        Returns:
            A boolean representing whether or not the event should be triggered.
        """
        return self.timestep == currentTimestep
    
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
        np.append(posWithCenter, [self.areas[0][0], self.areas[0][1]])
        rij2 = ServiceViscekHelper.getDifferences(posWithCenter, self.domainSize)
        affected = (rij2 <= self.areas[0][2]**2)[-1]

        match self.eventEffect:
            case EventEffect.ALIGN_TO_FIXED_ANGLE:
                orientations[affected] = ServiceOrientations.computeUvCoordinates(self.angle)
            case EventEffect.ALIGN_TO_FIXED_ANGLE_NOISE:
                orientations[affected] = ServiceOrientations.computeUvCoordinates(self.angle)
                orientations[affected] = self.__applyNoiseDistribution(orientations[affected])
            #case EventEffect.AWAY_FROM_ORIGIN:
                #orientations = self.computeAwayFromOrigin(positions)
            #case EventEffect.RANDOM:
                #orientations = self.__getRandomOrientation()
        orientations = ServiceOrientations.normalizeOrientations(orientations)
        return orientations


    def __computeFixedAngleTurn(self, orientation):
        """
        Determines the new uv-coordinates after turning the particle by the specified angle.
        The new angle is the equivalent of the old angle plus the angle specified by the event.

        Params:
            orientation ([U,V]): the current orientation of a single particle
        
        Returns:
            The new uv-coordinates for the orientation of the particle.
        """
        previousAngle = ServiceOrientations.computeAngleForOrientation(orientation)

        # add the event angle to the current angle
        newAngle = (previousAngle + self.angle) % (2 *np.pi)

        return ServiceOrientations.computeUvCoordinates(newAngle)
    
    def __applyNoiseDistribution(self, orientations):
        return orientations + np.random.normal(scale=self.noise, size=(len(orientations), len(self.domainSize)))


    def computeAwayFromOrigin(self, position):
        """
        Computes the (u,v)-coordinates for the orientation after turning away from the point of origin.

        Params:
            - position ([X,Y]): the position of the current particle that should turn away from the point of origin

        Returns:
            [U,V]-coordinates representing the new orientation of the current particle.
        """
        angle = self.__computeAngleWithRegardToOrigin(position)
        angle = ServiceOrientations.normaliseAngle(angle)
        return ServiceOrientations.computeUvCoordinates(angle)

    def __computeTowardsOrigin(self, position):
        """
        Computes the (u,v)-coordinates for the orientation after turning towards the point of origin.

        Params:
            - position ([X,Y]): the position of the current particle that should turn towards the point of origin

        Returns:
            [U,V]-coordinates representing the new orientation of the current particle.
        """
        angle = self.__computeAngleWithRegardToOrigin(position)
        angle = ServiceOrientations.normaliseAngle(angle)
        return ServiceOrientations.computeUvCoordinates(angle)

    def __computeAngleWithRegardToOrigin(self, position):
        """
        Computes the angle between the position of the current particle and the point of origin of the event.

        Params:
            - position ([X,Y]): the position of the current particle that should turn towards the point of origin

        Returns:
            The angle in radians between the two points.
        """
        orientationFromOrigin = position - self.getOriginPoint()
        angleRadian = ServiceOrientations.computeAngleForOrientation(orientationFromOrigin)
        return angleRadian

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

    
    # def __getRandomOrientation(self):
    #     """
    #     Selects a random orientation.

    #     Returns:
    #         A random orientation in [U,V]-coordinates.
    #     """
    #     return ServiceVicsekHelper.normalizeOrientations(np.random.rand(1, len(self.domainSize))-0.5)