import random
import math
import numpy as np
from scipy.spatial.transform import Rotation as R


from enums.EnumDistributionType import DistributionType
from enums.EnumEventEffect import EventEffect
from enums.EnumEventSelectionType import EventSelectionType
from enums.EnumColourType import ColourType

import events.BaseEvent as BaseEvent

import DefaultValues as dv
import services.ServiceOrientations as ServiceOrientations
import services.ServicePreparation as ServicePreparation
import services.ServiceVicsekHelper as ServiceVicsekHelper
import services.ServiceEvent as se

class ExternalStimulusOrientationChangeEvent(BaseEvent.BaseEvent):
    # TODO refactor to allow areas with a radius bigger than the radius of a particle, i.e. remove neighbourCells and determine all affected cells here
    """
    Representation of an event occurring at a specified time and place within the domain and affecting 
    a specified percentage of particles. After creation, the check()-method takes care of everything.
    """
    def __init__(self, startTimestep, duration, domainSize, eventEffect, distributionType, areas=None, radius=None, numberOfAffected=None, eventSelectionType=None, 
                 angle=None, noisePercentage=None, blockValues=False):
        """
        Creates an external stimulus event that affects part of the swarm at a given timestep.

        Params:
            - startTimestep (int): the first timestep at which the stimulus is presented and affects the swarm
            - duration (int): the number of timesteps during which the stimulus is present and affects the swarm
            - domainSize (tuple of floats): the size of the domain
            - eventEffect (EnumEventEffect): how the orientations should be affected
            - distributionType (DistributionType): whether the event is global or local in nature
            - areas (list of arrays containing [x_center, y_center, radius]) [optional]: where the event is supposed to take effect. Required for Local events
            - radius (float) [optional]: the event radius
            - noisePercentage (float, range: 0-100) [optional]: how much noise is added to the orientation determined by the event (only works for certain events)
            - blockValues (boolean) [optional]: whether the values (nsm, k, speed etc.) should be blocked after the update. By default False
            
        Returns:
            No return.
        """
        super().__init__(startTimestep=startTimestep, duration=duration, domainSize=domainSize, eventEffect=eventEffect, noisePercentage=noisePercentage, blockValues=blockValues)
        self.angle = angle
        self.distributionType = distributionType
        self.areas = areas
        self.numberOfAffected = numberOfAffected
        self.eventSelectionType = eventSelectionType

        match self.distributionType:
            case DistributionType.GLOBAL:
                self.radius = (domainSize[0] * domainSize[1]) /np.pi
            case DistributionType.LOCAL_SINGLE_SITE:
                self.radius = self.areas[0][2]

        if radius:
            self.radius = radius

        if self.distributionType != DistributionType.GLOBAL and self.areas == None:
            raise Exception("Local effects require the area to be specified")
        
        if self.numberOfAffected and self.radius:
            print("Radius is set. The full number of affected particles may not be reached.")
        
    def getShortPrintVersion(self):
        return f"t{self.startTimestep}d{self.duration}e{self.eventEffect.val}a{self.angle}dt{self.distributionType.value}a{self.areas}"

    def getParameterSummary(self):
        summary = super().getParameterSummary()
        summary["angle"] = self.angle
        summary["distributionType"] = self.distributionType.name
        summary["areas"] = self.areas
        summary["radius"] = self.radius
        summary["numberOfAffected"] = self.numberOfAffected
        if self.eventSelectionType:
            summary["eventSelectionType"] = self.eventSelectionType.value
        return summary
    
    def executeEvent(self, totalNumberOfParticles, positions, orientations, nsms, ks, speeds, dt=None, colourType=None):
        """
        Executes the event.

        Params:
            - totalNumberOfParticles (int): the total number of particles within the domain. Used to compute the number of affected particles
            - positions (array of tuples (x,y)): the position of every particle in the domain at the current timestep
            - orientations (array of tuples (u,v)): the orientation of every particle in the domain at the current timestep
            - nsms (array of NeighbourSelectionMechanisms): the neighbour selection mechanism currently selected by each individual at the current timestep
            - ks (array of int): the number of neighbours currently selected by each individual at the current timestep
            - speeds (array of float): the speed of every particle at the current timestep
            - dt (float) [optional]: the difference between the timesteps
            - colourType (ColourType) [optional]: if and how particles should be encoded for colour for future video rendering

        Returns:
            The orientations, neighbour selection mechanisms, ks, speeds, blockedness and colour of all particles after the event has been executed.
        """

        affected = se.selectAffected(eventSelectionType=self.eventSelectionType,
                                     totalNumberOfParticles=totalNumberOfParticles,
                                     positions=positions,
                                     originPoint=self.getOriginPoint(),
                                     domainSize=self.domainSize,
                                     radius=self.radius,
                                     numberOfAffected=self.numberOfAffected)

        match self.eventEffect:
            case EventEffect.ALIGN_TO_FIXED_ANGLE:
                orientations[affected] = ServiceOrientations.computeUvCoordinates(self.angle)
            case EventEffect.ALIGN_TO_FIXED_ANGLE_NOISE:
                orientations[affected] = ServiceOrientations.computeUvCoordinates(self.angle)
                orientations[affected] = self.applyNoiseDistribution(orientations[affected])
            case EventEffect.AWAY_FROM_ORIGIN:
                orientations[affected] = self.computeAwayFromOrigin(positions[affected])
            case EventEffect.RANDOM:
                orientations[affected] = self.__getRandomOrientations(np.count_nonzero(affected))
        orientations = ServiceOrientations.normalizeOrientations(orientations)

        colours = self.getColours(colourType=colourType, affected=affected, totalNumberOfParticles=totalNumberOfParticles)

        return orientations, nsms, ks, speeds, affected, colours # external events do not directly impact the values

    def computeAwayFromOrigin(self, positions):
        """
        Computes the (u,v)-coordinates for the orientation after turning away from the point of origin.

        Params:
            - position ([X,Y]): the position of the current particle that should turn away from the point of origin

        Returns:
            [U,V]-coordinates representing the new orientation of the current particle.
        """
        angles = self.__computeAngleWithRegardToOrigin(positions)
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

        Params:
            None

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

        Params:
            - numAffectedParticles (int): the number of particles affected by the event

        Returns:
            A random orientation in [U,V]-coordinates.
        """
        return ServiceOrientations.normalizeOrientations(np.random.rand(numAffectedParticles, len(self.domainSize))-0.5)