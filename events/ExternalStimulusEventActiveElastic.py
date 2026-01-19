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

class ExternalStimulusActiveElasticEvent(BaseEvent.BaseEvent):
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
    
    def check(self, totalNumberOfParticles, currentTimestep, positions, orientations, nsms, ks, speeds, dt=None, activationTimeDelays=None, 
              isActivationTimeDelayRelevantForEvent=False, colourType=None):
        """
        Checks if the event is triggered at the current timestep and executes it if relevant.

        Params:
            - totalNumberOfParticles (int): the total number of particles within the domain. Used to compute the number of affected particles
            - currentTimestep (int): the timestep within the experiment run to see if the event should be triggered
            - positions (array of tuples (x,y)): the position of every particle in the domain at the current timestep
            - orientations (array of tuples (u,v)): the orientation of every particle in the domain at the current timestep
            - nsms (array of NeighbourSelectionMechanisms): the neighbour selection mechanism currently selected by each individual at the current timestep
            - ks (array of int): the number of neighbours currently selected by each individual at the current timestep
            - speeds (array of float): the speed of every particle at the current timestep
            - dt (float) [optional]: the difference between the timesteps
            - activationTimeDelays (array of int) [optional]: the time delay for the updates of each individual
            - isActivationTimeDelayRelevantForEvent (boolean) [optional]: whether the event can affect particles that may not be ready to update due to a time delay. They may still be selected but will retain their current values
            - colourType (ColourType) [optional]: if and how particles should be encoded for colour for future video rendering

        Returns:
            The orientations of all particles - altered if the event has taken place, unaltered otherwise.
        """
        point_attraction_mask = np.full(totalNumberOfParticles, 0)
        affected = np.full(totalNumberOfParticles, False)
        point = np.array([0,0])
        lastUpdated = -1
        if self.checkTimestep(currentTimestep):
            if currentTimestep == self.startTimestep or currentTimestep == (self.startTimestep + self.duration):
                print(f"executing event at timestep {currentTimestep}")
            point_attraction_mask, affected, point = self.executeEvent(totalNumberOfParticles=totalNumberOfParticles, positions=positions, orientations=orientations, nsms=nsms, ks=ks, speeds=speeds, dt=dt, colourType=colourType)
            lastUpdated = currentTimestep

        return lastUpdated, point_attraction_mask, affected, point
    
    def executeEvent(self, totalNumberOfParticles, positions, orientations, nsms, ks, speeds, dt=None, colourType=None):
        """
        Executes the event,

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
            The affected individuals as a mask with a weight representing the alignment with the event position - negative values turning away, positive values turning towards
        """

        affected = se.selectAffected(eventSelectionType=self.eventSelectionType,
                                     totalNumberOfParticles=totalNumberOfParticles,
                                     positions=positions,
                                     originPoint=self.getOriginPoint(),
                                     domainSize=self.domainSize,
                                     radius=self.radius,
                                     numberOfAffected=self.numberOfAffected)

        point_attraction_mask = np.full(len(orientations), 0)

        match self.eventEffect:
            case EventEffect.ALIGN_TO_FIXED_ANGLE:
                scores = self.vector_direction_score(uv_vectors=orientations,
                                                     positions=positions,
                                                     origin=self.getOriginPoint())

                point_attraction_mask = scores
            case EventEffect.AWAY_FROM_ORIGIN:
                point_attraction_mask[affected] = -1
            case EventEffect.RANDOM:
                point_attraction_mask = (np.random.sample(len(orientations))-0.5) * 2


        return point_attraction_mask, affected, self.getOriginPoint() 

    def getOriginPoint(self):
        """
        Determines the point of origin of the event.

        Params:
            None

        Returns:
            The point of origin of the event in [X,Y]-coordinates.
        """
        return [0,0]

    def vector_direction_score(self, uv_vectors, positions, origin):
        """
        Determines the alignment to the origin point.

        Params:
            - uv_vectors (numpy array): the ideally targetted orientation vector for each agent
            - positions (numpy array): the position of each agent
            - origin (numpy array): the reference point

        Returns:
            A numpy array containing an alignment score in range [-1,1] with negative values indicating that
            the agent should move away from the origin point and a positive score indicating the opposite.
        """

        # Direction from vector position to the origin point
        d = origin - positions

        # Normalize vectors
        uv_norm = uv_vectors / np.linalg.norm(uv_vectors, axis=1, keepdims=True)
        d_norm = d / np.linalg.norm(d, axis=1, keepdims=True)

        # Dot product → [-1, 1]
        scores = np.sum(uv_norm * d_norm, axis=1)

        return scores
