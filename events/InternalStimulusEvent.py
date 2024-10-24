import random
import math
import numpy as np
from scipy.spatial.transform import Rotation as R


from enums.EnumDistributionType import DistributionType
from enums.EnumEventEffect import InternalEventEffect
from enums.EnumSwitchType import SwitchType
from enums.EnumColourType import ColourType

import DefaultValues as dv
import services.ServiceOrientations as ServiceOrientations
import services.ServicePreparation as ServicePreparation
import services.ServiceVicsekHelper as ServiceVicsekHelper

from events.BaseEvent import BaseEvent

class InternalStimulusOrientationChangeEvent(BaseEvent):
    # TODO refactor to allow areas with a radius bigger than the radius of a particle, i.e. remove neighbourCells and determine all affected cells here
    """
    Representation of an event occurring at a specified time and place within the domain and affecting 
    a specified percentage of particles. After creation, the check()-method takes care of everything.
    """
    def __init__(self, startTimestep, duration, domainSize, eventEffect, numberOfAffectedParticles=None, percentage=None, angle=None, 
                 noisePercentage=None, blockValues=False, alterValues=False, switchSummary=None):
        """
        Creates an event that affects part of the swarm at a given timestep.

        Params:
            - startTimestep (int): the first timestep at which the stimulus is presented and affects the swarm
            - duration (int): the number of timesteps during which the stimulus is present and affects the swarm
            - domainSize (tuple of floats): the size of the domain
            - eventEffect (EnumEventEffect): how the orientations should be affected
            - numberOfAffectedParticles (int) [optional]: how many particles should be affected
            - percentage (float) [optional]: how many particles should be affected in terms of percentage
            - angle (float) [optional]: the angle that the affected particles should be set to (only applicable to some EventEffects)
            - noisePercentage (float, range: 0-100) [optional]: how much noise is added to the orientation determined by the event (only works for certain events)
            - blockValues (boolean) [optional]: whether the values (nsm, k, speed etc.) should be blocked after the update. By default False
            - alterValues (boolean) [optional]: whether the values (nsm, k, speed etc.) should be altered by the event. If False, only the orientations will be updated. By default False
            - switchSummary (SwitchSummary) [optional]: The switches that are available to the particles. Required to perform value alterations
            
        Returns:
            No return.
        """
        super().__init__(startTimestep=startTimestep, duration=duration, domainSize=domainSize, eventEffect=eventEffect, 
                         noisePercentage=noisePercentage, blockValues=blockValues, alterValues=alterValues,
                         switchSummary=switchSummary)
        self.angle = angle
        self.percentage = percentage
        self.numberOfAffectedParticles = numberOfAffectedParticles
        self.affectedParticles = []
        self.angles = []
        
    def getShortPrintVersion(self):
        return f"t{self.startTimestep}d{self.duration}e{self.eventEffect.val}a{self.angle}dt{self.distributionType.value}a{self.areas}"

    def getParameterSummary(self):
        summary = super().getParameterSummary()
        summary["angle"] = self.angle
        summary["percentage"] = self.percentage
        summary["numberOfAffectedParticles"] = self.numberOfAffectedParticles
        return summary
    
    def executeEvent(self, totalNumberOfParticles, positions, orientations,nsms, ks, speeds, dt=None, colourType=None):
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

        if len(self.affectedParticles) == 0:
            self.affectedParticles = self.selectParticles(totalNumberOfParticles)
            print(self.affectedParticles)

        match self.eventEffect:
            case InternalEventEffect.ALIGN_TO_FIXED_ANGLE:
                orientations[self.affectedParticles] = ServiceOrientations.computeUvCoordinates(self.angle)
                if self.switchSummary:
                    if self.switchSummary.isActive(SwitchType.NEIGHBOUR_SELECTION_MECHANISM):
                        nsms[self.affectedParticles] = self.switchSummary.getBySwitchType(SwitchType.NEIGHBOUR_SELECTION_MECHANISM).orderSwitchValue
                    if self.switchSummary.isActive(SwitchType.K):
                        ks[self.affectedParticles] = self.switchSummary.getBySwitchType(SwitchType.K).orderSwitchValue      
                    if self.switchSummary.isActive(SwitchType.SPEED):
                        speeds[self.affectedParticles] = self.switchSummary.getBySwitchType(SwitchType.SPEED).orderSwitchValue    
            case InternalEventEffect.REINFORCE_RANDOM_ANGLE:
                if len(self.angles) == 0:
                    self.angles = self.getRandomOrientations(len(self.affectedParticles))
                orientations[self.affectedParticles] = self.angles
                if self.switchSummary:
                    if self.switchSummary.isActive(SwitchType.NEIGHBOUR_SELECTION_MECHANISM):
                        nsms[self.affectedParticles] = self.switchSummary.getBySwitchType(SwitchType.NEIGHBOUR_SELECTION_MECHANISM).disorderSwitchValue
                    if self.switchSummary.isActive(SwitchType.K):
                        ks[self.affectedParticles] = self.switchSummary.getBySwitchType(SwitchType.K).disorderSwitchValue      
                    if self.switchSummary.isActive(SwitchType.SPEED):
                        speeds[self.affectedParticles] = self.switchSummary.getBySwitchType(SwitchType.SPEED).disorderSwitchValue  
        orientations = ServiceOrientations.normalizeOrientations(orientations)
        blocked = np.full(totalNumberOfParticles, False)
        blocked[self.affectedParticles] = True
        colours = self.getColours(colourType=colourType, affected=self.affectedParticles, totalNumberOfParticles=totalNumberOfParticles)
        
        return orientations, nsms, ks, speeds, blocked, colours
    
    def selectParticles(self, totalNumberOfParticles):
        """
        Determines which particles are affected by the event.

        Params:
            - totalNumberOfParticles (int): how many particles are in the domain

        Returns:
            A list of indices of the affected particles
        """
        if self.numberOfAffectedParticles == None:
            self.numberOfAffectedParticles = int((self.percentage / 100) * totalNumberOfParticles)
        if self.numberOfAffectedParticles > totalNumberOfParticles:
            self.numberOfAffectedParticles = totalNumberOfParticles
        return np.random.choice(totalNumberOfParticles, self.numberOfAffectedParticles, replace=False)
    
    def getRandomOrientations(self, numAffectedParticles):
        """
        Selects a random orientation.

        Returns:
            A random orientation in [U,V]-coordinates.
        """
        return ServiceOrientations.normalizeOrientations(np.random.rand(numAffectedParticles, len(self.domainSize))-0.5)