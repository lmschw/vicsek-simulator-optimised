
import numpy as np

from enums.EnumColourType import ColourType

import services.ServicePreparation as ServicePreparation
import services.ServiceVicsekHelper as ServiceVicsekHelper

class BaseEvent:
    # TODO refactor to allow areas with a radius bigger than the radius of a particle, i.e. remove neighbourCells and determine all affected cells here
    # TODO make noisePercentage applicable to all events
    """
    Representation of an event occurring at a specified time and place within the domain and affecting 
    a specified percentage of particles. After creation, the check()-method takes care of everything.
    """
    def __init__(self, startTimestep, duration, domainSize, eventEffect, noisePercentage=None, blockValues=False, alterValues=False, 
                 switchSummary=None):
        """
        Creates an event that affects part of the swarm at a given timestep.

        Params:
            - startTimestep (int): the first timestep at which the stimulus is presented and affects the swarm
            - duration (int): the number of timesteps during which the stimulus is present and affects the swarm
            - domainSize (tuple of floats): the size of the domain
            - eventEffect (EnumEventEffect): how the orientations should be affected
            - noisePercentage (float, range: 0-100) [optional]: how much noise is added to the orientation determined by the event (only works for certain events)
            - blockValues (boolean) [optional]: whether the values (nsm, k, speed etc.) should be blocked after the update. By default False
            - alterValues (boolean) [optional]: whether the values (nsm, k, speed etc.) should be altered by the event. If False, only the orientations will be updated. By default False
            - switchSummary (SwitchSummary) [optional]: The switches that are available to the particles. Required to perform value alterations
            
        Returns:
            No return.
        """
        self.startTimestep = startTimestep
        self.duration = duration
        self.eventEffect = eventEffect
        self.domainSize = np.asarray(domainSize)
        self.noisePercentage = noisePercentage
        self.blockValues = blockValues
        self.alterValues = alterValues
        self.switchSummary = switchSummary
        if self.noisePercentage != None:
            self.noise = ServicePreparation.getNoiseAmplitudeValueForPercentage(self.noisePercentage)
        if self.alterValues == True and self.switchSummary == None:
            raise Exception("If the event is supposed to alter the values, a switchSummary needs to be supplied")
        

    def getShortPrintVersion(self):
        return f"t{self.startTimestep}d{self.duration}e{self.eventEffect.val}"

    def getParameterSummary(self):
        summary = {"startTimestep": self.startTimestep,
            "duration": self.duration,
            "eventEffect": self.eventEffect.val,
            "domainSize": self.domainSize.tolist(),
            "noisePercentage": self.noisePercentage,
            "blockValues": self.blockValues,
            "alterValues": self.alterValues
            }
        if self.switchSummary:
            summary["switchSummary"] = self.switchSummary.getParameterSummary()
        else:
            summary["switchSummary"] = None
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
        blocked = np.full(totalNumberOfParticles, False)
        colours = np.full(totalNumberOfParticles, 'k')
        if self.checkTimestep(currentTimestep):
            if currentTimestep == self.startTimestep or currentTimestep == (self.startTimestep + self.duration):
                print(f"executing event at timestep {currentTimestep}")
            alteredOrientations, alteredNsms, alteredKs, alteredSpeeds, blockedUpdate, colours = self.executeEvent(totalNumberOfParticles=totalNumberOfParticles, positions=positions, orientations=orientations, nsms=nsms, ks=ks, speeds=speeds, dt=dt, colourType=colourType)

            if isActivationTimeDelayRelevantForEvent:
                orientations = ServiceVicsekHelper.revertTimeDelayedChanges(currentTimestep, orientations, alteredOrientations, activationTimeDelays)
            else:
                orientations = alteredOrientations
                if self.blockValues:
                    blocked = blockedUpdate
                if self.alterValues:
                    nsms = alteredNsms
                    ks = alteredKs
                    speeds = alteredSpeeds
        return orientations, nsms, ks, speeds, blocked, colours

    def checkTimestep(self, currentTimestep):
        """
        Checks if the event should be triggered.

        Params:
            - currentTimestep (int): the timestep within the experiment run to see if the event should be triggered

        Returns:
            A boolean representing whether or not the event should be triggered.
        """
        return self.startTimestep <= currentTimestep and currentTimestep <= (self.startTimestep + self.duration)
    
    def applyNoiseDistribution(self, orientations):
        """
        Applies noise to the orientations.

        Params:
            - orientations (array of tuples (u,v)): the orientation of every particle at the current timestep

        Returns:
            An array of tuples (u,v) that represents the orientation of every particle at the current timestep after noise has been applied.
        """
        return orientations + np.random.normal(scale=self.noise, size=(len(orientations), len(self.domainSize)))
    
    def executeEvent(self, totalNumberOfParticles, positions, orientations, nsms, ks, speeds, dt, colourType=None):
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
        # base event does not do anything here
        return orientations, nsms, ks, speeds, np.full(totalNumberOfParticles, False), np.full(totalNumberOfParticles, 'k')
    
    def getColours(self, colourType, affected, totalNumberOfParticles):
        """
        Determines the colour of every particle for future video rendering.

        Params:
            - colourType (ColourType): if and how the particles should be encoded for colour for future video rendering.
            - affected (array of booleans): which particles are affected
            - totalNumberOfParticles (int): how many particles are in the domain

        Returns:
            Numpy array containing a string representation of the colour of every particle.
        """
        colours = np.full(totalNumberOfParticles, 'k')
        if colourType == ColourType.AFFECTED:
            colours[affected] = 'r'
        return colours