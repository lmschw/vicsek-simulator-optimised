
import numpy as np

import services.ServicePreparation as ServicePreparation

class BaseEvent:
    # TODO refactor to allow areas with a radius bigger than the radius of a particle, i.e. remove neighbourCells and determine all affected cells here
    """
    Representation of an event occurring at a specified time and place within the domain and affecting 
    a specified percentage of particles. After creation, the check()-method takes care of everything.
    """
    def __init__(self, startTimestep, duration, domainSize, eventEffect, noisePercentage=None):
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
        self.startTimestep = startTimestep
        self.duration = duration
        self.eventEffect = eventEffect
        self.domainSize = np.asarray(domainSize)
        self.noisePercentage = noisePercentage
        if self.noisePercentage != None:
            self.noise = ServicePreparation.getNoiseAmplitudeValueForPercentage(self.noisePercentage)

    def getShortPrintVersion(self):
        return f"t{self.startTimestep}d{self.duration}e{self.eventEffect.val}"

    def getParameterSummary(self):
        summary = {"startTimestep": self.startTimestep,
            "duration": self.duration,
            "eventEffect": self.eventEffect.val,
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
            if currentTimestep == self.startTimestep or currentTimestep == (self.startTimestep + self.duration):
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
        return self.startTimestep <= currentTimestep and currentTimestep <= (self.startTimestep + self.duration)
    
    def applyNoiseDistribution(self, orientations):
        return orientations + np.random.normal(scale=self.noise, size=(len(orientations), len(self.domainSize)))
    
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
        # base event does not do anything here
        return orientations