import numpy as np

import services.ServiceOrientations as ServiceOrientations
from events.BaseEvent import BaseEvent


class ExternalEventStimulusWallEvent(BaseEvent):
    def __init__(self, startTimestep, duration, wallTypeBehaviour, domainSize, noisePercentage=None, turnBy=0.314):
        """
        Creates an external stimulus event that affects part of the swarm at a given timestep.

        Params:
            - startTimestep (int): the first timestep at which the stimulus is presented and affects the swarm
            - endTimestep (int): the last timestep at which the stimulus is presented and affects the swarm

        Returns:
            No return.
        """
        super().__init__(startTimestep=startTimestep, duration=duration, domainSize=domainSize, eventEffect=None, noisePercentage=noisePercentage)
        self.wallTypeBehaviour = wallTypeBehaviour
        self.turnBy = turnBy

    def getParameterSummary(self):
        summary = super().getParameterSummary()
        summary["wallTypeBehaviour"] = self.wallTypeBehaviour.getParameterSummary()
        summary["turnBy"] = self.turnBy
        return summary

    def executeEvent(self, totalNumberOfParticles, positions, orientations, nsms, ks, speeds, dt, colourType=None):
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
        # TODO refactor related cose
        
        affected = self.wallTypeBehaviour.checkClosenessToBorder(positions)
        newOrientations = self.wallTypeBehaviour.getAvoidanceOrientations(positions, orientations, speeds, dt, self.turnBy)
        orientations = self.__applyNoiseDistribution(newOrientations)
        orientations = ServiceOrientations.normalizeOrientations(orientations)
        colours = self.getColours(colourType=colourType, affected=affected, totalNumberOfParticles=totalNumberOfParticles)
        return orientations, nsms, ks, speeds, affected, colours
    
    def __applyNoiseDistribution(self, orientations):
        if self.noise == None:
            return orientations
        return orientations + np.random.normal(scale=self.noise, size=(len(orientations), len(self.domainSize)))
