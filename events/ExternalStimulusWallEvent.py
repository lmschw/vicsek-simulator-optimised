import numpy as np

import services.ServiceOrientations as ServiceOrientations
from events.BaseEvent import BaseEvent


class ExternalEventStimulusWallEvent(BaseEvent):
    def __init__(self, startTimestep, duration, wallTypeBehaviour, domainSize, noisePercentage=None, turnBy=0.314):
        """
        Creates an external stimulus event that affects part of the swarm at a given timestep with regards to a wall.

        Params:
            - startTimestep (int): the first timestep at which the stimulus is presented and affects the swarm
            - duration (int): the number of timesteps during which the stimulus is present and affects the swarm
            - wallTypeBehaviour (WallType): the type, form and behaviour of the wall
            - domainSize (tuple of floats): the size of the domain
            - noisePercentage (float, range: 0-100) [optional]: how much noise is added to the orientation determined by the event (only works for certain events)
            - turnBy (float) [optional]: the angle that is added or subtracted from the current angle at every iteration to find an escape angle in case of potential collisions

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
            - nsms (array of NeighbourSelectionMechanisms): the neighbour selection mechanism currently selected by each individual at the current timestep
            - ks (array of int): the number of neighbours currently selected by each individual at the current timestep
            - speeds (array of float): the speed of every particle at the current timestep
            - dt (float) [optional]: the difference between the timesteps
            - colourType (ColourType) [optional]: if and how particles should be encoded for colour for future video rendering

        Returns:
            The orientations, neighbour selection mechanisms, ks, speeds, blockedness and colour of all particles after the event has been executed.
        """        
        affected = self.wallTypeBehaviour.checkClosenessToBorder(positions)
        newOrientations = self.wallTypeBehaviour.getAvoidanceOrientations(positions, orientations, speeds, dt, self.turnBy)
        orientations = self.applyNoiseDistribution(newOrientations)
        orientations = ServiceOrientations.normalizeOrientations(orientations)
        colours = self.getColours(colourType=colourType, affected=affected, totalNumberOfParticles=totalNumberOfParticles)
        return orientations, nsms, ks, speeds, affected, colours