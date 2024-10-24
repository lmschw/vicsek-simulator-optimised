from model.VicsekIndividualsMultiSwitch import VicsekWithNeighbourSelection
import DefaultValues as dv
import numpy as np

# TODO add colours and activationTimeDelay (incl. boolean)
class VicsekIndividualsWallEvents(VicsekWithNeighbourSelection):
    def __init__(self, domainSize, radius, noise, numberOfParticles, k, neighbourSelectionMechanism,
                 speed=dv.DEFAULT_SPEED, switchSummary=None, events=None, degreesOfVision=dv.DEFAULT_DEGREES_OF_VISION,
                 wallEvents=None):
        """
        Params:
            - domainSize (tuple of floats): the size of the domain
            - radius (float): the perception radius of the individuals
            - noise (float): the noise in the environment that is applied to the orientation of each particle at every timestep
            - numberOfParticles (int): how many particles are in the domain
            - k (int): how many neighbours should each individual consider (start value if k-switching is active)
            - neighbourSelectonMechansim (NeighbourSelectionMechanism): which neighbours each individual should consider (start value if nsm-switching is active)
            - speed (float) [optional]: the speed at which the particles move
            - switchSummary (SwitchSummary) [optional]: The switches that are available to the particles
            - events (list of BaseEvents or child classes) [optional]: the events that occur within the domain during the simulation
            - degreesOfVision (float, range(0, 2pi)) [optional]: how much of their surroundings each individual is able to see. By default 2pi
            - wallEvents (list of ExternalStimulusWallEvents) [optional]: the events that occur within the domain during the simulation and relate to walls

        Returns:
            No return.
        """
        super().__init__(domainSize=domainSize,
                         speed=speed,
                         radius=radius,
                         noise=noise,
                         numberOfParticles=numberOfParticles,
                         neighbourSelectionMechanism=neighbourSelectionMechanism,
                         k=k,
                         switchSummary=switchSummary,
                         degreesOfVision=degreesOfVision,
                         events=events)
        self.wallEvents = wallEvents

    def handleEvents(self, t, positions, orientations, nsms, ks, speeds):
        """
        Handles all types of events.

        Params:
            - t (int): the current timestep
            - positions (array of (x,y)-coordinates): the position of every particle at the current timestep
            - orientations (array of (u,v)-coordinates): the orientation of every particle at the current timestep
            - nsms (array of NeighbourSelectionMechanism): how every particle selects its neighbours at the current timestep
            - ks (array of ints): how many neighbours each particle considers at the current timestep
            - speeds (array of floats): how fast each particle moves at the current timestep
            - activationTimeDelays (array of ints): how often a particle is ready to update its orientation

        Returns:
            Arrays containing the updates orientations, neighbour selecton mechanisms, ks, speeds, which particles are blocked from updating and the colours assigned to each particle.
        """
        blocked = np.full(self.numberOfParticles, False)
        if self.events != None:
                for event in self.events:
                    orientations, nsms, ks, speeds, blocked = event.check(self.numberOfParticles, t, positions, orientations, nsms, ks, speeds, self.dt)
        if self.wallEvents != None:
            for event in self.wallEvents:
                orientations, nsms, ks, speeds, blocked = event.check(self.numberOfParticles, t, positions, orientations, nsms, ks, speeds, self.dt)
        return orientations, nsms, ks, speeds, blocked
