from model.VicsekIndividualsMultiSwitch import VicsekWithNeighbourSelection
import DefaultValues as dv
import numpy as np

class VicsekIndividualsWallEvents(VicsekWithNeighbourSelection):
    def __init__(self, domainSize, radius, noise, numberOfParticles, k, neighbourSelectionMechanism,
                 speed=dv.DEFAULT_SPEED, switchSummary=None, events=None, degreesOfVision=dv.DEFAULT_DEGREES_OF_VISION,
                 wallEvents=None):
        """
        Initialize the model with all its parameters

        Params:
            - neighbourSelectionMechanism (EnumNeighbourSelectionMechanism.NeighbourSelectionMechanism): how the particles select which of the other particles within their perception radius influence their orientation at any given time step
            - domainSize (tuple x,y) [optional]: the size of the domain for the particle movement
            - speed (int) [optional]: how fast the particles move
            - radius (int) [optional]: defines the perception field of the individual particles, i.e. the area in which it can perceive other particles
            - noise (float) [optional]: noise amplitude. adds noise to the orientation adaptation
            - numberOfParticles (int) [optional]: the number of particles within the domain, n
            - k (int) [optional]: the number of neighbours a particle considers when updating its orientation at every time step
            - switchValues (tuple (orderValue, disorderValue)) [optional]: the value that is supposed to create order and the value that is supposed to create disorder.
                    Must be the same type as the switchType
            - orderThresholds (array) [optional]: the difference in local order compared to the previous timesteps that will cause a switch.
                    If only one number is supplied (as an array with one element), will be used to check if the difference between the previous and the current local order is greater than the threshold or as the lower threshold with the upper threshold being (1-orderThreshold)
                    If two numbers are supplied, will be used as a lower and an upper threshold that triggers a switch: [lowerThreshold, upperThreshold]
            - numberPreviousStepsForThreshold (int) [optional]: the number of previous timesteps that are considered for the average to be compared to the threshold value(s)
            - switchingActive (boolean) [optional]: if False, the particles cannot update their values
   
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
        blocked = np.full(self.numberOfParticles, False)
        if self.events != None:
                for event in self.events:
                    orientations, nsms, ks, speeds, blocked = event.check(self.numberOfParticles, t, positions, orientations, nsms, ks, speeds, self.dt)
        if self.wallEvents != None:
            for event in self.wallEvents:
                orientations, nsms, ks, speeds, blocked = event.check(self.numberOfParticles, t, positions, orientations, nsms, ks, speeds, self.dt)
        return orientations, nsms, ks, speeds, blocked
