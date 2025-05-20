import random
import numpy as np

from model.VicsekIndividualsMultiSwitch import VicsekWithNeighbourSelection
from enums.EnumThresholdEvaluationMethod import ThresholdEvaluationMethod
import DefaultValues as dv

class VicsekWithNeighbourSelectionAndKStaggering(VicsekWithNeighbourSelection):

    def __init__(self, domainSize, radius, noise, numberOfParticles, kValues, neighbourSelectionMechanism, speed=dv.DEFAULT_SPEED, switchSummary=None, 
                 events=None, degreesOfVision=dv.DEFAULT_DEGREES_OF_VISION, activationTimeDelays=[], isActivationTimeDelayRelevantForEvents=False, 
                 colourType=None, thresholdEvaluationMethod=ThresholdEvaluationMethod.LOCAL_ORDER, updateIfNoNeighbours=True, returnHistories=True, 
                 logPath=None, logInterval=1, percentageFirstKValue=0.5, enforcePercentageSplit=True, initialChoiceOnly=True):
        super().__init__(domainSize, radius, noise, numberOfParticles, kValues[0], neighbourSelectionMechanism, speed, switchSummary, events, 
                         degreesOfVision, activationTimeDelays, isActivationTimeDelayRelevantForEvents, colourType, thresholdEvaluationMethod, 
                         updateIfNoNeighbours, returnHistories, logPath, logInterval)
        self.kValues = kValues
        self.percentageFirstKValue = percentageFirstKValue
        self.enforcePercentageSplit = enforcePercentageSplit
        self.initialChoiceOnly = initialChoiceOnly

        if switchSummary != None:
            print("WARNING: The k value distribution is only applied at initialisation. If a k switch is active, this will change the distribution accordingly.")


    def initialiseSwitchingValues(self):
        nsms, ks, speeds, activationTimeDelays = super().initialiseSwitchingValues()
        if self.enforcePercentageSplit:
            numAffected = int(self.percentageFirstKValue * self.numberOfParticles)
            affected = random.sample(range(0, self.numberOfParticles), numAffected)
            ks[affected] = self.kValues[1] # initialised with kValues[0]
        else:
            rands = np.random.random(self.numberOfParticles)
            k0s = np.full(self.numberOfParticles, self.kValues[0])
            k1s = np.full(self.numberOfParticles, self.kValues[1])
            ks = np.where(rands <= self.percentageFirstKValue/100, k0s, k1s)
        return nsms, ks, speeds, activationTimeDelays

    def prepareKs(self, ks):
        if self.initialChoiceOnly:
            ks = ks
        else:
            ks = np.random.choice(self.kValues, size=self.numberOfParticles, p=[self.percentageFirstKValue, 1-self.percentageFirstKValue])
        return ks
