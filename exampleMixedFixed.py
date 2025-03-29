import time
import numpy as np

from model.VicsekIndividualsMultiSwitchKStaggering import VicsekWithNeighbourSelectionAndKStaggering
from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from enums.EnumSwitchType import SwitchType
from events.ExternalStimulusEvent import ExternalStimulusOrientationChangeEvent
from enums.EnumEventEffect import EventEffect
from enums.EnumDistributionType import DistributionType
from enums.EnumEventSelectionType import EventSelectionType
from enums.EnumColourType import ColourType
from enums.EnumThresholdEvaluationMethod import ThresholdEvaluationMethod

from model.SwitchInformation import SwitchInformation
from model.SwitchSummary import SwitchSummary

import services.ServicePreparation as ServicePreparation
import services.ServiceGeneral as ServiceGeneral
import services.ServiceSavedModel as ServiceSavedModel

# ------------------------ SETUP --------------------------------------

# GENERAL
domainSize = (50, 50)
speed = 1
initialAngleX = 0.5
initialAngleY = 0.5
colourType = None
degreesOfVision = 2 * np.pi

tmax = 15000
iStart = 1
iStop = 11
saveLocation = "results_noise_stagger"

# TEST VALS
nsms = [NeighbourSelectionMechanism.ALL,
        NeighbourSelectionMechanism.RANDOM,
        NeighbourSelectionMechanism.NEAREST,
        NeighbourSelectionMechanism.FARTHEST,
        NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE,
        NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE]



ks = [1,2]

eventEffects = [EventEffect.ALIGN_TO_FIXED_ANGLE,
                EventEffect.AWAY_FROM_ORIGIN,
                EventEffect.RANDOM]

nsmCombos = [[NeighbourSelectionMechanism.FARTHEST, NeighbourSelectionMechanism.NEAREST],
             [NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE, NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE]]

kCombos = [[1,5]]

densities = [0.06]
radii = [10]
initialConditions = ["ordered", "random"]

percentageFirstValue = 0.5

noisePercentages = [1, 2, 3, 4, 5]

startOverall = time.time()

for noisePercentage in noisePercentages:
    noise = ServicePreparation.getNoiseAmplitudeValueForPercentage(noisePercentage)
    # ------------------------ FIXED STRATEGIES ---------------------------------
    enforceSplit = True
    for density in densities:
        n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
        for radius in radii:
            for nsm in nsms:
                for initialCondition in initialConditions:
                    for i in range(iStart, iStop):
                        
                        events = []
                        if initialCondition == "ordered":
                            initialState = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(None, domainSize, n, angleX=initialAngleX, angleY=initialAngleY)
                        else:
                            initialState = (None, None, None)
                        switchSummary = None
                        
                        simulator = VicsekWithNeighbourSelectionAndKStaggering(domainSize=domainSize,
                                                                radius=radius,
                                                                noise=noise,
                                                                numberOfParticles=n,
                                                                kValues=ks,
                                                                neighbourSelectionMechanism=nsm,
                                                                speed=speed,
                                                                switchSummary=switchSummary,
                                                                degreesOfVision=degreesOfVision,
                                                                events=events,
                                                                colourType=colourType,
                                                                thresholdEvaluationMethod=None,
                                                                updateIfNoNeighbours=False, 
                                                                percentageFirstKValue=percentageFirstValue,
                                                                enforcePercentageSplit=enforceSplit)
                        simulationData, switchTypeValues = simulator.simulate(initialState=initialState, tmax=tmax)

                        savePath = f"{saveLocation}local_nosw_noev_stagger=({percentageFirstValue},{ks},{enforceSplit})_d={density}_r={radius}_{initialCondition}_nsm={nsm.value}_ks={ks[0]}-{ks[1]}_noise={noisePercentage}_{i}.json"
                        ServiceSavedModel.saveModel(simulationData=simulationData, path=savePath, 
                                                    modelParams=simulator.getParameterSummary(), switchValues=switchTypeValues)

    # ------------------------ NOSW, NOEV ---------------------------------
    enforceSplit = False
    for density in densities:
        n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
        for radius in radii:
            for nsm in nsms:
                for initialCondition in initialConditions:
                    for i in range(iStart, iStop):
                        
                        events = []
                        if initialCondition == "ordered":
                            initialState = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(None, domainSize, n, angleX=initialAngleX, angleY=initialAngleY)
                        else:
                            initialState = (None, None, None)
                        switchSummary = None
                        
                        simulator = VicsekWithNeighbourSelectionAndKStaggering(domainSize=domainSize,
                                                                radius=radius,
                                                                noise=noise,
                                                                numberOfParticles=n,
                                                                kValues=ks,
                                                                neighbourSelectionMechanism=nsm,
                                                                speed=speed,
                                                                switchSummary=switchSummary,
                                                                degreesOfVision=degreesOfVision,
                                                                events=events,
                                                                colourType=colourType,
                                                                thresholdEvaluationMethod=None,
                                                                updateIfNoNeighbours=False, 
                                                                percentageFirstKValue=percentageFirstValue,
                                                                enforcePercentageSplit=enforceSplit)
                        simulationData, switchTypeValues = simulator.simulate(initialState=initialState, tmax=tmax)

                        savePath = f"{saveLocation}local_nosw_noev_stagger=({percentageFirstValue},{ks},{enforceSplit})_d={density}_r={radius}_{initialCondition}_nsm={nsm.value}_ks={ks[0]}-{ks[1]}_noise={noisePercentage}_{i}.json"
                        ServiceSavedModel.saveModel(simulationData=simulationData, path=savePath, 
                                                    modelParams=simulator.getParameterSummary(), switchValues=switchTypeValues)
