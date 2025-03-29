import time
import numpy as np

from model.VicsekIndividualsMultiSwitch import VicsekWithNeighbourSelection
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
saveLocation = ""

# SWITCHING
thresholdEvaluationMethod = ThresholdEvaluationMethod.LOCAL_ORDER
threshold = [0.1]
numberOfPreviousSteps = 100
updateIfNoNeighbours = False

# EVENT
eventStart = 5000
eventDuration = 1000
eventDistributionType = DistributionType.LOCAL_SINGLE_SITE
eventAngle = np.pi
eventNumberAffected = None
eventSelectionType = EventSelectionType.RANDOM

# TEST VALS
nsms = [NeighbourSelectionMechanism.ALL,
        NeighbourSelectionMechanism.RANDOM,
        NeighbourSelectionMechanism.NEAREST,
        NeighbourSelectionMechanism.FARTHEST,
        NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE,
        NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE]

nsmsReduced = [NeighbourSelectionMechanism.NEAREST,
               NeighbourSelectionMechanism.FARTHEST,
               NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE,
               NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE]

ks = [1,5]

eventEffects = [EventEffect.ALIGN_TO_FIXED_ANGLE,
                EventEffect.AWAY_FROM_ORIGIN,
                EventEffect.RANDOM]

nsmCombos = [[NeighbourSelectionMechanism.FARTHEST, NeighbourSelectionMechanism.NEAREST],
             [NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE, NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE]]

kCombos = [[1,5]]

densities = [0.06, 0.05, 0.07]
radii = [10, 5]
initialConditions = ["ordered", "random"]

noisePercentages = [1, 2, 3, 4, 5]

startOverall = time.time()

# ------------------------ NOSW, NOEV ---------------------------------
for density in densities:
    ServiceGeneral.logWithTime(f"Starting NOSW, NOEV d={density}")
    startDensity = time.time()
    n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
    for radius in radii:
        ServiceGeneral.logWithTime(f"Starting NOSW, NOEV d={density}, r={radius}")
        startRadius = time.time()
        for noisePercentage in noisePercentages:
            startNoise = time.time()
            noise = ServicePreparation.getNoiseAmplitudeValueForPercentage(noisePercentage)
            for nsm in nsms:
                ServiceGeneral.logWithTime(f"Starting NOSW, NOEV d={density}, r={radius}, noise={noisePercentage}, nsm={nsm.value}")
                startNsm = time.time()
                for k in ks:
                    ServiceGeneral.logWithTime(f"Starting NOSW, NOEV d={density}, r={radius}, noise={noisePercentage}, nsm={nsm.value}, k={k}")
                    startK = time.time()
                    for initialCondition in initialConditions:
                        ServiceGeneral.logWithTime(f"Starting NOSW, NOEV d={density}, r={radius}, noise={noisePercentage}, nsm={nsm.value}, k={k}, init={initialCondition}")
                        startIC = time.time()
                        for i in range(iStart, iStop):
                            startI = time.time()
                            ServiceGeneral.logWithTime(f"Starting NOSW, NOEV d={density}, r={radius}, noise={noisePercentage}, nsm={nsm.value}, k={k}, init={initialCondition}, i={i}")
                            
                            events = []
                            if initialCondition == "ordered":
                                initialState = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(None, domainSize, n, angleX=initialAngleX, angleY=initialAngleY)
                            else:
                                initialState = (None, None, None)
                            switchSummary = None
                            
                            simulator = VicsekWithNeighbourSelection(domainSize=domainSize,
                                                                    radius=radius,
                                                                    noise=noise,
                                                                    numberOfParticles=n,
                                                                    k=k,
                                                                    neighbourSelectionMechanism=nsm,
                                                                    speed=speed,
                                                                    switchSummary=switchSummary,
                                                                    degreesOfVision=degreesOfVision,
                                                                    events=events,
                                                                    colourType=colourType,
                                                                    thresholdEvaluationMethod=thresholdEvaluationMethod,
                                                                    updateIfNoNeighbours=updateIfNoNeighbours)
                            simulationData, switchTypeValues = simulator.simulate(initialState=initialState, tmax=tmax)

                            savePath = f"{saveLocation}local_nosw_noev_d={density}_r={radius}_{initialCondition}_nsm={nsm.value}_k={k}_noise={noise}_{i}.json"
                            ServiceSavedModel.saveModel(simulationData=simulationData, path=savePath, 
                                                        modelParams=simulator.getParameterSummary(), switchValues=switchTypeValues)
                            endI = time.time()
                            ServiceGeneral.logWithTime(f"Completed NOSW, NOEV d={density}, r={radius}, noise={noisePercentage}, nsm={nsm.value}, k={k}, init={initialCondition}, i={i} in {ServiceGeneral.formatTime(endI-startI)}")
                        endIC = time.time()
                        ServiceGeneral.logWithTime(f"Completed NOSW, NOEV d={density}, r={radius}, noise={noisePercentage}, nsm={nsm.value}, k={k}, init={initialCondition} in {ServiceGeneral.formatTime(endIC-startIC)}")
                    endK = time.time()
                    ServiceGeneral.logWithTime(f"Completed NOSW, NOEV d={density}, r={radius}, noise={noisePercentage}, nsm={nsm.value}, k={k} in {ServiceGeneral.formatTime(endK-startK)}")
                endNsm = time.time()
                ServiceGeneral.logWithTime(f"Completed NOSW, NOEV d={density}, r={radius}, noise={noisePercentage}, nsm={nsm.value} in {ServiceGeneral.formatTime(endNsm-startNsm)}")
            endNoise = time.time()
            ServiceGeneral.logWithTime(f"Completed NOSW, NOEV d={density}, r={radius}, noise={noisePercentage} in {ServiceGeneral.formatTime(endNoise-startNoise)}")
        endRadius = time.time()
        ServiceGeneral.logWithTime(f"Completed NOSW, NOEV d={density}, r={radius} in {ServiceGeneral.formatTime(endRadius-startRadius)}")
    endDensity = time.time()
    ServiceGeneral.logWithTime(f"Completed NOSW, NOEV d={density} in {ServiceGeneral.formatTime(endDensity-startDensity)}")

endOverall = time.time()
ServiceGeneral.logWithTime(f"Completed run in {ServiceGeneral.formatTime(endOverall-startOverall)}")
