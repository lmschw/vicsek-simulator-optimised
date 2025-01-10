import time
import numpy as np
import scipy.integrate as integrate

from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from enums.EnumSwitchType import SwitchType
from events.ExternalStimulusEvent import ExternalStimulusOrientationChangeEvent
from enums.EnumEventEffect import EventEffect
from enums.EnumDistributionType import DistributionType
from enums.EnumEventSelectionType import EventSelectionType

from model.SwitchInformation import SwitchInformation
from model.SwitchSummary import SwitchSummary
from model.VicsekIndividualsMultiSwitch import VicsekWithNeighbourSelection

import services.ServicePreparation as ServicePreparation
import services.ServiceGeneral as ServiceGeneral
import services.ServiceSavedModel as ServiceSavedModel
import services.ServiceMetric as ServiceMetric

radius = 20
density = 0.09
numIterationsPerRun = 10
startTimestepEvaluation = 2001

def getSwitchInformation(switchType, threshold, previousSteps):
    match switchType:
        case SwitchType.NEIGHBOUR_SELECTION_MECHANISM:
            return SwitchInformation(switchType=SwitchType.NEIGHBOUR_SELECTION_MECHANISM, 
                                        values=(NeighbourSelectionMechanism.FARTHEST, NeighbourSelectionMechanism.NEAREST),
                                        thresholds=[threshold],
                                        numberPreviousStepsForThreshold=previousSteps
                                        )
        case SwitchType.K:
            return SwitchInformation(switchType=SwitchType.K, 
                                    values=(5, 1),
                                    thresholds=[threshold],
                                    numberPreviousStepsForThreshold=previousSteps
                                    )
        
def getSwitchSummary(switchType, threshold, previousSteps):
    return SwitchSummary([getSwitchInformation(switchType, threshold, previousSteps)])

def getEvent(domainSize, radius, eventEffect):
    return ExternalStimulusOrientationChangeEvent(startTimestep=1000,
                                                duration=1000,  
                                                domainSize=domainSize, 
                                                eventEffect=eventEffect, 
                                                distributionType=DistributionType.LOCAL_SINGLE_SITE, 
                                                areas=[[domainSize[0]/2, domainSize[1]/2, radius]],
                                                angle=np.pi,
                                                radius=radius,
                                                eventSelectionType=EventSelectionType.RANDOM
                                                )

def getEvents(domainSize, radius, eventEffect):
    return [getEvent(domainSize, radius, eventEffect)]


def getFinalOrderForSwitching(switchType, eventEffect, nsm, k, threshold, previousSteps):
    domainSize = (50, 50)
    n = ServicePreparation.getNumberOfParticlesForConstantDensity(density, domainSize)
    noisePercentage = 1
    noise = ServicePreparation.getNoiseAmplitudeValueForPercentage(noisePercentage)
    speed = 1
    degreesOfVision = 2*np.pi
    tmax = 5000
    numberOfPreviousSteps = int(previousSteps)

    if eventEffect == EventEffect.ALIGN_TO_FIXED_ANGLE:
        startOrder = 0
        targetOrder = 1
        initialState = (None, None, None)
    else:
        startOrder = 1
        targetOrder = 0
        initialState = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(None, domainSize, n, angleX=0.5, angleY=0.5)

    events = getEvents(domainSize, radius, eventEffect)
    switchSummary = getSwitchSummary(switchType, threshold, numberOfPreviousSteps)

    results = {t: [] for t in range(tmax)}
    for i in range(numIterationsPerRun):
        simulator = VicsekWithNeighbourSelection(domainSize=domainSize,
                                                radius=radius,
                                                noise=noise,
                                                numberOfParticles=n,
                                                k=k,
                                                neighbourSelectionMechanism=nsm,
                                                speed=speed,
                                                switchSummary=switchSummary,
                                                degreesOfVision=degreesOfVision,
                                                events=events)
        simulationData, switchTypeValues = simulator.simulate(initialState=initialState, tmax=tmax)

        times, positions, orientations = simulationData

        [results[t].append(ServiceMetric.computeGlobalOrder(orientations[t])) for t in range(tmax)]
        # ServiceSavedModel.saveModel(simulationData=simulationData, path=f"test_{i+1}.json", 
        #                             modelParams=simulator.getParameterSummary())


    resultsArr = [np.average(results[t]) for t in range(tmax)]
    target = (startTimestepEvaluation) * [startOrder] + (tmax-startTimestepEvaluation) * [targetOrder]

    resultsIntegral = integrate.simpson(y=resultsArr[startTimestepEvaluation: tmax], x=range(startTimestepEvaluation, tmax))
    targetIntegral = integrate.simpson(y=target[startTimestepEvaluation: tmax], x=range(startTimestepEvaluation, tmax))

    if targetOrder == 1:
        return targetIntegral-resultsIntegral
    return resultsIntegral-targetIntegral
    
def getFinalOrderForPositionBasedNsmSwitchingForDistant(threshold, previousSteps):
    """
    ordered start. has to be k = 1 to allow a different behaviour when changing nsm
    """
    switchType = SwitchType.NEIGHBOUR_SELECTION_MECHANISM
    eventEffect = EventEffect.ALIGN_TO_FIXED_ANGLE
    nsm = NeighbourSelectionMechanism.NEAREST
    k = 1
    return getFinalOrderForSwitching(switchType=switchType, eventEffect=eventEffect, nsm=nsm, k=k, threshold=threshold, previousSteps=previousSteps)

def getFinalOrderForPositionBasedNsmSwitchingForPredator(threshold, previousSteps):
    """
    ordered start. has to be k = 1 to allow a different behaviour when changing nsm
    """
    switchType = SwitchType.NEIGHBOUR_SELECTION_MECHANISM
    eventEffect = EventEffect.AWAY_FROM_ORIGIN
    nsm = NeighbourSelectionMechanism.FARTHEST
    k = 1
    return getFinalOrderForSwitching(switchType=switchType, eventEffect=eventEffect, nsm=nsm, k=k, threshold=threshold, previousSteps=previousSteps)

def getFinalOrderForPositionBasedNsmSwitchingForRandom(threshold, previousSteps):
    """
    ordered start. has to be k = 1 to allow a different behaviour when changing nsm
    """
    switchType = SwitchType.NEIGHBOUR_SELECTION_MECHANISM
    eventEffect = EventEffect.RANDOM
    nsm = NeighbourSelectionMechanism.FARTHEST
    k = 1
    return getFinalOrderForSwitching(switchType=switchType, eventEffect=eventEffect, nsm=nsm, k=k, threshold=threshold, previousSteps=previousSteps)

def getFinalOrderForOrientationBasedNsmSwitchingForDistant(threshold, previousSteps):
    """
    ordered start. has to be k = 1 to allow a different behaviour when changing nsm
    """
    switchType = SwitchType.NEIGHBOUR_SELECTION_MECHANISM
    eventEffect = EventEffect.ALIGN_TO_FIXED_ANGLE
    nsm = NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE
    k = 1
    return getFinalOrderForSwitching(switchType=switchType, eventEffect=eventEffect, nsm=nsm, k=k, threshold=threshold, previousSteps=previousSteps)

def getFinalOrderForOrientationBasedNsmSwitchingForPredator(threshold, previousSteps):
    """
    ordered start. has to be k = 1 to allow a different behaviour when changing nsm
    """
    switchType = SwitchType.NEIGHBOUR_SELECTION_MECHANISM
    eventEffect = EventEffect.AWAY_FROM_ORIGIN
    nsm = NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE
    k = 1
    return getFinalOrderForSwitching(switchType=switchType, eventEffect=eventEffect, nsm=nsm, k=k, threshold=threshold, previousSteps=previousSteps)

def getFinalOrderForOrientationBasedNsmSwitchingForRandom(threshold, previousSteps):
    """
    ordered start. has to be k = 1 to allow a different behaviour when changing nsm
    """
    switchType = SwitchType.NEIGHBOUR_SELECTION_MECHANISM
    eventEffect = EventEffect.RANDOM
    nsm = NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE
    k = 1
    return getFinalOrderForSwitching(switchType=switchType, eventEffect=eventEffect, nsm=nsm, k=k, threshold=threshold, previousSteps=previousSteps)


def getFinalOrderForPositionBasedKSwitchingForDistant(threshold, previousSteps):
    """
    disordered start. has to be NEAREST to allow a different behaviour when changing k
    """
    switchType = SwitchType.K
    eventEffect = EventEffect.ALIGN_TO_FIXED_ANGLE
    nsm = NeighbourSelectionMechanism.NEAREST
    k = 1
    return getFinalOrderForSwitching(switchType=switchType, eventEffect=eventEffect, nsm=nsm, k=k, threshold=threshold, previousSteps=previousSteps)

def getFinalOrderForPositionBasedKSwitchingForPredator(threshold, previousSteps):
    """
    ordered start. has to be NEAREST to allow a different behaviour when changing k
    """
    switchType = SwitchType.K
    eventEffect = EventEffect.AWAY_FROM_ORIGIN
    nsm = NeighbourSelectionMechanism.NEAREST
    k = 5
    return getFinalOrderForSwitching(switchType=switchType, eventEffect=eventEffect, nsm=nsm, k=k, threshold=threshold, previousSteps=previousSteps)

def getFinalOrderForPositionBasedKSwitchingForRandom(threshold, previousSteps):
    """
    ordered start. has to be NEAREST to allow a different behaviour when changing k
    """
    switchType = SwitchType.K
    eventEffect = EventEffect.RANDOM
    nsm = NeighbourSelectionMechanism.NEAREST
    k = 5
    return getFinalOrderForSwitching(switchType=switchType, eventEffect=eventEffect, nsm=nsm, k=k, threshold=threshold, previousSteps=previousSteps)

def getFinalOrderForOrientationBasedKSwitchingForDistant(threshold, previousSteps):
    """
    disordered start. has to be NEAREST to allow a different behaviour when changing k
    """
    switchType = SwitchType.K
    eventEffect = EventEffect.ALIGN_TO_FIXED_ANGLE
    nsm = NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE
    k = 1
    return getFinalOrderForSwitching(switchType=switchType, eventEffect=eventEffect, nsm=nsm, k=k, threshold=threshold, previousSteps=previousSteps)

def getFinalOrderForOrientationBasedKSwitchingForPredator(threshold, previousSteps):
    """
    ordered start. has to be NEAREST to allow a different behaviour when changing k
    """
    switchType = SwitchType.K
    eventEffect = EventEffect.AWAY_FROM_ORIGIN
    nsm = NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE
    k = 5
    return getFinalOrderForSwitching(switchType=switchType, eventEffect=eventEffect, nsm=nsm, k=k, threshold=threshold, previousSteps=previousSteps)

def getFinalOrderForOrientationBasedKSwitchingForRandom(threshold, previousSteps):
    """
    ordered start. has to be NEAREST to allow a different behaviour when changing k
    """
    switchType = SwitchType.K
    eventEffect = EventEffect.RANDOM
    nsm = NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE
    k = 5
    return getFinalOrderForSwitching(switchType=switchType, eventEffect=eventEffect, nsm=nsm, k=k, threshold=threshold, previousSteps=previousSteps)

def getOverallFinalOrder(threshold, previousSteps):
    comboResults = []
    comboResults.append(getFinalOrderForPositionBasedKSwitchingForDistant(threshold, previousSteps))
    ServiceGeneral.logWithTime(f"{threshold}-{previousSteps} - pos-k-distant: {comboResults[-1]}")
    comboResults.append(getFinalOrderForPositionBasedKSwitchingForPredator(threshold, previousSteps))
    ServiceGeneral.logWithTime(f"{threshold}-{previousSteps} - pos-k-predator: {comboResults[-1]}")
    comboResults.append(getFinalOrderForPositionBasedKSwitchingForRandom(threshold, previousSteps))
    ServiceGeneral.logWithTime(f"{threshold}-{previousSteps} - pos-k-random: {comboResults[-1]}")

    comboResults.append(getFinalOrderForPositionBasedNsmSwitchingForDistant(threshold, previousSteps))
    ServiceGeneral.logWithTime(f"{threshold}-{previousSteps} - pos-nsm-distant: {comboResults[-1]}")
    comboResults.append(getFinalOrderForPositionBasedNsmSwitchingForPredator(threshold, previousSteps))
    ServiceGeneral.logWithTime(f"{threshold}-{previousSteps} - pos-nsm-predator: {comboResults[-1]}")
    comboResults.append(getFinalOrderForPositionBasedNsmSwitchingForRandom(threshold, previousSteps))
    ServiceGeneral.logWithTime(f"{threshold}-{previousSteps} - pos-nsm-random: {comboResults[-1]}")

    comboResults.append(getFinalOrderForOrientationBasedKSwitchingForDistant(threshold, previousSteps))
    ServiceGeneral.logWithTime(f"{threshold}-{previousSteps} - ori-k-distant: {comboResults[-1]}")
    comboResults.append(getFinalOrderForOrientationBasedKSwitchingForPredator(threshold, previousSteps))
    ServiceGeneral.logWithTime(f"{threshold}-{previousSteps} - ori-k-predator: {comboResults[-1]}")
    comboResults.append(getFinalOrderForOrientationBasedKSwitchingForRandom(threshold, previousSteps))
    ServiceGeneral.logWithTime(f"{threshold}-{previousSteps} - ori-k-random: {comboResults[-1]}")

    comboResults.append(getFinalOrderForOrientationBasedNsmSwitchingForDistant(threshold, previousSteps))
    ServiceGeneral.logWithTime(f"{threshold}-{previousSteps} - ori-nsm-distant: {comboResults[-1]}")
    comboResults.append(getFinalOrderForOrientationBasedNsmSwitchingForPredator(threshold, previousSteps))
    ServiceGeneral.logWithTime(f"{threshold}-{previousSteps} - ori-nsm-predator: {comboResults[-1]}")
    comboResults.append(getFinalOrderForOrientationBasedNsmSwitchingForRandom(threshold, previousSteps))
    ServiceGeneral.logWithTime(f"{threshold}-{previousSteps} - ori-nsm-random: {comboResults[-1]}")

    ServiceGeneral.logWithTime(f"total: {np.average(np.array(list(results.values())))}")
    ServiceGeneral.logWithTime(f"{max(results, key = results.get)}: {max(results.values())}")
    ServiceSavedModel.saveDict(f"grid_search_threshold_previousSteps_d={density}_r={radius}.json", dict=results)
    return np.average(np.array(comboResults))

ServiceGeneral.logWithTime("start")
results = {}
for threshold in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
    for previousSteps in [1, 10, 50, 100, 500, 1000, 2000, 3000, 4000, 5000]:
        comboResults = []
        comboResults.append(getFinalOrderForPositionBasedKSwitchingForDistant(threshold, previousSteps))
        results[f"{threshold}-{previousSteps}-pos-k-d"] = comboResults[-1]
        ServiceGeneral.logWithTime(f"{threshold}-{previousSteps} - pos-k-distant: {comboResults[-1]}")
        comboResults.append(getFinalOrderForPositionBasedKSwitchingForPredator(threshold, previousSteps))
        results[f"{threshold}-{previousSteps}-pos-k-p"] = comboResults[-1]
        ServiceGeneral.logWithTime(f"{threshold}-{previousSteps} - pos-k-predator: {comboResults[-1]}")
        comboResults.append(getFinalOrderForPositionBasedKSwitchingForRandom(threshold, previousSteps))
        results[f"{threshold}-{previousSteps}-pos-k-r"] = comboResults[-1]
        ServiceGeneral.logWithTime(f"{threshold}-{previousSteps} - pos-k-random: {comboResults[-1]}")

        comboResults.append(getFinalOrderForPositionBasedNsmSwitchingForDistant(threshold, previousSteps))
        results[f"{threshold}-{previousSteps}-pos-nsm-d"] = comboResults[-1]
        ServiceGeneral.logWithTime(f"{threshold}-{previousSteps} - pos-nsm-distant: {comboResults[-1]}")
        comboResults.append(getFinalOrderForPositionBasedNsmSwitchingForPredator(threshold, previousSteps))
        results[f"{threshold}-{previousSteps}-pos-nsm-p"] = comboResults[-1]
        ServiceGeneral.logWithTime(f"{threshold}-{previousSteps} - pos-nsm-predator: {comboResults[-1]}")
        comboResults.append(getFinalOrderForPositionBasedNsmSwitchingForRandom(threshold, previousSteps))
        results[f"{threshold}-{previousSteps}-pos-nsm-r"] = comboResults[-1]
        ServiceGeneral.logWithTime(f"{threshold}-{previousSteps} - pos-nsm-random: {comboResults[-1]}")

        comboResults.append(getFinalOrderForOrientationBasedKSwitchingForDistant(threshold, previousSteps))
        results[f"{threshold}-{previousSteps}-ori-k-d"] = comboResults[-1]
        ServiceGeneral.logWithTime(f"{threshold}-{previousSteps} - ori-k-distant: {comboResults[-1]}")
        comboResults.append(getFinalOrderForOrientationBasedKSwitchingForPredator(threshold, previousSteps))
        results[f"{threshold}-{previousSteps}-ori-k-p"] = comboResults[-1]
        ServiceGeneral.logWithTime(f"{threshold}-{previousSteps} - ori-k-predator: {comboResults[-1]}")
        comboResults.append(getFinalOrderForOrientationBasedKSwitchingForRandom(threshold, previousSteps))
        results[f"{threshold}-{previousSteps}-ori-k-r"] = comboResults[-1]
        ServiceGeneral.logWithTime(f"{threshold}-{previousSteps} - ori-k-random: {comboResults[-1]}")

        comboResults.append(getFinalOrderForOrientationBasedNsmSwitchingForDistant(threshold, previousSteps))
        results[f"{threshold}-{previousSteps}-ori-nsm-d"] = comboResults[-1]
        ServiceGeneral.logWithTime(f"{threshold}-{previousSteps} - ori-nsm-distant: {comboResults[-1]}")
        comboResults.append(getFinalOrderForOrientationBasedNsmSwitchingForPredator(threshold, previousSteps))
        results[f"{threshold}-{previousSteps}-ori-nsm-p"] = comboResults[-1]
        ServiceGeneral.logWithTime(f"{threshold}-{previousSteps} - ori-nsm-predator: {comboResults[-1]}")
        comboResults.append(getFinalOrderForOrientationBasedNsmSwitchingForRandom(threshold, previousSteps))
        results[f"{threshold}-{previousSteps}-ori-nsm-r"] = comboResults[-1]
        ServiceGeneral.logWithTime(f"{threshold}-{previousSteps} - ori-nsm-random: {comboResults[-1]}")

        results[f"{threshold}-{previousSteps}-overall"] = np.average(np.array(comboResults))
        ServiceGeneral.logWithTime(f"{threshold}-{previousSteps}: {results[f"{threshold}-{previousSteps}-overall"]}")

ServiceGeneral.logWithTime(f"total: {np.average(np.array(list(results.values())))}")
ServiceGeneral.logWithTime(f"{max(results, key = results.get)}: {max(results.values())}")
ServiceSavedModel.saveDict(f"grid_search_threshold_previousSteps_integral_d={density}_r={radius}.json", dict=results)
