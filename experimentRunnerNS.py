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
noisePercentage = 1
noise = ServicePreparation.getNoiseAmplitudeValueForPercentage(noisePercentage)
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
eventDuration = 100
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

densities = [0.06]
radii = [10]
initialConditions = ["ordered", "random"]


startOverall = time.time()

# ------------------------ NOSW, NOEV ---------------------------------
for density in densities:
    ServiceGeneral.logWithTime(f"Starting NOSW, NOEV d={density}")
    startDensity = time.time()
    n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
    for radius in radii:
        ServiceGeneral.logWithTime(f"Starting NOSW, NOEV d={density}, r={radius}")
        startRadius = time.time()
        for nsm in nsms:
            ServiceGeneral.logWithTime(f"Starting NOSW, NOEV d={density}, r={radius}, nsm={nsm.value}")
            startNsm = time.time()
            for k in ks:
                ServiceGeneral.logWithTime(f"Starting NOSW, NOEV d={density}, r={radius}, nsm={nsm.value}, k={k}")
                startK = time.time()
                for initialCondition in initialConditions:
                    ServiceGeneral.logWithTime(f"Starting NOSW, NOEV d={density}, r={radius}, nsm={nsm.value}, k={k}, init={initialCondition}")
                    startIC = time.time()
                    for i in range(iStart, iStop):
                        startI = time.time()
                        ServiceGeneral.logWithTime(f"Starting NOSW, NOEV d={density}, r={radius}, nsm={nsm.value}, k={k}, init={initialCondition}, i={i}")
                        
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

                        savePath = f"{saveLocation}local_nosw_noev_d={density}_r={radius}_{initialCondition}_nsm={nsm.value}_k={k}_{i}.json"
                        ServiceSavedModel.saveModel(simulationData=simulationData, path=savePath, 
                                                    modelParams=simulator.getParameterSummary(), switchValues=switchTypeValues)
                        endI = time.time()
                        ServiceGeneral.logWithTime(f"Completed NOSW, NOEV d={density}, r={radius}, nsm={nsm.value}, k={k}, init={initialCondition}, i={i} in {ServiceGeneral.formatTime(endI-startI)}")
                    endIC = time.time()
                    ServiceGeneral.logWithTime(f"Completed NOSW, NOEV d={density}, r={radius}, nsm={nsm.value}, k={k}, init={initialCondition} in {ServiceGeneral.formatTime(endIC-startIC)}")
                endK = time.time()
                ServiceGeneral.logWithTime(f"Completed NOSW, NOEV d={density}, r={radius}, nsm={nsm.value}, k={k} in {ServiceGeneral.formatTime(endK-startK)}")
            endNsm = time.time()
            ServiceGeneral.logWithTime(f"Completed NOSW, NOEV d={density}, r={radius}, nsm={nsm.value} in {ServiceGeneral.formatTime(endNsm-startNsm)}")
        endRadius = time.time()
        ServiceGeneral.logWithTime(f"Completed NOSW, NOEV d={density}, r={radius} in {ServiceGeneral.formatTime(endRadius-startRadius)}")
    endDensity = time.time()
    ServiceGeneral.logWithTime(f"Completed NOSW, NOEV d={density} in {ServiceGeneral.formatTime(endDensity-startDensity)}")

                    
# ------------------------ NOSW, 1EV ----------------------------------
for density in densities:
    ServiceGeneral.logWithTime(f"Starting NOSW, 1EV d={density}")
    startDensity = time.time()
    n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
    for radius in radii:
        ServiceGeneral.logWithTime(f"Starting NOSW, 1EV d={density}, r={radius}")
        startRadius = time.time()
        for nsm in nsmsReduced:
            ServiceGeneral.logWithTime(f"Starting NOSW, 1EV d={density}, r={radius}, nsm={nsm.value}")
            startNsm = time.time()
            for k in ks:
                ServiceGeneral.logWithTime(f"Starting NOSW, 1EV d={density}, r={radius}, nsm={nsm.value}, k={k}")
                startK = time.time()
                for eventEffect in eventEffects:
                    ServiceGeneral.logWithTime(f"Starting NOSW, 1EV d={density}, r={radius}, nsm={nsm.value}, k={k}, ee={eventEffect.label}")
                    startEvent = time.time()
                    for initialCondition in initialConditions:
                        ServiceGeneral.logWithTime(f"Starting NOSW, 1EV d={density}, r={radius}, nsm={nsm.value}, k={k}, ee={eventEffect.label}, init={initialCondition}")
                        startIC = time.time()
                        for i in range(iStart, iStop):
                            startI = time.time()
                            ServiceGeneral.logWithTime(f"Starting NOSW, 1EV d={density}, r={radius}, nsm={nsm.value}, k={k}, ee={eventEffect.label}, init={initialCondition}, i={i}")
                            
                            event = ExternalStimulusOrientationChangeEvent(startTimestep=eventStart,
                                               duration=eventDuration,  
                                               domainSize=domainSize, 
                                               eventEffect=eventEffect, 
                                               distributionType=eventDistributionType, 
                                               areas=[[domainSize[0]/2, domainSize[1]/2, radius]],
                                               angle=eventAngle,
                                               noisePercentage=noise,
                                               radius=radius,
                                               numberOfAffected=eventNumberAffected,
                                               eventSelectionType=eventSelectionType
                                               )
                            events = [event]
                            
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

                            savePath = f"{saveLocation}local_nosw_1ev_d={density}_r={radius}_{initialCondition}_nsm={nsm.value}_k={k}_ee={eventEffect.val}_{i}.json"
                            ServiceSavedModel.saveModel(simulationData=simulationData, path=savePath, 
                                                        modelParams=simulator.getParameterSummary(), switchValues=switchTypeValues)
                            endI = time.time()
                            ServiceGeneral.logWithTime(f"Completed NOSW, 1EV d={density}, r={radius}, nsm={nsm.value}, k={k}, ee={eventEffect.label}, init={initialCondition}, i={i} in {ServiceGeneral.formatTime(endI-startI)}")
                        endIC = time.time()
                        ServiceGeneral.logWithTime(f"Completed NOSW, 1EV d={density}, r={radius}, nsm={nsm.value}, k={k}, ee={eventEffect.label}, init={initialCondition} in {ServiceGeneral.formatTime(endIC-startIC)}")
                    endEvent = time.time()
                    ServiceGeneral.logWithTime(f"Completed NOSW, 1EV d={density}, r={radius}, nsm={nsm.value}, k={k}, ee={eventEffect.label} in {ServiceGeneral.formatTime(endEvent-startEvent)}")
                endK = time.time()
                ServiceGeneral.logWithTime(f"Completed NOSW, 1EV d={density}, r={radius}, nsm={nsm.value}, k={k} in {ServiceGeneral.formatTime(endK-startK)}")
            endNsm = time.time()
            ServiceGeneral.logWithTime(f"Completed NOSW, 1EV d={density}, r={radius}, nsm={nsm.value} in {ServiceGeneral.formatTime(endNsm-startNsm)}")
        endRadius = time.time()
        ServiceGeneral.logWithTime(f"Completed NOSW, 1EV d={density}, r={radius} in {ServiceGeneral.formatTime(endRadius-startRadius)}")
    endDensity = time.time()
    ServiceGeneral.logWithTime(f"Completed NOSW, 1EV d={density} in {ServiceGeneral.formatTime(endDensity-startDensity)}")

            
# ------------------------ NSMSW, NOEV --------------------------------
for density in densities:
    ServiceGeneral.logWithTime(f"Starting NSMSW, NOEV d={density}")
    startDensity = time.time()
    n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
    for radius in radii:
        ServiceGeneral.logWithTime(f"Starting NSMSW, NOEV d={density}, r={radius}")
        startRadius = time.time()
        for nsmCombo in nsmCombos:
            ServiceGeneral.logWithTime(f"Starting NSMSW, NOEV d={density}, r={radius}, nsmCombo={nsmCombo[0].value}-{nsmCombo[1].value}")
            startNsm = time.time()
            for k in ks:
                ServiceGeneral.logWithTime(f"Starting NSMSW, NOEV d={density}, r={radius}, nsmCombo={nsmCombo[0].value}-{nsmCombo[1].value}, k={k}")
                startK = time.time()
                for initialCondition in initialConditions:
                    ServiceGeneral.logWithTime(f"Starting NSMSW, NOEV d={density}, r={radius}, nsmCombo={nsmCombo[0].value}-{nsmCombo[1].value}, k={k}, init={initialCondition}")
                    startIC = time.time()
                    for i in range(iStart, iStop):
                        startI = time.time()
                        ServiceGeneral.logWithTime(f"Starting NSMSW, NOEV d={density}, r={radius}, nsmCombo={nsmCombo[0].value}-{nsmCombo[1].value}, k={k}, init={initialCondition}, i={i}")
                        
                        events = []
                        infoNsm = SwitchInformation(switchType=SwitchType.NEIGHBOUR_SELECTION_MECHANISM, 
                            values=nsmCombo,
                            thresholds=threshold,
                            numberPreviousStepsForThreshold=numberOfPreviousSteps
                            )
                        switchSummary = SwitchSummary([infoNsm])

                        if initialCondition == "ordered":
                            startValue = nsmCombo[0]
                            initialState = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(None, domainSize, n, angleX=initialAngleX, angleY=initialAngleY)
                        else:
                            startValue = nsmCombo[1]
                            initialState = (None, None, None)

                        simulator = VicsekWithNeighbourSelection(domainSize=domainSize,
                                                                radius=radius,
                                                                noise=noise,
                                                                numberOfParticles=n,
                                                                k=k,
                                                                neighbourSelectionMechanism=startValue,
                                                                speed=speed,
                                                                switchSummary=switchSummary,
                                                                degreesOfVision=degreesOfVision,
                                                                events=events,
                                                                colourType=colourType,
                                                                thresholdEvaluationMethod=thresholdEvaluationMethod,
                                                                updateIfNoNeighbours=updateIfNoNeighbours)
                        simulationData, switchTypeValues = simulator.simulate(initialState=initialState, tmax=tmax)

                        savePath = f"{saveLocation}local_nsmsw_noev_d={density}_r={radius}_{initialCondition}_st={startValue.value}_nsmCombo={nsmCombo[0].value}-{nsmCombo[1].value}_k={k}_{i}.json"
                        ServiceSavedModel.saveModel(simulationData=simulationData, path=savePath, 
                                                    modelParams=simulator.getParameterSummary(), switchValues=switchTypeValues)
                        endI = time.time()
                        ServiceGeneral.logWithTime(f"Completed NSMSW, NOEV d={density}, r={radius}, nsmCombo={nsmCombo[0].value}-{nsmCombo[1].value}, k={k}, init={initialCondition}, i={i} in {ServiceGeneral.formatTime(endI-startI)}")
                    endIC = time.time()
                    ServiceGeneral.logWithTime(f"Completed NSMSW, NOEV d={density}, r={radius}, nsmCombo={nsmCombo[0].value}-{nsmCombo[1].value}, k={k}, init={initialCondition} in {ServiceGeneral.formatTime(endIC-startIC)}")
                endK = time.time()
                ServiceGeneral.logWithTime(f"Completed NSMSW, NOEV d={density}, r={radius}, nsmCombo={nsmCombo[0].value}-{nsmCombo[1].value}, k={k} in {ServiceGeneral.formatTime(endK-startK)}")
            endNsm = time.time()
            ServiceGeneral.logWithTime(f"Completed NSMSW, NOEV d={density}, r={radius}, nsmCombo={nsmCombo[0].value}-{nsmCombo[1].value} in {ServiceGeneral.formatTime(endNsm-startNsm)}")
        endRadius = time.time()
        ServiceGeneral.logWithTime(f"Completed NSMSW, NOEV d={density}, r={radius} in {ServiceGeneral.formatTime(endRadius-startRadius)}")
    endDensity = time.time()
    ServiceGeneral.logWithTime(f"Completed NSMSW, NOEV d={density} in {ServiceGeneral.formatTime(endDensity-startDensity)}")

# ------------------------ NSMSW, 1EV ---------------------------------
for density in densities:
    ServiceGeneral.logWithTime(f"Starting NSMSW, 1EV d={density}")
    startDensity = time.time()
    n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
    for radius in radii:
        ServiceGeneral.logWithTime(f"Starting NSMSW, 1EV d={density}, r={radius}")
        startRadius = time.time()
        for nsmCombo in nsmCombos:
            ServiceGeneral.logWithTime(f"Starting NSMSW, 1EV d={density}, r={radius}, nsmCombo={nsmCombo[0].value}-{nsmCombo[1].value}")
            startNsm = time.time()
            for k in ks:
                ServiceGeneral.logWithTime(f"Starting NSMSW, 1EV d={density}, r={radius}, nsmCombo={nsmCombo[0].value}-{nsmCombo[1].value}, k={k}")
                startK = time.time()
                for eventEffect in eventEffects:
                    ServiceGeneral.logWithTime(f"Starting NSMSW, 1EV d={density}, r={radius}, nsmCombo={nsmCombo[0].value}-{nsmCombo[1].value}, k={k}, ee={eventEffect.val}")
                    startEvent = time.time()
                    for initialCondition in initialConditions:
                        ServiceGeneral.logWithTime(f"Starting NSMSW, 1EV d={density}, r={radius}, nsmCombo={nsmCombo[0].value}-{nsmCombo[1].value}, k={k}, ee={eventEffect.val}, init={initialCondition}")
                        startIC = time.time()
                        for i in range(iStart, iStop):
                            startI = time.time()
                            ServiceGeneral.logWithTime(f"Starting NSMSW, 1EV d={density}, r={radius}, nsmCombo={nsmCombo[0].value}-{nsmCombo[1].value}, k={k}, ee={eventEffect.val}, init={initialCondition}, i={i}")
                                
                            event = ExternalStimulusOrientationChangeEvent(startTimestep=eventStart,
                                                duration=eventDuration,  
                                                domainSize=domainSize, 
                                                eventEffect=eventEffect, 
                                                distributionType=eventDistributionType, 
                                                areas=[[domainSize[0]/2, domainSize[1]/2, radius]],
                                                angle=eventAngle,
                                                noisePercentage=noise,
                                                radius=radius,
                                                numberOfAffected=eventNumberAffected,
                                                eventSelectionType=eventSelectionType
                                                )
                            events = [event]

                            infoNsm = SwitchInformation(switchType=SwitchType.NEIGHBOUR_SELECTION_MECHANISM, 
                                values=nsmCombo,
                                thresholds=threshold,
                                numberPreviousStepsForThreshold=numberOfPreviousSteps
                                )
                            switchSummary = SwitchSummary([infoNsm])

                            if initialCondition == "ordered":
                                startValue = nsmCombo[0]
                                initialState = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(None, domainSize, n, angleX=initialAngleX, angleY=initialAngleY)
                            else:
                                startValue = nsmCombo[1]
                                initialState = (None, None, None)

                            simulator = VicsekWithNeighbourSelection(domainSize=domainSize,
                                                                    radius=radius,
                                                                    noise=noise,
                                                                    numberOfParticles=n,
                                                                    k=k,
                                                                    neighbourSelectionMechanism=startValue,
                                                                    speed=speed,
                                                                    switchSummary=switchSummary,
                                                                    degreesOfVision=degreesOfVision,
                                                                    events=events,
                                                                    colourType=colourType,
                                                                    thresholdEvaluationMethod=thresholdEvaluationMethod,
                                                                    updateIfNoNeighbours=updateIfNoNeighbours)
                            simulationData, switchTypeValues = simulator.simulate(initialState=initialState, tmax=tmax)

                            savePath = f"{saveLocation}local_nsmsw_1ev_d={density}_r={radius}_{initialCondition}_st={startValue.value}_nsmCombo={nsmCombo[0].value}-{nsmCombo[1].value}_k={k}_ee={eventEffect.val}_{i}.json"
                            ServiceSavedModel.saveModel(simulationData=simulationData, path=savePath, 
                                                        modelParams=simulator.getParameterSummary(), switchValues=switchTypeValues)
                            endI = time.time()
                            ServiceGeneral.logWithTime(f"Completed NSMSW, 1EV d={density}, r={radius}, nsmCombo={nsmCombo[0].value}-{nsmCombo[1].value}, k={k}, ee={eventEffect.val}, init={initialCondition}, i={i} in {ServiceGeneral.formatTime(endI-startI)}")
                        endIC = time.time()
                        ServiceGeneral.logWithTime(f"Completed NSMSW, 1EV d={density}, r={radius}, nsmCombo={nsmCombo[0].value}-{nsmCombo[1].value}, k={k}, ee={eventEffect.val}, init={initialCondition} in {ServiceGeneral.formatTime(endIC-startIC)}")
                    endEvent = time.time()
                    ServiceGeneral.logWithTime(f"Completed NSMSW, 1EV d={density}, r={radius}, nsmCombo={nsmCombo[0].value}-{nsmCombo[1].value}, k={k}, ee={eventEffect.val} in {ServiceGeneral.formatTime(endEvent-startEvent)}")
                endK = time.time()
                ServiceGeneral.logWithTime(f"Completed NSMSW, 1EV d={density}, r={radius}, nsmCombo={nsmCombo[0].value}-{nsmCombo[1].value}, k={k} in {ServiceGeneral.formatTime(endK-startK)}")
            endNsm = time.time()
            ServiceGeneral.logWithTime(f"Completed NSMSW, 1EV d={density}, r={radius}, nsmCombo={nsmCombo[0].value}-{nsmCombo[1].value} in {ServiceGeneral.formatTime(endNsm-startNsm)}")
        endRadius = time.time()
        ServiceGeneral.logWithTime(f"Completed NSMSW, 1EV d={density}, r={radius} in {ServiceGeneral.formatTime(endRadius-startRadius)}")
    endDensity = time.time()
    ServiceGeneral.logWithTime(f"Completed NSMSW, 1EV d={density} in {ServiceGeneral.formatTime(endDensity-startDensity)}")

# ------------------------ KSW, NOEV ----------------------------------
for density in densities:
    ServiceGeneral.logWithTime(f"Starting KSW, NOEV d={density}")
    startDensity = time.time()
    n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
    for radius in radii:
        ServiceGeneral.logWithTime(f"Starting KSW, NOEV d={density}, r={radius}")
        startRadius = time.time()
        for nsm in nsmsReduced:
            ServiceGeneral.logWithTime(f"Starting KSW, NOEV d={density}, r={radius}, nsm={nsm.value}")
            startNsm = time.time()
            for kCombo in kCombos:
                ServiceGeneral.logWithTime(f"Starting KSW, NOEV d={density}, r={radius}, nsm={nsm.value}, kCombo={kCombo[0]}-{kCombo[1]}")
                startK = time.time()
                for initialCondition in initialConditions:
                    ServiceGeneral.logWithTime(f"Starting KSW, NOEV d={density}, r={radius}, nsm={nsm.value}, kCombo={kCombo[0]}-{kCombo[1]}, init={initialCondition}")
                    startIC = time.time()
                    for i in range(iStart, iStop):
                        startI = time.time()
                        ServiceGeneral.logWithTime(f"Starting KSW, NOEV d={density}, r={radius}, nsm={nsm.value}, kCombo={kCombo[0]}-{kCombo[1]}, init={initialCondition}, i={i}")
                        
                        events = []
                        infoK = SwitchInformation(switchType=SwitchType.K, 
                            values=kCombo,
                            thresholds=threshold,
                            numberPreviousStepsForThreshold=numberOfPreviousSteps
                            )
                        switchSummary = SwitchSummary([infoK])

                        if initialCondition == "ordered":
                            startValue = kCombo[0]
                            initialState = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(None, domainSize, n, angleX=initialAngleX, angleY=initialAngleY)
                        else:
                            startValue = kCombo[1]
                            initialState = (None, None, None)

                        simulator = VicsekWithNeighbourSelection(domainSize=domainSize,
                                                                radius=radius,
                                                                noise=noise,
                                                                numberOfParticles=n,
                                                                k=startValue,
                                                                neighbourSelectionMechanism=nsm,
                                                                speed=speed,
                                                                switchSummary=switchSummary,
                                                                degreesOfVision=degreesOfVision,
                                                                events=events,
                                                                colourType=colourType,
                                                                thresholdEvaluationMethod=thresholdEvaluationMethod,
                                                                updateIfNoNeighbours=updateIfNoNeighbours)
                        simulationData, switchTypeValues = simulator.simulate(initialState=initialState, tmax=tmax)

                        savePath = f"{saveLocation}local_ksw_noev_d={density}_r={radius}_{initialCondition}_st={startValue}_nsm={nsm.value}_kCombo={kCombo[0]}-{kCombo[1]}_{i}.json"
                        ServiceSavedModel.saveModel(simulationData=simulationData, path=savePath, 
                                                    modelParams=simulator.getParameterSummary(), switchValues=switchTypeValues)
                        endI = time.time()
                        ServiceGeneral.logWithTime(f"Completed KSW, NOEV d={density}, r={radius}, nsm={nsm.value}, kCombo={kCombo[0]}-{kCombo[1]}, init={initialCondition}, i={i} in {ServiceGeneral.formatTime(endI-startI)}")
                    endIC = time.time()
                    ServiceGeneral.logWithTime(f"Completed KSW, NOEV d={density}, r={radius}, nsm={nsm.value}, kCombo={kCombo[0]}-{kCombo[1]}, init={initialCondition} in {ServiceGeneral.formatTime(endIC-startIC)}")
                endK = time.time()
                ServiceGeneral.logWithTime(f"Completed KSW, NOEV d={density}, r={radius}, nsm={nsm.value}, kCombo={kCombo[0]}-{kCombo[1]} in {ServiceGeneral.formatTime(endK-startK)}")
            endNsm = time.time()
            ServiceGeneral.logWithTime(f"Completed KSW, NOEV d={density}, r={radius}, nsm={nsm.value} in {ServiceGeneral.formatTime(endNsm-startNsm)}")
        endRadius = time.time()
        ServiceGeneral.logWithTime(f"Completed KSW, NOEV d={density}, r={radius} in {ServiceGeneral.formatTime(endRadius-startRadius)}")
    endDensity = time.time()
    ServiceGeneral.logWithTime(f"Completed KSW, NOEV d={density} in {ServiceGeneral.formatTime(endDensity-startDensity)}")

# ------------------------ KSW, 1EV -----------------------------------
for density in densities:
    ServiceGeneral.logWithTime(f"Starting KSW, 1EV d={density}")
    startDensity = time.time()
    n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
    for radius in radii:
        ServiceGeneral.logWithTime(f"Starting KSW, 1EV d={density}, r={radius}")
        startRadius = time.time()
        for nsm in nsmsReduced:
            ServiceGeneral.logWithTime(f"Starting KSW, 1EV d={density}, r={radius}, nsm={nsm.value}")
            startNsm = time.time()
            for kCombo in kCombos:
                ServiceGeneral.logWithTime(f"Starting KSW, 1EV d={density}, r={radius}, nsm={nsm.value}, kCombo={kCombo[0]}-{kCombo[1]}")
                startK = time.time()
                for eventEffect in eventEffects:
                    ServiceGeneral.logWithTime(f"Starting KSW, 1EV d={density}, r={radius}, nsm={nsm.value}, kCombo={kCombo[0]}-{kCombo[1]}, ee={eventEffect.val}")
                    startEvent = time.time()
                    for initialCondition in initialConditions:
                        ServiceGeneral.logWithTime(f"Starting KSW, 1EV d={density}, r={radius}, nsm={nsm.value}, kCombo={kCombo[0]}-{kCombo[1]}, ee={eventEffect.val}, init={initialCondition}")
                        startIC = time.time()
                        for i in range(iStart, iStop):
                            startI = time.time()
                            ServiceGeneral.logWithTime(f"Starting KSW, 1EV d={density}, r={radius}, nsm={nsm.value}, kCombo={kCombo[0]}-{kCombo[1]}, ee={eventEffect.val}, init={initialCondition}, i={i}")
                            
                            event = ExternalStimulusOrientationChangeEvent(startTimestep=eventStart,
                                                duration=eventDuration,  
                                                domainSize=domainSize, 
                                                eventEffect=eventEffect, 
                                                distributionType=eventDistributionType, 
                                                areas=[[domainSize[0]/2, domainSize[1]/2, radius]],
                                                angle=eventAngle,
                                                noisePercentage=noise,
                                                radius=radius,
                                                numberOfAffected=eventNumberAffected,
                                                eventSelectionType=eventSelectionType
                                                )
                            events = [event]
                            infoK = SwitchInformation(switchType=SwitchType.K, 
                                values=kCombo,
                                thresholds=threshold,
                                numberPreviousStepsForThreshold=numberOfPreviousSteps
                                )
                            switchSummary = SwitchSummary([infoK])

                            if initialCondition == "ordered":
                                startValue = kCombo[0]
                                initialState = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(None, domainSize, n, angleX=initialAngleX, angleY=initialAngleY)
                            else:
                                startValue = kCombo[1]
                                initialState = (None, None, None)

                            simulator = VicsekWithNeighbourSelection(domainSize=domainSize,
                                                                    radius=radius,
                                                                    noise=noise,
                                                                    numberOfParticles=n,
                                                                    k=startValue,
                                                                    neighbourSelectionMechanism=nsm,
                                                                    speed=speed,
                                                                    switchSummary=switchSummary,
                                                                    degreesOfVision=degreesOfVision,
                                                                    events=events,
                                                                    colourType=colourType,
                                                                    thresholdEvaluationMethod=thresholdEvaluationMethod,
                                                                    updateIfNoNeighbours=updateIfNoNeighbours)
                            simulationData, switchTypeValues = simulator.simulate(initialState=initialState, tmax=tmax)

                            savePath = f"{saveLocation}local_ksw_1ev_d={density}_r={radius}_{initialCondition}_st={startValue}_nsm={nsm.value}_kCombo={kCombo[0]}-{kCombo[1]}_ee={eventEffect.val}_{i}.json"
                            ServiceSavedModel.saveModel(simulationData=simulationData, path=savePath, 
                                                        modelParams=simulator.getParameterSummary(), switchValues=switchTypeValues)
                            endI = time.time()
                            ServiceGeneral.logWithTime(f"Completed KSW, 1EV d={density}, r={radius}, nsm={nsm.value}, kCombo={kCombo[0]}-{kCombo[1]}, ee={eventEffect.val}, init={initialCondition}, i={i} in {ServiceGeneral.formatTime(endI-startI)}")
                        endIC = time.time()
                        ServiceGeneral.logWithTime(f"Completed KSW, 1EV d={density}, r={radius}, nsm={nsm.value}, kCombo={kCombo[0]}-{kCombo[1]}, ee={eventEffect.val}, init={initialCondition} in {ServiceGeneral.formatTime(endIC-startIC)}")
                    endEvent = time.time()
                    ServiceGeneral.logWithTime(f"Completed KSW, 1EV d={density}, r={radius}, nsm={nsm.value}, kCombo={kCombo[0]}-{kCombo[1]}, ee={eventEffect.val} in {ServiceGeneral.formatTime(endK-startK)}")
                endK = time.time()
                ServiceGeneral.logWithTime(f"Completed KSW, 1EV d={density}, r={radius}, nsm={nsm.value}, kCombo={kCombo[0]}-{kCombo[1]} in {ServiceGeneral.formatTime(endK-startK)}")
            endNsm = time.time()
            ServiceGeneral.logWithTime(f"Completed KSW, 1EV d={density}, r={radius}, nsm={nsm.value} in {ServiceGeneral.formatTime(endNsm-startNsm)}")
        endRadius = time.time()
        ServiceGeneral.logWithTime(f"Completed KSW, 1EV d={density}, r={radius} in {ServiceGeneral.formatTime(endRadius-startRadius)}")
    endDensity = time.time()
    ServiceGeneral.logWithTime(f"Completed KSW, 1EV d={density} in {ServiceGeneral.formatTime(endDensity-startDensity)}")

endOverall = time.time()
ServiceGeneral.logWithTime(f"Completed run in {ServiceGeneral.formatTime(endOverall-startOverall)}")
