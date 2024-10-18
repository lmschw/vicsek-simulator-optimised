
import time
import numpy as np

from model.VicsekIndividualsMultiSwitch import VicsekWithNeighbourSelection
from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from enums.EnumSwitchType import SwitchType
from events.ExternalStimulusEvent import ExternalStimulusOrientationChangeEvent
from enums.EnumEventEffect import EventEffect
from enums.EnumDistributionType import DistributionType
from enums.EnumEventSelectionType import EventSelectionType

from model.SwitchInformation import SwitchInformation
from model.SwitchSummary import SwitchSummary

import services.ServicePreparation as ServicePreparation
import services.ServiceGeneral as ServiceGeneral
import services.ServiceSavedModel as ServiceSavedModel


# ----------------------------- GENERAL - base values -------------------------------------------------------------------
tmaxGlobal = 3000
tmaxLocal = 15000

noisePercentage = 1
noise = ServicePreparation.getNoiseAmplitudeValueForPercentage(noisePercentage)

domainSize = (50, 50)
numberPreviousSteps = 100
speed = 1
thresholds = [0.1]

eventStartTimestep = 5000
eventDuration = 1000
distributionType = DistributionType.LOCAL_SINGLE_SITE

densities = [0.01, 0.05, 0.09]
radii = [5, 10, 20]
ks = [1, 5]
eventEffects = [EventEffect.ALIGN_TO_FIXED_ANGLE,
                EventEffect.AWAY_FROM_ORIGIN,
                EventEffect.RANDOM]

iStart = 1
iStop = 11

tstart = time.time()
baseDataLocation = "D:/data/new/"

# ----------------------------- GLOBAL - no switching, no events --------------------------------------------------------

tstartGlobal = time.time()
ServiceGeneral.logWithTime("start GLOBAL - no switching, no events")
tmax = tmaxGlobal
switchSummary = None


for density in densities:
    dStart = time.time()
    n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
    for radius in radii:
        rStart = time.time()
        for nsm in [NeighbourSelectionMechanism.RANDOM,
                    NeighbourSelectionMechanism.ALL,
                    NeighbourSelectionMechanism.NEAREST, 
                    NeighbourSelectionMechanism.FARTHEST, 
                    NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE,
                    NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE]:
            nsmStart = time.time()
            for k in ks:
                kStart = time.time()
                for startingCondition in ["ordered", "random"]:
                    stStart = time.time()
                    for i in range(iStart, iStop):
                        iStartTime = time.time()
                        ServiceGeneral.logWithTime(f"Starting d={density}, r={radius}, nsm={nsm.value}, k={k}, st={startingCondition}, i={i}")
                        simulator = VicsekWithNeighbourSelection(domainSize=domainSize,
                                                                radius=radius,
                                                                noise=noise,
                                                                numberOfParticles=n,
                                                                k=k,
                                                                neighbourSelectionMechanism=nsm,
                                                                speed=speed,
                                                                degreesOfVision=np.pi*2)
                        if startingCondition == "ordered":
                            initialState = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(None, domainSize, n)
                            simulationData, switchTypeValues = simulator.simulate(initialState=initialState, tmax=tmax)
                        else:
                            simulationData, switchTypeValues = simulator.simulate(tmax=tmax)

                        savePath = f"{baseDataLocation}global_nosw_noev_{startingCondition}_d={density}_n={n}_r={radius}_nsm={nsm.value}_k={k}_{i}.json"
                        ServiceSavedModel.saveModel(simulationData=simulationData, path=savePath, 
                                                    modelParams=simulator.getParameterSummary())
                        iEnd = time.time()
                        ServiceGeneral.logWithTime(f"Completed d={density}, r={radius}, nsm={nsm.value}, k={k}, st={startingCondition}, i={i} in {ServiceGeneral.formatTime(iEnd-iStartTime)}")
                    stEnd = time.time()
                    ServiceGeneral.logWithTime(f"Completed d={density}, r={radius}, nsm={nsm.value}, k={k}, st={startingCondition} in {ServiceGeneral.formatTime(stEnd-stStart)}")
                kEnd = time.time()
                ServiceGeneral.logWithTime(f"Completed d={density}, r={radius}, nsm={nsm.value}, k={k} in {ServiceGeneral.formatTime(kEnd-kStart)}")
            nsmEnd = time.time()
            ServiceGeneral.logWithTime(f"Completed d={density}, r={radius}, nsm={nsm.value} in {ServiceGeneral.formatTime(nsmEnd-nsmStart)}")
        rEnd = time.time()
        ServiceGeneral.logWithTime(f"Completed d={density}, r={radius} in {ServiceGeneral.formatTime(rEnd-rStart)}")
    dEnd = time.time()
    ServiceGeneral.logWithTime(f"Completed d={density} in {ServiceGeneral.formatTime(dEnd-dStart)}")

tendGlobal = time.time()
ServiceGeneral.logWithTime(f"completed GLOBAL - nosw, noev: {ServiceGeneral.formatTime(tendGlobal-tstartGlobal)}")


# ----------------------------- LOCAL - no switching with event ---------------------------------------------------------
tstartLocalNoswEv = time.time()
ServiceGeneral.logWithTime("start LOCAL - no switching with event")
tmax = tmaxLocal
switchSummary = None

for density in densities:
    dStart = time.time()
    n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
    for radius in radii:
        rStart = time.time()
        for nsm in [NeighbourSelectionMechanism.ALL,
                    NeighbourSelectionMechanism.RANDOM,
                    NeighbourSelectionMechanism.NEAREST, 
                    NeighbourSelectionMechanism.FARTHEST, 
                    NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE,
                    NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE]:
            nsmStart = time.time()
            for k in ks:
                kStart = time.time()
                for eventEffect in eventEffects:
                    eeStart = time.time()
                    areas = [domainSize[0]/2, domainSize[1]/2, radius]
                    event = ExternalStimulusOrientationChangeEvent(startTimestep=eventStartTimestep,
                                                                   duration=eventDuration,
                                                                   domainSize=domainSize,
                                                                   eventEffect=eventEffect,
                                                                   distributionType=distributionType,
                                                                   areas=[areas],
                                                                   radius=radius,
                                                                   angle=np.pi
                                                                   )
                    for startingCondition in ["ordered", "random"]:
                        stStart = time.time()
                        for i in range(iStart, iStop):
                            iStartTime = time.time()
                            ServiceGeneral.logWithTime(f"Starting d={density}, r={radius}, nsm={nsm.value}, k={k}, st={startingCondition}, i={i}")
                            simulator = VicsekWithNeighbourSelection(domainSize=domainSize,
                                                                    radius=radius,
                                                                    noise=noise,
                                                                    numberOfParticles=n,
                                                                    k=k,
                                                                    neighbourSelectionMechanism=nsm,
                                                                    speed=speed,
                                                                    switchSummary=switchSummary,
                                                                    degreesOfVision=np.pi*2,
                                                                    events=[event])
                            if startingCondition == "ordered":
                                initialState = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(None, domainSize, n)
                                simulationData, switchTypeValues = simulator.simulate(initialState=initialState, tmax=tmax)
                            else:
                                simulationData, switchTypeValues = simulator.simulate(tmax=tmax)

                            savePath = f"{baseDataLocation}local_nosw_1ev_{startingCondition}_d={density}_n={n}_r={radius}_nsm={nsm.value}_k={k}_ee={eventEffect.val}_{i}.json"
                            ServiceSavedModel.saveModel(simulationData=simulationData, path=savePath, 
                                                        modelParams=simulator.getParameterSummary())
                            iEnd = time.time()
                            ServiceGeneral.logWithTime(f"Completed d={density}, r={radius}, nsm={nsm.value}, k={k}, ee={eventEffect.val}, st={startingCondition}, i={i} in {ServiceGeneral.formatTime(iEnd-iStartTime)}")
                        stEnd = time.time()
                        ServiceGeneral.logWithTime(f"Completed d={density}, r={radius}, nsm={nsm.value}, k={k}, ee={eventEffect.val}, st={startingCondition} in {ServiceGeneral.formatTime(stEnd-stStart)}")
                    eeEnd = time.time()
                    ServiceGeneral.logWithTime(f"Completed d={density}, r={radius}, nsm={nsm.value}, k={k}, ee={eventEffect.val} in {ServiceGeneral.formatTime(eeEnd-eeStart)}")
                kEnd = time.time()
                ServiceGeneral.logWithTime(f"Completed d={density}, r={radius}, nsm={nsm.value}, k={k} in {ServiceGeneral.formatTime(kEnd-kStart)}")
            nsmEnd = time.time()
            ServiceGeneral.logWithTime(f"Completed d={density}, r={radius}, nsm={nsm.value} in {ServiceGeneral.formatTime(nsmEnd-nsmStart)}")
        rEnd = time.time()
        ServiceGeneral.logWithTime(f"Completed d={density}, r={radius} in {ServiceGeneral.formatTime(rEnd-rStart)}")
    dEnd = time.time()
    ServiceGeneral.logWithTime(f"Completed d={density} in {ServiceGeneral.formatTime(dEnd-dStart)}")

tendLocalNoswEv = time.time()
ServiceGeneral.logWithTime(f"completed LOCAL - no switching with event: {ServiceGeneral.formatTime(tendLocalNoswEv-tstartLocalNoswEv)}")

# ----------------------------- LOCAL - k switching with event ----------------------------------------------------------
tstartLocalKswEv = time.time()
ServiceGeneral.logWithTime("start LOCAL - k switching with event")
tmax = tmaxLocal

for density in densities:
    dStart = time.time()
    n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
    for radius in radii:
        rStart = time.time()
        for nsm in [NeighbourSelectionMechanism.NEAREST, 
                    NeighbourSelectionMechanism.FARTHEST, 
                    NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE,
                    NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE]:
            nsmStart = time.time()
            kCombo = (5,1)
            kSwitch = SwitchInformation(switchType=SwitchType.K,
                                        values=kCombo,
                                        thresholds=thresholds,
                                        numberPreviousStepsForThreshold=numberPreviousSteps)
            switchSummary = SwitchSummary([kSwitch])

            for eventEffect in eventEffects:
                eeStart = time.time()
                areas = [domainSize[0]/2, domainSize[1]/2, radius]
                event = ExternalStimulusOrientationChangeEvent(startTimestep=eventStartTimestep,
                                                                duration=eventDuration,
                                                                domainSize=domainSize,
                                                                eventEffect=eventEffect,
                                                                distributionType=distributionType,
                                                                areas=[areas],
                                                                radius=radius,
                                                                angle=np.pi
                                                                )
                for startingCondition in ["ordered", "random"]:
                    stStart = time.time()
                    if startingCondition == "ordered":
                        startValue = kCombo[0]
                    else:
                        startValue = kCombo[1]
                    for i in range(iStart, iStop):
                        iStartTime = time.time()
                        ServiceGeneral.logWithTime(f"Starting d={density}, r={radius}, nsm={nsm.value}, st={startingCondition}, i={i}")
                        simulator = VicsekWithNeighbourSelection(domainSize=domainSize,
                                                                radius=radius,
                                                                noise=noise,
                                                                numberOfParticles=n,
                                                                k=startValue,
                                                                neighbourSelectionMechanism=nsm,
                                                                speed=speed,
                                                                switchSummary=switchSummary,
                                                                degreesOfVision=np.pi*2,
                                                                events=[event])
                        if startingCondition == "ordered":
                            initialState = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(None, domainSize, n)
                            simulationData, switchTypeValues = simulator.simulate(initialState=initialState, tmax=tmax)
                        else:
                            simulationData, switchTypeValues = simulator.simulate(tmax=tmax)

                        savePath = f"{baseDataLocation}local_ksw_1ev_{startingCondition}_st={startValue}_d={density}_n={n}_r={radius}_nsm={nsm.value}_kCombo={kCombo[0]}-{kCombo[1]}_ee={eventEffect.val}_{i}.json"
                        ServiceSavedModel.saveModel(simulationData=simulationData, path=savePath, 
                                                    modelParams=simulator.getParameterSummary())
                        iEnd = time.time()
                        ServiceGeneral.logWithTime(f"Completed d={density}, r={radius}, nsm={nsm.value},  ee={eventEffect.val}, st={startingCondition}, i={i} in {ServiceGeneral.formatTime(iEnd-iStartTime)}")
                    stEnd = time.time()
                    ServiceGeneral.logWithTime(f"Completed d={density}, r={radius}, nsm={nsm.value}, ee={eventEffect.val}, st={startingCondition} in {ServiceGeneral.formatTime(stEnd-stStart)}")
                eeEnd = time.time()
                ServiceGeneral.logWithTime(f"Completed d={density}, r={radius}, nsm={nsm.value},ee={eventEffect.val} in {ServiceGeneral.formatTime(eeEnd-eeStart)}")
            nsmEnd = time.time()
            ServiceGeneral.logWithTime(f"Completed d={density}, r={radius}, nsm={nsm.value} in {ServiceGeneral.formatTime(nsmEnd-nsmStart)}")
        rEnd = time.time()
        ServiceGeneral.logWithTime(f"Completed d={density}, r={radius} in {ServiceGeneral.formatTime(rEnd-rStart)}")
    dEnd = time.time()
    ServiceGeneral.logWithTime(f"Completed d={density} in {ServiceGeneral.formatTime(dEnd-dStart)}")
tendLocalKswEv = time.time()
ServiceGeneral.logWithTime(f"completed k switching with event: {ServiceGeneral.formatTime(tendLocalKswEv-tstartLocalKswEv)}")

# ----------------------------- LOCAL - nsm switching with event --------------------------------------------------------
tstartLocalNsmswEv = time.time()
ServiceGeneral.logWithTime("start LOCAL - nsm switching with event")
tmax = tmaxLocal

for density in densities:
    dStart = time.time()
    n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
    for radius in radii:
        rStart = time.time()
        for nsmCombo in [(NeighbourSelectionMechanism.NEAREST, NeighbourSelectionMechanism.FARTHEST), 
                    (NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE, NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE)]:
            nsmStart = time.time()
            nsmSwitch = SwitchInformation(switchType=SwitchType.NEIGHBOUR_SELECTION_MECHANISM,
                                        values=nsmCombo,
                                        thresholds=thresholds,
                                        numberPreviousStepsForThreshold=numberPreviousSteps)
            switchSummary = SwitchSummary([nsmSwitch])

            for k in ks:
                kStart = time.time()

                for eventEffect in eventEffects:
                    eeStart = time.time()
                    areas = [domainSize[0]/2, domainSize[1]/2, radius]
                    event = ExternalStimulusOrientationChangeEvent(startTimestep=eventStartTimestep,
                                                                    duration=eventDuration,
                                                                    domainSize=domainSize,
                                                                    eventEffect=eventEffect,
                                                                    distributionType=distributionType,
                                                                    areas=[areas],
                                                                    radius=radius,
                                                                    angle=np.pi
                                                                    )
                    
                    for startingCondition in ["ordered", "random"]:
                        stStart = time.time()
                        if startingCondition == "ordered":
                            startValue = nsmCombo[0]
                        else:
                            startValue = nsmCombo[1]
                        for i in range(iStart, iStop):
                            iStartTime = time.time()
                            ServiceGeneral.logWithTime(f"Starting d={density}, r={radius}, nsmCombo={nsmCombo[0].value}-{nsmCombo[1].value} k={k}, st={startingCondition}, i={i}")
                            simulator = VicsekWithNeighbourSelection(domainSize=domainSize,
                                                                    radius=radius,
                                                                    noise=noise,
                                                                    numberOfParticles=n,
                                                                    k=k,
                                                                    neighbourSelectionMechanism=startValue,
                                                                    speed=speed,
                                                                    switchSummary=switchSummary,
                                                                    degreesOfVision=np.pi*2,
                                                                    events=[event])
                            if startingCondition == "ordered":
                                initialState = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(None, domainSize, n)
                                simulationData, switchTypeValues = simulator.simulate(initialState=initialState, tmax=tmax)
                            else:
                                simulationData, switchTypeValues = simulator.simulate(tmax=tmax)

                            savePath = f"{baseDataLocation}local_nsmsw_1ev_{startingCondition}_st={startValue.value}_d={density}_n={n}_r={radius}_nsmCombo={nsmCombo[0].value}-{nsmCombo[1]}_k={k}_ee={eventEffect.val}_{i}.json"
                            ServiceSavedModel.saveModel(simulationData=simulationData, path=savePath, 
                                                        modelParams=simulator.getParameterSummary())
                            iEnd = time.time()
                            ServiceGeneral.logWithTime(f"Completed d={density}, r={radius}, nsm={nsm.value}, k={k}, ee={eventEffect.val}, st={startingCondition}, i={i} in {ServiceGeneral.formatTime(iEnd-iStartTime)}")
                        stEnd = time.time()
                        ServiceGeneral.logWithTime(f"Completed d={density}, r={radius}, nsm={nsm.value}, k={k}, ee={eventEffect.val}, st={startingCondition} in {ServiceGeneral.formatTime(stEnd-stStart)}")
                    eeEnd = time.time()
                    ServiceGeneral.logWithTime(f"Completed d={density}, r={radius}, nsm={nsm.value}, k={k}, ee={eventEffect.val} in {ServiceGeneral.formatTime(eeEnd-eeStart)}")
                kEnd = time.time()
                ServiceGeneral.logWithTime(f"Completed d={density}, r={radius}, nsm={nsm.value}, k={k} in {ServiceGeneral.formatTime(kEnd-kStart)}")
            nsmEnd = time.time()
            ServiceGeneral.logWithTime(f"Completed d={density}, r={radius}, nsm={nsm.value} in {ServiceGeneral.formatTime(nsmEnd-nsmStart)}")
        rEnd = time.time()
        ServiceGeneral.logWithTime(f"Completed d={density}, r={radius} in {ServiceGeneral.formatTime(rEnd-rStart)}")
    dEnd = time.time()
    ServiceGeneral.logWithTime(f"Completed d={density} in {ServiceGeneral.formatTime(kEnd-kStart)}")

tendLocalNsmswEv = time.time()
ServiceGeneral.logWithTime(f"completed nsm switching with event: {ServiceGeneral.formatTime(tendLocalNsmswEv-tstartLocalNsmswEv)}")


tend = time.time()
ServiceGeneral.logWithTime(f"duration: {ServiceGeneral.formatTime(tend-tstart)}")