
import os
import time
import numpy as np

from concurrent.futures import ProcessPoolExecutor, as_completed

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

# Variant of neighbour_selection_data.py covering only the switching combinations (k-switching and
# neighbour-selection-mechanism switching, with and without an event), with updateIfNoNeighbours=False:
# an individual that currently has no other neighbours within its perception radius keeps its current
# switch value instead of updating it. This exists to compare against the main sweep's default
# behaviour (updateIfNoNeighbours=True), under which an isolated individual's local order is computed
# from itself alone and is therefore always exactly 1.0 - i.e. it is always nudged towards the
# order-inducing switch value regardless of the swarm's actual state. Output paths are marked with
# "_blockiso" so they don't collide with the main sweep's data.

def runSimulationJob(simulator, initialState, tmax, label):
    """
    Runs a single simulation to completion. Executed in a worker process.

    np.random.seed(None) re-seeds the (legacy) global numpy RNG from OS entropy. Worker
    processes are forked and would otherwise all inherit the exact same RNG state from the
    parent at fork time, which would make their random draws (e.g. random starting
    positions/orientations) correlated across workers instead of independent as they are
    when everything runs sequentially in a single process.
    """
    np.random.seed(None)
    jobStart = time.time()
    if initialState is not None:
        simulator.simulate(initialState=initialState, tmax=tmax)
    else:
        simulator.simulate(tmax=tmax)
    return label, time.time() - jobStart


def runJobs(executor, jobs, sectionLabel):
    """
    Submits all jobs collected for a section to the executor and logs completions as they arrive.
    """
    sectionStart = time.time()
    futures = [executor.submit(runSimulationJob, simulator, initialState, tmax, label)
               for simulator, initialState, tmax, label in jobs]
    for future in as_completed(futures):
        label, elapsed = future.result()
        ServiceGeneral.logWithTime(f"Completed {label} in {ServiceGeneral.formatTime(elapsed)}")
    ServiceGeneral.logWithTime(f"completed {sectionLabel}: {ServiceGeneral.formatTime(time.time()-sectionStart)}")


def main():
    # ----------------------------- GENERAL - base values -------------------------------------------------------------------
    tmaxLocal = 15000

    noisePercentage = 1
    noise = ServicePreparation.getNoiseAmplitudeValueForPercentage(noisePercentage)

    domainSize = (50, 50)
    numberPreviousSteps = 100
    speed = 1

    eventStartTimestep = 5000
    eventDuration = 1000
    distributionType = DistributionType.LOCAL_SINGLE_SITE

    densities = [0.01, 0.06, 0.09]
    radii = [5, 10, 20]
    threshold_options = [0.1, 0.2, 0.3, 0.4, 0.5]
    # SwitchInformation requires each threshold entry to be a sequence (it calls len() on it).
    switchThresholdOptions = [[t] for t in threshold_options]
    eventEffects = [EventEffect.ALIGN_TO_FIXED_ANGLE,
                    EventEffect.AWAY_FROM_ORIGIN,
                    EventEffect.RANDOM]

    reducedNeighbourSelectionMechanisms = [NeighbourSelectionMechanism.NEAREST,
                                            NeighbourSelectionMechanism.FARTHEST,
                                            NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE,
                                            NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE]

    ks = [1, 5]
    nsmCombos = [[NeighbourSelectionMechanism.NEAREST, NeighbourSelectionMechanism.FARTHEST],
                [NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE, NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE]]

    saveInterval = 100

    tstart = time.time()
    baseDataLocation = "/media/lilly/WD Elements/data_july26/"

    numberOfWorkers = os.cpu_count()
    ServiceGeneral.logWithTime(f"using a process pool of {numberOfWorkers} workers")

    with ProcessPoolExecutor(max_workers=numberOfWorkers) as executor:

        for j in range(0, 51, 10):
            iStart = j+1
            iStop = j + 11

            # ----------------------------- LOCAL - k switching with event, blocked while isolated ---------------------------
            ServiceGeneral.logWithTime("start LOCAL - k switching with event, blocked while isolated")
            tmax = tmaxLocal

            jobs = []
            for density in densities:
                n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
                for radius in radii:
                    for nsm in reducedNeighbourSelectionMechanisms:
                        kCombo = (5,1)
                        for thresholds in switchThresholdOptions:
                            kSwitch = SwitchInformation(switchType=SwitchType.K,
                                                        values=kCombo,
                                                        thresholds=thresholds,
                                                        numberPreviousStepsForThreshold=numberPreviousSteps)
                            switchSummary = SwitchSummary([kSwitch])
                            for eventEffect in eventEffects:
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
                                    if startingCondition == "ordered":
                                        startValue = kCombo[0]
                                    else:
                                        startValue = kCombo[1]
                                    for i in range(iStart, iStop):
                                        savePath = f"{baseDataLocation}local_ksw_1ev_blockiso_{startingCondition}_st={startValue}_d={density}_n={n}_r={radius}_nsm={nsm.value}_kCombo={kCombo[0]}-{kCombo[1]}_th={thresholds}_ee={eventEffect.val}_{i}"
                                        simulator = VicsekWithNeighbourSelection(domainSize=domainSize,
                                                                                radius=radius,
                                                                                noise=noise,
                                                                                numberOfParticles=n,
                                                                                k=startValue,
                                                                                neighbourSelectionMechanism=nsm,
                                                                                speed=speed,
                                                                                switchSummary=switchSummary,
                                                                                degreesOfVision=np.pi*2,
                                                                                events=[event],
                                                                                logPath=savePath,
                                                                                logInterval=saveInterval,
                                                                                returnHistories=False,
                                                                                updateIfNoNeighbours=False)
                                        label = f"KSW BLOCKISO d={density}, r={radius}, nsm={nsm.value}, thresholds={thresholds}, ee={eventEffect.val}, st={startingCondition}, i={i}"
                                        if startingCondition == "ordered":
                                            initialState = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(None, domainSize, n)
                                        else:
                                            initialState = None
                                        jobs.append((simulator, initialState, tmax, label))

            runJobs(executor, jobs, "k switching with event, blocked while isolated")

            # ----------------------------- LOCAL - nsm switching with event, blocked while isolated -------------------------
            ServiceGeneral.logWithTime("start LOCAL - nsm switching with event, blocked while isolated")
            tmax = tmaxLocal

            jobs = []
            for density in densities:
                n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
                for radius in radii:
                    for nsmCombo in nsmCombos:
                        for k in ks:
                            for thresholds in switchThresholdOptions:
                                nsmSwitch = SwitchInformation(switchType=SwitchType.NEIGHBOUR_SELECTION_MECHANISM,
                                                            values=nsmCombo,
                                                            thresholds=thresholds,
                                                            numberPreviousStepsForThreshold=numberPreviousSteps)
                                switchSummary = SwitchSummary([nsmSwitch])
                                for eventEffect in eventEffects:
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
                                        if startingCondition == "ordered":
                                            startValue = nsmCombo[0]
                                        else:
                                            startValue = nsmCombo[1]
                                        for i in range(iStart, iStop):
                                            savePath = f"{baseDataLocation}local_nsmsw_1ev_blockiso_{startingCondition}_st={startValue.value}_d={density}_n={n}_r={radius}_nsmCombo={nsmCombo[0].value}-{nsmCombo[1].value}_k={k}_th={thresholds}_ee={eventEffect.val}_{i}"
                                            simulator = VicsekWithNeighbourSelection(domainSize=domainSize,
                                                                                    radius=radius,
                                                                                    noise=noise,
                                                                                    numberOfParticles=n,
                                                                                    k=k,
                                                                                    neighbourSelectionMechanism=startValue,
                                                                                    speed=speed,
                                                                                    switchSummary=switchSummary,
                                                                                    degreesOfVision=np.pi*2,
                                                                                    events=[event],
                                                                                    logPath=savePath,
                                                                                    logInterval=saveInterval,
                                                                                    returnHistories=False,
                                                                                    updateIfNoNeighbours=False)
                                            label = f"NSMSW BLOCKISO d={density}, r={radius}, nsmCombo={nsmCombo[0].value}-{nsmCombo[1].value}, k={k}, thresholds={thresholds}, ee={eventEffect.val}, st={startingCondition}, i={i}"
                                            if startingCondition == "ordered":
                                                initialState = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(None, domainSize, n)
                                            else:
                                                initialState = None
                                            jobs.append((simulator, initialState, tmax, label))

            runJobs(executor, jobs, "nsm switching with event, blocked while isolated")

            # ----------------------------- LOCAL - k switching without event, blocked while isolated -------------------------
            ServiceGeneral.logWithTime("start LOCAL - k switching without event, blocked while isolated")
            tmax = tmaxLocal

            jobs = []
            for density in densities:
                n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
                for radius in radii:
                    for nsm in reducedNeighbourSelectionMechanisms:
                        kCombo = (5,1)
                        for thresholds in switchThresholdOptions:
                            kSwitch = SwitchInformation(switchType=SwitchType.K,
                                                        values=kCombo,
                                                        thresholds=thresholds,
                                                        numberPreviousStepsForThreshold=numberPreviousSteps)
                            switchSummary = SwitchSummary([kSwitch])
                            for startingCondition in ["ordered", "random"]:
                                if startingCondition == "ordered":
                                    startValue = kCombo[0]
                                else:
                                    startValue = kCombo[1]
                                for i in range(iStart, iStop):
                                    savePath = f"{baseDataLocation}local_ksw_noev_blockiso_{startingCondition}_st={startValue}_d={density}_n={n}_r={radius}_nsm={nsm.value}_kCombo={kCombo[0]}-{kCombo[1]}_th={thresholds}_{i}"
                                    simulator = VicsekWithNeighbourSelection(domainSize=domainSize,
                                                                            radius=radius,
                                                                            noise=noise,
                                                                            numberOfParticles=n,
                                                                            k=startValue,
                                                                            neighbourSelectionMechanism=nsm,
                                                                            speed=speed,
                                                                            switchSummary=switchSummary,
                                                                            degreesOfVision=np.pi*2,
                                                                            events=[],
                                                                            logPath=savePath,
                                                                            logInterval=saveInterval,
                                                                            returnHistories=False,
                                                                            updateIfNoNeighbours=False)
                                    label = f"KSW NOEV BLOCKISO d={density}, r={radius}, nsm={nsm.value}, thresholds={thresholds}, st={startingCondition}, i={i}"
                                    if startingCondition == "ordered":
                                        initialState = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(None, domainSize, n)
                                    else:
                                        initialState = None
                                    jobs.append((simulator, initialState, tmax, label))

            runJobs(executor, jobs, "k switching without event, blocked while isolated")

            # ----------------------------- LOCAL - nsm switching without event, blocked while isolated -----------------------
            ServiceGeneral.logWithTime("start LOCAL - nsm switching without event, blocked while isolated")
            tmax = tmaxLocal

            jobs = []
            for density in densities:
                n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
                for radius in radii:
                    for nsmCombo in nsmCombos:
                        for k in ks:
                            for thresholds in switchThresholdOptions:
                                nsmSwitch = SwitchInformation(switchType=SwitchType.NEIGHBOUR_SELECTION_MECHANISM,
                                                            values=nsmCombo,
                                                            thresholds=thresholds,
                                                            numberPreviousStepsForThreshold=numberPreviousSteps)
                                switchSummary = SwitchSummary([nsmSwitch])
                                for startingCondition in ["ordered", "random"]:
                                    if startingCondition == "ordered":
                                        startValue = nsmCombo[0]
                                    else:
                                        startValue = nsmCombo[1]
                                    for i in range(iStart, iStop):
                                        savePath = f"{baseDataLocation}local_nsmsw_noev_blockiso_{startingCondition}_st={startValue.value}_d={density}_n={n}_r={radius}_nsmCombo={nsmCombo[0].value}-{nsmCombo[1].value}_k={k}_th={thresholds}_{i}"
                                        simulator = VicsekWithNeighbourSelection(domainSize=domainSize,
                                                                                radius=radius,
                                                                                noise=noise,
                                                                                numberOfParticles=n,
                                                                                k=k,
                                                                                neighbourSelectionMechanism=startValue,
                                                                                speed=speed,
                                                                                switchSummary=switchSummary,
                                                                                degreesOfVision=np.pi*2,
                                                                                events=[],
                                                                                logPath=savePath,
                                                                                logInterval=saveInterval,
                                                                                returnHistories=False,
                                                                                updateIfNoNeighbours=False)
                                        label = f"NSMSW NOEV BLOCKISO d={density}, r={radius}, nsmCombo={nsmCombo[0].value}-{nsmCombo[1].value}, k={k}, thresholds={thresholds}, st={startingCondition}, i={i}"
                                        if startingCondition == "ordered":
                                            initialState = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(None, domainSize, n)
                                        else:
                                            initialState = None
                                        jobs.append((simulator, initialState, tmax, label))

            runJobs(executor, jobs, "nsm switching without event, blocked while isolated")

        tend = time.time()
        ServiceGeneral.logWithTime(f"duration: {ServiceGeneral.formatTime(tend-tstart)}")


if __name__ == "__main__":
    main()
