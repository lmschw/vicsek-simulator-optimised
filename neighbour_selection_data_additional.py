
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


# This script runs a large number of independent Vicsek simulations (one per parameter
# combination). Each simulation is CPU-bound and does not depend on any other, so they
# are executed in a process pool to use all available CPU cores instead of one at a time.
# The simulation logic itself (VicsekWithNeighbourSelection.simulate) is untouched, so the
# behaviour of any single run is identical to running it on its own.

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
    tmaxGlobal = 3000
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
    ks = [1, 5]
    threshold_options = [0.1, 0.2, 0.3, 0.4, 0.5]
    # Used only for the switching sections below: SwitchInformation requires each threshold
    # entry to be a sequence (it calls len() on it), matching the already-fixed sibling
    # scripts (e.g. neighbour_selection_data_run_ksw_lod.py).
    switchThresholdOptions = [[t] for t in threshold_options]
    eventEffects = [EventEffect.ALIGN_TO_FIXED_ANGLE,
                    EventEffect.AWAY_FROM_ORIGIN,
                    EventEffect.RANDOM]

    neighbourSelectionMechanisms = [NeighbourSelectionMechanism.ALL,
                                    NeighbourSelectionMechanism.RANDOM,
                                    NeighbourSelectionMechanism.NEAREST,
                                    NeighbourSelectionMechanism.FARTHEST,
                                    NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE,
                                    NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE]
    reducedNeighbourSelectionMechanisms = [NeighbourSelectionMechanism.NEAREST,
                                            NeighbourSelectionMechanism.FARTHEST,
                                            NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE,
                                            NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE]

    nsmCombos = [[NeighbourSelectionMechanism.NEAREST, NeighbourSelectionMechanism.FARTHEST],
                [NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE, NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE]]

    saveInterval = 1

    tstart = time.time()
    baseDataLocation = "/media/lilly/WD Elements/data_july26/"

    numberOfWorkers = os.cpu_count()
    ServiceGeneral.logWithTime(f"using a process pool of {numberOfWorkers} workers")


    with ProcessPoolExecutor(max_workers=numberOfWorkers) as executor:
        for j in range(0, 51, 10):
            iStart = j+1
            iStop = j + 11  
            saveInterval = 100

    # ------------------------- ADDITIONAL K VALUES ------------------------------------------------------
            ServiceGeneral.logWithTime("start GLOBAL - no switching, no events")
            tmax = tmaxGlobal
            switchSummary = None

            jobs = []
            for density in densities:
                n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
                for radius in radii:
                    for nsm in neighbourSelectionMechanisms:
                        for k in [0,2,3,4]:
                            for startingCondition in ["ordered", "random"]:
                                for i in range(iStart, iStop):
                                    savePath = f"{baseDataLocation}global_nosw_noev_{startingCondition}_d={density}_n={n}_r={radius}_nsm={nsm.value}_k={k}_{i}"
                                    simulator = VicsekWithNeighbourSelection(domainSize=domainSize,
                                                                            radius=radius,
                                                                            noise=noise,
                                                                            numberOfParticles=n,
                                                                            k=k,
                                                                            neighbourSelectionMechanism=nsm,
                                                                            speed=speed,
                                                                            degreesOfVision=np.pi*2,
                                                                            logPath=savePath,
                                                                            logInterval=saveInterval,
                                                                            returnHistories=False)
                                    label = f"GLOBAL d={density}, r={radius}, nsm={nsm.value}, k={k}, st={startingCondition}, i={i}"
                                    if startingCondition == "ordered":
                                        initialState = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(None, domainSize, n)
                                    else:
                                        initialState = None
                                    jobs.append((simulator, initialState, tmax, label))

            runJobs(executor, jobs, "GLOBAL - nosw, noev")



    # ------------------------------ HOD SWITCHING ---------------------------------------------
            tmax = tmaxLocal
            switchSummary = None
            nsm = NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE

            jobs = []
            for density in [0.09, 0.06]:
                n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
                for radius in [10]:
                    for k in ks:
                        for thresholds in threshold_options:
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
                                    for i in range(iStart, iStop):
                                        savePath = f"{baseDataLocation}local_nosw_1ev_{startingCondition}_d={density}_n={n}_r={radius}_nsm={nsm.value}_k={k}_th={thresholds}_ee={eventEffect.val}_{i}"

                                        simulator = VicsekWithNeighbourSelection(domainSize=domainSize,
                                                                                radius=radius,
                                                                                noise=noise,
                                                                                numberOfParticles=n,
                                                                                k=k,
                                                                                neighbourSelectionMechanism=nsm,
                                                                                speed=speed,
                                                                                switchSummary=switchSummary,
                                                                                degreesOfVision=np.pi*2,
                                                                                events=[event],
                                                                                logPath=savePath,
                                                                                logInterval=saveInterval,
                                                                                returnHistories=False)
                                        label = f"NOSW d={density}, r={radius}, nsm={nsm.value}, k={k}, thresholds={thresholds}, ee={eventEffect.val}, st={startingCondition}, i={i}"
                                        if startingCondition == "ordered":
                                            initialState = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(None, domainSize, n)
                                        else:
                                            initialState = None
                                        jobs.append((simulator, initialState, tmax, label))

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
                                    savePath = f"{baseDataLocation}local_ksw_1ev_{startingCondition}_st={startValue}_d={density}_n={n}_r={radius}_nsm={nsm.value}_kCombo={kCombo[0]}-{kCombo[1]}_th={thresholds}_ee={eventEffect.val}_{i}"
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
                                                                            returnHistories=False)
                                    label = f"KSW d={density}, r={radius}, nsm={nsm.value}, thresholds={thresholds}, ee={eventEffect.val}, st={startingCondition}, i={i}"
                                    if startingCondition == "ordered":
                                        initialState = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(None, domainSize, n)
                                    else:
                                        initialState = None
                                    jobs.append((simulator, initialState, tmax, label))
            runJobs(executor, jobs, "hod events")


        tend = time.time()
        ServiceGeneral.logWithTime(f"duration: {ServiceGeneral.formatTime(tend-tstart)}")


if __name__ == "__main__":
    main()
