
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

eventStartTimestep = 5000
eventDuration = 1000
distributionType = DistributionType.LOCAL_SINGLE_SITE

densities = [0.01, 0.06, 0.09]
radii = [5, 10, 20]
ks = [1, 5]
threshold_options = [[0.1], [0.2], [0.3], [0.4], [0.5]]
eventEffects = [EventEffect.ALIGN_TO_FIXED_ANGLE,
                EventEffect.AWAY_FROM_ORIGIN,
                EventEffect.RANDOM]

neighbourSelectionMechanisms = [NeighbourSelectionMechanism.ALL,
                                NeighbourSelectionMechanism.RANDOM,
                                NeighbourSelectionMechanism.NEAREST,
                                NeighbourSelectionMechanism.FARTHEST,
                                NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE,
                                NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE]
reducedNeighbourSelectionMechanisms = [NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE]

nsmCombos = [[NeighbourSelectionMechanism.NEAREST, NeighbourSelectionMechanism.FARTHEST],
            [NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE, NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE]]

iStart = 1
iStop = 51

tstart = time.time()
baseDataLocation = "D:/data_may26/"

saveInterval = 50

# ----------------------------- LOCAL - k switching with event ----------------------------------------------------------
tstartLocalKswEv = time.time()
ServiceGeneral.logWithTime("start LOCAL - k switching with event")
tmax = tmaxLocal

for density in densities:
    dStart = time.time()
    n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
    for radius in radii:
        rStart = time.time()
        for nsm in reducedNeighbourSelectionMechanisms:
            nsmStart = time.time()
            for thresholds in threshold_options:
                thresholdStart = time.time()
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
                            if startingCondition == "ordered":
                                initialState = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(None, domainSize, n)
                                simulator.simulate(initialState=initialState, tmax=tmax)
                            else:
                                simulator.simulate(tmax=tmax)
                            iEnd = time.time()
                            ServiceGeneral.logWithTime(f"Completed KSW d={density}, r={radius}, nsm={nsm.value}, thresholds={thresholds}, ee={eventEffect.val}, st={startingCondition}, i={i} in {ServiceGeneral.formatTime(iEnd-iStartTime)}")
                        stEnd = time.time()
                        ServiceGeneral.logWithTime(f"Completed KSW d={density}, r={radius}, nsm={nsm.value}, thresholds={thresholds}, ee={eventEffect.val}, st={startingCondition} in {ServiceGeneral.formatTime(stEnd-stStart)}")
                    eeEnd = time.time()
                    ServiceGeneral.logWithTime(f"Completed KSW d={density}, r={radius}, nsm={nsm.value}, thresholds={thresholds},ee={eventEffect.val} in {ServiceGeneral.formatTime(eeEnd-eeStart)}")
                thresholdEnd = time.time()
                ServiceGeneral.logWithTime(f"Completed KSW d={density}, r={radius}, nsm={nsm.value}, thresholds={thresholds} in {ServiceGeneral.formatTime(thresholdEnd-thresholdStart)}")
            nsmEnd = time.time()
            ServiceGeneral.logWithTime(f"Completed KSW d={density}, r={radius}, nsm={nsm.value} in {ServiceGeneral.formatTime(nsmEnd-nsmStart)}")
        rEnd = time.time()
        ServiceGeneral.logWithTime(f"Completed KSW d={density}, r={radius} in {ServiceGeneral.formatTime(rEnd-rStart)}")
    dEnd = time.time()
    ServiceGeneral.logWithTime(f"Completed KSW d={density} in {ServiceGeneral.formatTime(dEnd-dStart)}")
tendLocalKswEv = time.time()
ServiceGeneral.logWithTime(f"completed k switching with event: {ServiceGeneral.formatTime(tendLocalKswEv-tstartLocalKswEv)}")

tend = time.time()
ServiceGeneral.logWithTime(f"duration: {ServiceGeneral.formatTime(tend-tstart)}")