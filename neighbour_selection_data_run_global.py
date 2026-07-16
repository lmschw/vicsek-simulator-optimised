
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
threshold_options = [0.1, 0.2, 0.3, 0.4, 0.5]
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

iStart = 1
iStop = 51
saveInterval = 1

tstart = time.time()
baseDataLocation = "D:/data_may26/"

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
        for nsm in neighbourSelectionMechanisms:
            nsmStart = time.time()
            for k in ks:
                kStart = time.time()
                for startingCondition in ["ordered", "random"]:
                    stStart = time.time()
                    for i in range(iStart, iStop):
                        iStartTime = time.time()
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
                        if startingCondition == "ordered":
                            initialState = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(None, domainSize, n)
                            simulator.simulate(initialState=initialState, tmax=tmax)
                        else:
                            simulator.simulate(tmax=tmax)
                        iEnd = time.time()
                        ServiceGeneral.logWithTime(f"Completed GLOBAL d={density}, r={radius}, nsm={nsm.value}, k={k}, st={startingCondition}, i={i} in {ServiceGeneral.formatTime(iEnd-iStartTime)}")
                    stEnd = time.time()
                    ServiceGeneral.logWithTime(f"Completed GLOBAL d={density}, r={radius}, nsm={nsm.value}, k={k}, st={startingCondition} in {ServiceGeneral.formatTime(stEnd-stStart)}")
                kEnd = time.time()
                ServiceGeneral.logWithTime(f"Completed GLOBAL d={density}, r={radius}, nsm={nsm.value}, k={k} in {ServiceGeneral.formatTime(kEnd-kStart)}")
            nsmEnd = time.time()
            ServiceGeneral.logWithTime(f"Completed GLOBAL d={density}, r={radius}, nsm={nsm.value} in {ServiceGeneral.formatTime(nsmEnd-nsmStart)}")
        rEnd = time.time()
        ServiceGeneral.logWithTime(f"Completed GLOBAL d={density}, r={radius} in {ServiceGeneral.formatTime(rEnd-rStart)}")
    dEnd = time.time()
    ServiceGeneral.logWithTime(f"Completed GLOBAL d={density} in {ServiceGeneral.formatTime(dEnd-dStart)}")

tendGlobal = time.time()
ServiceGeneral.logWithTime(f"completed GLOBAL - nosw, noev: {ServiceGeneral.formatTime(tendGlobal-tstartGlobal)}")

tend = time.time()
ServiceGeneral.logWithTime(f"duration: {ServiceGeneral.formatTime(tend-tstart)}")