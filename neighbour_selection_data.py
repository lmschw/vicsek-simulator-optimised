
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

domainSize = (100, 100)
numberPreviousSteps = 100
speed = 1

densities = [0.01, 0.05, 0.09]
radii = [5, 10, 20]
ks = [1, 5]

iStart = 1
iStop = 11


# ----------------------------- GLOBAL - no switching, no events --------------------------------------------------------

tstart = time.time()

ServiceGeneral.logWithTime("start GLOBAL - no switching, no events")
tmax = tmaxGlobal
switchSummary = None

for density in densities:
    dStart = time.time()
    n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
    for radius in radii:
        rStart = time.time()
        for nsm in [NeighbourSelectionMechanism.NEAREST, 
                    NeighbourSelectionMechanism.FARTHEST, 
                    NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE,
                    NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE,
                    NeighbourSelectionMechanism.ALL]:
            nsmStart = time.time()
            for k in ks:
                kStart = time.time()
                for startingCondition in ["ordered", "random"]:
                    stStart = time.time()
                    for i in range(iStart, iStop):
                        iStart = time.time()
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
                                                                events=[])
                        if startingCondition == "ordered":
                            initialState = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(None, domainSize, n)
                            simulationData, switchTypeValues = simulator.simulate(initialState=initialState, tmax=tmax)
                        else:
                            simulationData, switchTypeValues = simulator.simulate(tmax=tmax)

                        savePath = f"global_nosw_noev_{startingCondition}_d={density}_n={n}_r={radius}_nsm={nsm.value}_k={k}_{i}.json"
                        ServiceSavedModel.saveModel(simulationData=simulationData, path=savePath, 
                                                    modelParams=simulator.getParameterSummary())
                        iEnd = time.time()
                        ServiceGeneral.logWithTime(f"Completed d={density}, r={radius}, nsm={nsm.value}, k={k}, st={startingCondition}, i={i} in {ServiceGeneral.formatTime(iEnd-iStart)}")
                    stEnd = time.time()
                    ServiceGeneral.logWithTime(f"Completed d={density}, r={radius}, nsm={nsm.value}, k={k}, st={startingCondition} in {ServiceGeneral.formatTime(stEnd-stStart)}")
                kEnd = time.time()
                ServiceGeneral.logWithTime(f"Completed d={density}, r={radius}, nsm={nsm.value}, k={k} in {ServiceGeneral.formatTime(kEnd-kStart)}")
            nsmEnd = time.time()
            ServiceGeneral.logWithTime(f"Completed d={density}, r={radius}, nsm={nsm.value} in {ServiceGeneral.formatTime(nsmEnd-nsmStart)}")
        rEnd = time.time()
        ServiceGeneral.logWithTime(f"Completed d={density}, r={radius} in {ServiceGeneral.formatTime(rEnd-rStart)}")
    dEnd = time.time()
    ServiceGeneral.logWithTime(f"Completed d={density} in {ServiceGeneral.formatTime(kEnd-kStart)}")

tend = time.time()
ServiceGeneral.logWithTime(f"duration: {ServiceGeneral.formatTime(tend-tstart)}")
# ----------------------------- LOCAL - no switching with event ---------------------------------------------------------
# ----------------------------- LOCAL - k switching with event ----------------------------------------------------------
# ----------------------------- LOCAL - nsm switching with event --------------------------------------------------------