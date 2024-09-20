import time

from VicsekIndividuals import VicsekWithNeighbourSelection
from EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism

import ServicePreparation
import ServiceGeneral
import ServiceSavedModel

#noise = 0.063
noise = 0.063
domainSize = (100, 100)
n = 300
speed = 1

radius = 20
k = 1
nsm = NeighbourSelectionMechanism.NEAREST

noise = 0.063
domainSize = (100, 100)
n = 100
speed = 1

radius = 50
k = 1
nsm = NeighbourSelectionMechanism.NEAREST

tmax = 10000

threshold = [0.1]

tstart = time.time()

ServiceGeneral.logWithTime("start")

initialState = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(None, domainSize, n, angleX=0.5, angleY=0.5)
simulator = VicsekWithNeighbourSelection(domainSize=domainSize,
                                         radius=radius,
                                         noise=noise,
                                         numberOfParticles=n,
                                         k=k,
                                         neighbourSelectionMechanism=nsm,
                                         speed=speed,
                                         orderThresholds=threshold,
                                         numberPreviousStepsForThreshold=10,
                                         switchingActive=False)
simulationData, switchTypeValues = simulator.simulate(initialState=initialState, tmax=tmax)

ServiceSavedModel.saveModel(simulationData=simulationData, path="test.json", 
                            modelParams=simulator.getParameterSummary())

tend = time.time()
ServiceGeneral.logWithTime(f"duration: {ServiceGeneral.formatTime(tend-tstart)}")