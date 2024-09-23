import time
import numpy as np

from VicsekIndividuals import VicsekWithNeighbourSelection
from EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from EnumSwitchType import SwitchType
from ExternalStimulusEvent import ExternalStimulusOrientationChangeEvent
from EnumEventEffect import EventEffect
from EnumDistributionType import DistributionType

import ServicePreparation
import ServiceGeneral
import ServiceSavedModel


domainSize = (22.36, 22.36)
noisePercentage = 1
noise = ServicePreparation.getNoiseAmplitudeValueForPercentage(noisePercentage)
#noise = 0
n = 100
speed = 1

# TODO: check why N-F always returns to order

radius = 50
k = 1
nsm = NeighbourSelectionMechanism.NEAREST
"""
switchType = SwitchType.NEIGHBOUR_SELECTION_MODE
switchValues = (NeighbourSelectionMechanism.FARTHEST,NeighbourSelectionMechanism.NEAREST)
"""
switchType = SwitchType.K
switchValues = (5,1)

tmax = 10000

threshold = [0.1]

event = ExternalStimulusOrientationChangeEvent(startTimestep=500,
                                               duration=499,  
                                               domainSize=domainSize, 
                                               eventEffect=EventEffect.ALIGN_TO_FIXED_ANGLE, 
                                               distributionType=DistributionType.LOCAL_SINGLE_SITE, 
                                               areas=[[11, 11, radius]],
                                               angle=np.pi,
                                               noisePercentage=1,
                                               )

tstart = time.time()

ServiceGeneral.logWithTime("start")

#initialState = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(switchValues[0], domainSize, n, angleX=0.5, angleY=0.5)
simulator = VicsekWithNeighbourSelection(domainSize=domainSize,
                                         radius=radius,
                                         noise=noise,
                                         numberOfParticles=n,
                                         k=k,
                                         neighbourSelectionMechanism=nsm,
                                         speed=speed,
                                         orderThresholds=threshold,
                                         numberPreviousStepsForThreshold=100,
                                         switchingActive=True,
                                         switchType=switchType,
                                         switchValues=switchValues)
#simulationData, switchTypeValues = simulator.simulate(initialState=initialState, tmax=tmax, events=[event])
simulationData, switchTypeValues = simulator.simulate(tmax=tmax, events=[event])

ServiceSavedModel.saveModel(simulationData=simulationData, path="test.json", 
                            modelParams=simulator.getParameterSummary())

tend = time.time()
ServiceGeneral.logWithTime(f"duration: {ServiceGeneral.formatTime(tend-tstart)}")