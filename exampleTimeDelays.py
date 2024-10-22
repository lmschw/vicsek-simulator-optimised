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


domainSize = (25, 25)
noisePercentage = 1
noise = ServicePreparation.getNoiseAmplitudeValueForPercentage(noisePercentage)

n = 10
speed = 1

threshold = 0.1
numberOfPreviousSteps = 100

radius = 100
k = 1
nsm = NeighbourSelectionMechanism.NEAREST

infoNsm = SwitchInformation(switchType=SwitchType.NEIGHBOUR_SELECTION_MECHANISM, 
                            values=(NeighbourSelectionMechanism.FARTHEST, NeighbourSelectionMechanism.NEAREST),
                            thresholds=[threshold],
                            numberPreviousStepsForThreshold=numberOfPreviousSteps
                            )

infoK = SwitchInformation(switchType=SwitchType.K, 
                        values=(5, 1),
                        thresholds=[threshold],
                        numberPreviousStepsForThreshold=numberOfPreviousSteps
                        )

infoSpeed = SwitchInformation(switchType=SwitchType.SPEED, 
                        values=(1, 0.1),
                        thresholds=[threshold],
                        numberPreviousStepsForThreshold=numberOfPreviousSteps
                        )

infoAtd = SwitchInformation(switchType=SwitchType.ACTIVATION_TIME_DELAY,
                            values=(1,5),
                            thresholds=[threshold],
                            numberPreviousStepsForThreshold=numberOfPreviousSteps)

switchSummary = SwitchSummary([infoAtd])

"""
switchType = SwitchType.NEIGHBOUR_SELECTION_MECHANISM
switchValues = (NeighbourSelectionMechanism.FARTHEST,NeighbourSelectionMechanism.NEAREST)
"""

"""
switchType = SwitchType.K
switchValues = (5,1)
"""
"""
switchType = SwitchType.SPEED
switchValues = (0.1, 1)
"""

#timeDelays = [1, 2, 3, 1, 2, 3, 1, 2, 3, 2]
timeDelays = np.ones(n)

tmax = 5000

threshold = [0.1]

event = ExternalStimulusOrientationChangeEvent(startTimestep=1000,
                                               duration=1000,  
                                               domainSize=domainSize, 
                                               eventEffect=EventEffect.ALIGN_TO_FIXED_ANGLE, 
                                               distributionType=DistributionType.LOCAL_SINGLE_SITE, 
                                               areas=[[domainSize[0]/2, domainSize[1]/2, radius]],
                                               angle=np.pi,
                                               noisePercentage=1,
                                               radius=radius,
                                               eventSelectionType=EventSelectionType.RANDOM
                                               )

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
                                         switchSummary=switchSummary,
                                         degreesOfVision=np.pi*2,
                                         activationTimeDelays=timeDelays,
                                         events=[event])
simulationData, switchTypeValues = simulator.simulate(initialState=initialState, tmax=tmax)
#simulationData, switchTypeValues = simulator.simulate(tmax=tmax)

ServiceSavedModel.saveModel(simulationData=simulationData, path="test.json", 
                            modelParams=simulator.getParameterSummary(), switchValues=switchTypeValues)

tend = time.time()
ServiceGeneral.logWithTime(f"duration: {ServiceGeneral.formatTime(tend-tstart)}")