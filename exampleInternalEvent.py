import time
import numpy as np

from model.VicsekIndividualsMultiSwitch import VicsekWithNeighbourSelection
from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from enums.EnumSwitchType import SwitchType
from events.InternalStimulusEvent import InternalStimulusOrientationChangeEvent
from enums.EnumEventEffect import InternalEventEffect
from enums.EnumDistributionType import DistributionType

from model.SwitchInformation import SwitchInformation
from model.SwitchSummary import SwitchSummary

import services.ServicePreparation as ServicePreparation
import services.ServiceGeneral as ServiceGeneral
import services.ServiceSavedModel as ServiceSavedModel


domainSize = (22.36, 22.36)
domainSize = (25, 25)
#domainSize = (50, 50)
noisePercentage = 1
noise = ServicePreparation.getNoiseAmplitudeValueForPercentage(noisePercentage)
#noise = 0
n = 100
speed = 1

threshold = 0.1
numberOfPreviousSteps = 100

# TODO: check why N-F always returns to order -> only with high radius

radius = 10
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

switchSummary = SwitchSummary([infoNsm])

# infoNsm2 = SwitchInformation(switchType=SwitchType.NEIGHBOUR_SELECTION_MECHANISM, 
#                             values=(NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE, NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE),
#                             thresholds=[threshold],
#                             numberPreviousStepsForThreshold=numberOfPreviousSteps
#                             )
# switchSummary2 = SwitchSummary([infoNsm2])

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

tmax = 5000

threshold = [0.1]

event = InternalStimulusOrientationChangeEvent(startTimestep=1000,
                                               duration=1000,  
                                               domainSize=domainSize, 
                                               eventEffect=InternalEventEffect.ALIGN_TO_FIXED_ANGLE, 
                                               percentage=1,
                                               angle=np.pi,
                                               noisePercentage=1,
                                               blockValues=True,
                                               alterValues=True,
                                               switchSummary=switchSummary
                                               )

tstart = time.time()

ServiceGeneral.logWithTime("start")

#initialState = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(None, domainSize, n, angleX=0.5, angleY=0.5)
simulator = VicsekWithNeighbourSelection(domainSize=domainSize,
                                         radius=radius,
                                         noise=noise,
                                         numberOfParticles=n,
                                         k=k,
                                         neighbourSelectionMechanism=nsm,
                                         speed=speed,
                                         switchSummary=switchSummary,
                                         events=[event])
#simulationData, switchTypeValues = simulator.simulate(initialState=initialState, tmax=tmax)
simulationData, switchTypeValues = simulator.simulate(tmax=tmax)

ServiceSavedModel.saveModel(simulationData=simulationData, path="test.json", 
                            modelParams=simulator.getParameterSummary())

tend = time.time()
ServiceGeneral.logWithTime(f"duration: {ServiceGeneral.formatTime(tend-tstart)}")