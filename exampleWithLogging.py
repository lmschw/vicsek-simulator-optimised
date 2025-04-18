import time
import numpy as np

from model.VicsekIndividualsMultiSwitch import VicsekWithNeighbourSelection
from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from enums.EnumSwitchType import SwitchType
from events.ExternalStimulusEvent import ExternalStimulusOrientationChangeEvent
from enums.EnumEventEffect import EventEffect
from enums.EnumDistributionType import DistributionType
from enums.EnumEventSelectionType import EventSelectionType
from enums.EnumColourType import ColourType
from enums.EnumThresholdEvaluationMethod import ThresholdEvaluationMethod

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
print(f"noiseP={noisePercentage}, noise={noise}")
#noise = 0
n = ServicePreparation.getNumberOfParticlesForConstantDensity(0.05, domainSize)
print(n)
speed = 1

threshold = 0.1
numberOfPreviousSteps = 100

radius = 20
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

switchSummary = SwitchSummary([infoNsm, infoK])

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

threshold = [threshold]

event = ExternalStimulusOrientationChangeEvent(startTimestep=1000,
                                               duration=1000,  
                                               domainSize=domainSize, 
                                               eventEffect=EventEffect.AWAY_FROM_ORIGIN, 
                                               distributionType=DistributionType.LOCAL_SINGLE_SITE, 
                                               areas=[[domainSize[0]/2, domainSize[1]/2, radius]],
                                               angle=np.pi,
                                               noisePercentage=noise,
                                               radius=radius,
                                               numberOfAffected=None,
                                               eventSelectionType=EventSelectionType.RANDOM
                                               )

tstart = time.time()

ServiceGeneral.logWithTime("start")

for i in range(1, 2):
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
                                            events=[event],
                                            colourType=None,
                                            thresholdEvaluationMethod=ThresholdEvaluationMethod.LOCAL_ORDER,
                                            updateIfNoNeighbours=False,
                                            returnHistories=False,
                                            logPath=f'test_{i}.csv',
                                            logInterval=1000)
    simulationData, switchTypeValues = simulator.simulate(initialState=initialState, tmax=tmax)
    #simulationData, switchTypeValues, colours = simulator.simulate(tmax=tmax)
    times, positions, orientations = simulationData

    # ServiceSavedModel.saveModel(simulationData=simulationData, path=f"test_{i}.json", 
    #                             modelParams=simulator.getParameterSummary(), switchValues=switchTypeValues, colours=colours)

tend = time.time()
ServiceGeneral.logWithTime(f"duration: {ServiceGeneral.formatTime(tend-tstart)}")

