import events.WallBehaviour as wb
from enums.EnumWallInfluenceType import WallInfluenceType


import time
import numpy as np

from model.VicsekIndividualsWallEvents import VicsekIndividualsWallEvents
from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from enums.EnumSwitchType import SwitchType
from events.ExternalStimulusEvent import ExternalStimulusOrientationChangeEvent
from events.ExternalStimulusWallEvent import ExternalEventStimulusWallEvent
from enums.EnumEventEffect import EventEffect
from enums.EnumDistributionType import DistributionType
from enums.EnumEventSelectionType import EventSelectionType
from enums.EnumWallInfluenceType import WallInfluenceType

from model.SwitchInformation import SwitchInformation
from model.SwitchSummary import SwitchSummary

import services.ServicePreparation as ServicePreparation
import services.ServiceGeneral as ServiceGeneral
import services.ServiceSavedModel as ServiceSavedModel
import services.ServiceOrientations as ServiceOrientations


circle = wb.WallTypeCircle(name="circle", wallInfluenceType=WallInfluenceType.CLOSE_TO_BORDER, influenceDistance=50, focusPoint=[0,0], radius=100)
""" 
position = [1, -98]
position2 = [29, 17]
position3 = [-1, 20]
positions = np.array([position, position2, position3])
orientation = ServiceOrientations.computeUvCoordinates(3*np.pi/2)
orientation2 = ServiceOrientations.computeUvCoordinates(2*np.pi/2)
orientation3 = ServiceOrientations.computeUvCoordinates(1*np.pi/2)
orientations = np.array([orientation, orientation2, orientation3])
speeds = np.array([1, 2, 3])
print(ServiceOrientations.computeAnglesForOrientations(orientations))
avoidances = circle.getAvoidanceOrientation(positions, orientations, speeds, dt=1, turnBy=0.314)
print(avoidances)
print(ServiceOrientations.computeAnglesForOrientations(avoidances))
"""

domainSize = (22.36, 22.36)
domainSize = (25, 25)
#domainSize = (50, 50)
noisePercentage = 1
noise = ServicePreparation.getNoiseAmplitudeValueForPercentage(noisePercentage)
#noise = 0
n = 10
speed = 1

threshold = 0.1
numberOfPreviousSteps = 100

# TODO: check why N-F always returns to order -> only with high radius

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

switchSummary = SwitchSummary([])
switchSummary = None

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
                                               numberOfAffected=1,
                                               eventSelectionType=EventSelectionType.RANDOM
                                               )

wallEvent = ExternalEventStimulusWallEvent(startTimestep=1000,
                                           duration=1000,
                                           wallTypeBehaviour=circle,
                                           domainSize=domainSize,
                                           noisePercentage=noisePercentage,
                                           turnBy=0.314)

tmax = 5000

tstart = time.time()

ServiceGeneral.logWithTime("start")

initialState = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(None, domainSize, n, angleX=0.5, angleY=0.5)
simulator = VicsekIndividualsWallEvents(domainSize=domainSize,
                                         radius=radius,
                                         noise=noise,
                                         numberOfParticles=n,
                                         k=k,
                                         neighbourSelectionMechanism=nsm,
                                         speed=speed,
                                         switchSummary=switchSummary,
                                         degreesOfVision=np.pi*2,
                                         events=[],
                                         wallEvents=[wallEvent])
simulationData, switchTypeValues = simulator.simulate(initialState=initialState, tmax=tmax)
#simulationData, switchTypeValues = simulator.simulate(tmax=tmax)

ServiceSavedModel.saveModel(simulationData=simulationData, path="test.json", 
                            modelParams=simulator.getParameterSummary())

tend = time.time()
ServiceGeneral.logWithTime(f"duration: {ServiceGeneral.formatTime(tend-tstart)}")