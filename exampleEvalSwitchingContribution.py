import numpy as np

from enums.EnumSwitchType import SwitchType
from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from enums.EnumEventSelectionType import EventSelectionType
import services.ServiceNetwork as snw
import services.ServiceSavedModel as ssm


switchType = SwitchType.NEIGHBOUR_SELECTION_MECHANISM
target_switch_value = NeighbourSelectionMechanism.NEAREST
datafileLocation = ""
filename = "test_event_tmax=500_50_1.csv"
modelParams, simulationData, switchValues = ssm.loadModelFromCsv(f"{datafileLocation}{filename}.csv", f"{datafileLocation}{filename}_modelParams.csv", switchTypes=[switchType])

switchValues = switchValues[switchType.switchTypeValueKey]
times, positions, orientations = simulationData

domainSize = modelParams["domainSize"]
snw.computeContributionRateByTargetSwitchValue(positions=positions, orientations=orientations,
                                               switchValues=switchValues, targetSwitchValue=target_switch_value,
                                               domainSize=domainSize, radius=modelParams["radius"],
                                               eventSelectionType=EventSelectionType.RANDOM, eventOriginPoint=(domainSize[0]/2, domainSize[1]/2))