import numpy as np

from enums.EnumSwitchType import SwitchType
from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from enums.EnumEventSelectionType import EventSelectionType
from enums.EnumMetrics import TimeDependentMetrics
import services.ServiceNetwork as snw
import services.ServiceSavedModel as ssm
from evaluators.EvaluatorDependentInformation import EvaluatorDependentInformation


switchType = SwitchType.NEIGHBOUR_SELECTION_MECHANISM
target_switch_value = NeighbourSelectionMechanism.NEAREST
datafileLocation = ""
filename = "test_event_tmax=1000_50_1.json"
#modelParams, simulationData, switchValues = ssm.loadModelFromCsv(f"{datafileLocation}{filename}.csv", f"{datafileLocation}{filename}_modelParams.csv", switchTypes=[switchType])
modelParams, simulationData, switchValues = ssm.loadModel(filename, switchTypes=[switchType], loadSwitchValues=True)

switchValues = switchValues[switchType.switchTypeValueKey]
times, positions, orientations = simulationData

domainSize = modelParams["domainSize"]
radius = modelParams["radius"]

"""
snw.computeContributionRateByTargetSwitchValue(positions=positions, orientations=orientations,
                                               switchValues=switchValues, targetSwitchValue=target_switch_value,
                                               domainSize=domainSize, radius=modelParams["radius"],
                                               eventSelectionType=EventSelectionType.RANDOM, eventOriginPoint=(domainSize[0]/2, domainSize[1]/2))
"""

eval = EvaluatorDependentInformation(metric=TimeDependentMetrics.NETWORK_HOP_DISTANCE,
                                     positions=positions,
                                     orientations=orientations,
                                     domain_size=domainSize,
                                     radius=radius,
                                     switch_values=switchValues,
                                     target_switch_value=target_switch_value,
                                     event_origin_point=(domainSize[0]/2, domainSize[1]/2),
                                     include_affected=True,
                                     contribution_threshold=0)

eval.evaluateAndVisualize(show=True)