

from enums.EnumMetrics import TimeDependentMetrics
from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from enums.EnumSwitchType import SwitchType
from evaluators.EvaluatorDependentInformation import EvaluatorDependentInformation
import services.ServiceSavedModel as ssm

metric = TimeDependentMetrics.TIME_TO_SWITCH
domainSize = (25, 25)
radius = 5
event_center = (12.5, 12.5)
event_start = 1000
target_switch_value = NeighbourSelectionMechanism.NEAREST
save_path = "test_timetoswitch.svg"

modelParams, simulationData, switch_values = ssm.loadModel(path="test_event_1.json", loadSwitchValues=True, switchTypes=[SwitchType.NEIGHBOUR_SELECTION_MECHANISM])
times, positions, orientations = simulationData

switch_values = switch_values[SwitchType.NEIGHBOUR_SELECTION_MECHANISM.switchTypeValueKey]
evaluator = EvaluatorDependentInformation(metric=metric,
                                          positions=positions,
                                          orientations=orientations,
                                          domain_size=domainSize,
                                          radius=radius,
                                          event_origin_point=event_center,
                                          event_start=event_start,
                                          switch_values=switch_values,
                                          target_switch_value=target_switch_value)

evaluator.evaluateAndVisualize(xLabel="start timestep", yLabel="duration", savePath=save_path, show=False)