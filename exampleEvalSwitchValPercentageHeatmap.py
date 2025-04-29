from animator.AnimatorHeatmap import Animator
from enums.EnumSwitchType import SwitchType
import services.ServiceSavedModel as ssm
import services.ServiceSwitchAnalysis as ssa

#data = [0.1,0.2,0.3], [[[0.1,0.2],[0.1,0.3],[0.1,0.1]],[[0.2,0.2],[0.2,0.2],[0.2,0.2]],[[1,0.3],[0.3,0.3],[0.3,0.3]]]

switchType = SwitchType.K
target_switch_value = 1
datafileLocation = ""
filename = "test_info_ordered_predator_d=0.06_tmax=3000_1"
modelParams, simulationData, switchValues = ssm.loadModelFromCsv(f"{datafileLocation}{filename}.csv", f"{datafileLocation}{filename}_modelParams.csv", switchTypes=[switchType])

switchValues = switchValues[switchType.switchTypeValueKey]
times, grid = ssa.compute_infection_percentage_grid_values(simulation_data=simulationData, 
                                                           switch_values=switchValues,
                                                           target_switch_value=target_switch_value,
                                                           domain_size=modelParams["domainSize"])

animator = Animator()
animator.prepareAnimation()
animator.setSimulationData((times, grid))
animator.saveAnimation(f"heatmap_{filename}.mp4")