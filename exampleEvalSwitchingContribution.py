import numpy as np

from enums.EnumSwitchType import SwitchType
import services.ServiceNetwork as snw
import services.ServiceSavedModel as ssm


switchType = SwitchType.K
target_switch_value = 1
datafileLocation = ""
filename = "test_info_ordered_predator_d=0.06_tmax=3000_1"
modelParams, simulationData, switchValues = ssm.loadModelFromCsv(f"{datafileLocation}{filename}.csv", f"{datafileLocation}{filename}_modelParams.csv", switchTypes=[switchType])

switchValues = switchValues[switchType.switchTypeValueKey]
times, positions, orientations = simulationData

snw.computeContributionRateByTargetSwitchValue(positions=positions, orientations=orientations,
                                               switchValues=switchValues, targetSwitchValue=target_switch_value,
                                               domainSize=modelParams["domainSize"], radius=modelParams["radius"])