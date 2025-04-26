

from enums.EnumMetrics import TimeDependentMetrics
from enums.EnumSwitchType import SwitchType
from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from enums.EnumEventEffect import EventEffect   
from evaluators.EvaluatorMultiDependentInformation import EvaluatorMultiDependentInformation
import services.ServiceSavedModel as ssm
import services.ServiceGeneral as sg

metric = TimeDependentMetrics.TIME_TO_SWITCH
domainSize = (50, 50)
radius = 20
threshold = 0.11
use_agglo = True

location = "j:/duration_tests/"


for str in ["ordered", "random"]:
    for nsm in [NeighbourSelectionMechanism.NEAREST, 
                NeighbourSelectionMechanism.FARTHEST, 
                NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE,
                NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE]:
        disorderVal, orderVal = [1,5]
        if str == "ordered":
            startVal = orderVal
        else:
            startVal = disorderVal
        for eventEffect in [EventEffect.ALIGN_TO_FIXED_ANGLE,
                            EventEffect.AWAY_FROM_ORIGIN,
                            EventEffect.RANDOM]:
            if eventEffect == EventEffect.ALIGN_TO_FIXED_ANGLE:
                targetVal = orderVal
            else:
                targetVal = disorderVal
            base_path = f"local_1e_switchType=K_{str}_st={startVal}_o={orderVal}_do={disorderVal}_d=0.09_n=225_r=10_nsm={nsm.value}_noise=1_ee={eventEffect.value}_duration=1000"
            sg.logWithTime(base_path)
            filenames = sg.createListOfFilenamesForI(baseFilename=f"{location}{base_path}", minI=1, maxI=11, fileTypeString="csv")
            modelParams, simulationData, switchValues = ssm.loadModels(paths=filenames, switchTypes=[SwitchType.K], loadColours=False, loadSwitchValues=True, loadFromCsv=True)

            save_path = f"{base_path}.svg"
            positions = []
            orientations = []
            for i in range(len(simulationData)):
                times, pos, ori = simulationData[i]
                positions.append(pos)
                orientations.append(ori)

            positions = [positions]
            orientations = [orientations]

            evaluator = EvaluatorMultiDependentInformation(metric=metric,
                                                    positions=positions,
                                                    orientations=orientations,
                                                    domain_size=domainSize,
                                                    radius=radius,
                                                    threshold=threshold,
                                                    target_switch_value=targetVal,
                                                    switch_values=switchValues,
                                                    use_agglomerative_clustering=use_agglo)

            evaluator.evaluateAndVisualize(xLabel="durations", yLabel="number occurrences", savePath=save_path, show=False)

for str in ["ordered", "random"]:
    for nsmCombo in [[NeighbourSelectionMechanism.NEAREST, NeighbourSelectionMechanism.FARTHEST],
                     [NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE, NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE]]:
        disorderVal, orderVal = nsmCombo
        if str == "ordered":
            startVal = orderVal
        else:
            startVal = disorderVal
        for eventEffect in [EventEffect.ALIGN_TO_FIXED_ANGLE,
                            EventEffect.AWAY_FROM_ORIGIN,
                            EventEffect.RANDOM]:
            if eventEffect == EventEffect.ALIGN_TO_FIXED_ANGLE:
                targetVal = orderVal
            else:
                targetVal = disorderVal
            base_path = f"local_1e_switchType=MODE_{str}_st=NeighbourSelectionMode.{startVal.name}_o={orderVal.value}_do={disorderVal.value}_d=0.09_n=225_r=10_k=1_noise=1_ee={eventEffect.val}_duration=1000"
            sg.logWithTime(base_path)
            filenames = sg.createListOfFilenamesForI(baseFilename=f"{location}{base_path}", minI=1, maxI=11, fileTypeString="csv")
            modelParams, simulationData, switchValues = ssm.loadModels(paths=filenames, switchTypes=[SwitchType.K], loadColours=False, loadSwitchValues=True, loadFromCsv=True)

            save_path = f"{base_path}.svg"
            positions = []
            orientations = []
            for i in range(len(simulationData)):
                times, pos, ori = simulationData[i]
                positions.append(pos)
                orientations.append(ori)

            positions = [positions]
            orientations = [orientations]

            evaluator = EvaluatorMultiDependentInformation(metric=metric,
                                                    positions=positions,
                                                    orientations=orientations,
                                                    domain_size=domainSize,
                                                    radius=radius,
                                                    threshold=threshold,
                                                    target_switch_value=targetVal,
                                                    switch_values=switchValues,
                                                    use_agglomerative_clustering=use_agglo)

            evaluator.evaluateAndVisualize(xLabel="durations", yLabel="number occurrences", savePath=save_path, show=False)