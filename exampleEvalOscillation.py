from evaluators.EvaluatorAvalanches import EvaluatorAvalanches
import services.ServiceSavedModel as ssm

filename = "test_stress_9_tmax=500000_1"
modelParams, simulationData, switchTypeValues = ssm.loadModel(f"{filename}.json", loadSwitchValues=True)
times, positions, orientations = simulationData

evaluator = EvaluatorAvalanches(orientations=orientations, orderThreshold=0.9, savePath=filename)
evaluator.evaluateAvalanches()