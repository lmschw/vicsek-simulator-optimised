

from enums.EnumMetrics import TimeDependentMetrics
from evaluators.EvaluatorDependentInformation import EvaluatorDependentInformation
import services.ServiceSavedModel as ssm

metric = TimeDependentMetrics.CLUSTER_DURATION
domainSize = (25, 25)
radius = 100
threshold = 0.11
use_agglo = True
save_path = "test_clusterduration.svg"

modelParams, simulationData = ssm.loadModel(path="test_clusters.json")
times, positions, orientations = simulationData

evaluator = EvaluatorDependentInformation(metric=metric,
                                          positions=positions,
                                          orientations=orientations,
                                          domain_size=domainSize,
                                          radius=radius,
                                          threshold=threshold,
                                          use_agglomerative_clustering=use_agglo)

evaluator.evaluateAndVisualize(xLabel="durations", yLabel="number occurrences", savePath=save_path, show=True)