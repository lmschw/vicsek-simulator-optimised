import services.ServicePreparation as ServicePreparation
import services.ServiceMetric as ServiceMetric
import services.ServiceVicsekHelper as ServiceVicsekHelper
import services.ServiceOrientations as ServiceOrientations
import services.ServiceSavedModel as ServiceSavedModel
import services.ServiceGeneral as ServiceGeneral

import evaluators.EvaluatorMultiComp as EvaluatorMultiComp

from enums.EnumMetrics import Metrics
from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism

import numpy as np
import time



#print(ServicePreparation.getNumberOfParticlesForConstantDensity(0.1, (100, 100)))
#print(ServicePreparation.getDensity((100, 100), 100))
#print(ServicePreparation.getDomainSizeForConstantDensity(0.01, 5))

switchValues = [5,1]
domainSize = (100, 100)
n = 10
radius = 20


"""

positions = domainSize*np.random.rand(n,len(domainSize))
orientations = ServiceOrientations.normalizeOrientations(np.random.rand(n, len(domainSize))-0.5)

orientations[1] = orientations[8]
orientations[3] = orientations[8]
orientations[7] = orientations[1]
orientations[0] = orientations[2]



initialState = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(switchValues[0], domainSize, n, angleX=0.5, angleY=0.5)
positions = initialState[0]
orientations = initialState[1]


print(ServiceMetric.findClustersWithRadius(positions, orientations, domainSize, radius, threshold=0.01))
"""

#modelParams, simulationData, switchTypeValues = ServiceSavedModel.loadModel("test.json", loadSwitchValues=True)
#time, positions, orientations = simulationData



"""
for t in [0, 1000, 2000, 3000]:
    print(ServiceMetric.findClustersWithRadius(positions[t], orientations[t], domainSize, radius, threshold=0.01))
"""

metric = Metrics.AVG_DISTANCE_NEIGHBOURS
labels = [""]
xAxisLabel = "timesteps"
yAxisLabel = metric.label
startEval = time.time()
modelParams = []
simulationData = []
switchTypeValues = []
modelParamsDensity, simulationDataDensity, siwtchTypeValuesDensity = ServiceSavedModel.loadModels(["test.json"], loadSwitchValues=True)
modelParams.append(modelParamsDensity)
simulationData.append(simulationDataDensity)
switchTypeValues.append(siwtchTypeValuesDensity)
threshold = 0.01
evaluator = EvaluatorMultiComp.EvaluatorMultiAvgComp(modelParams, metric, simulationData, evaluationTimestepInterval=1, threshold=threshold, switchTypeValues=switchTypeValues, switchTypeOptions=(NeighbourSelectionMechanism.FARTHEST, NeighbourSelectionMechanism.NEAREST))
savePath = f"{metric.val}_test_new_implementation.jpeg"
evaluator.evaluateAndVisualize(labels=labels, xLabel=xAxisLabel, yLabel=yAxisLabel, savePath=savePath)    
endEval = time.time()
print(f"Duration eval {ServiceGeneral.formatTime(endEval-startEval)}")