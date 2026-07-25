import time
import numpy as np

from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from enums.EnumMetrics import Metrics
from enums.EnumSwitchType import SwitchType
from enums.EnumDistributionType import DistributionType
from enums.EnumEventEffect import EventEffect

import evaluators.EvaluatorMultiComp as EvaluatorMultiComp
import services.ServiceSavedModel as ServiceSavedModel
import services.ServicePreparation as ServicePreparation
import services.ServiceGeneral as ServiceGeneral
import services.ServiceMetric as ServiceMetric

import DefaultValues as dv
import animator.AnimatorMatplotlib as AnimatorMatplotlib
import animator.Animator2D as Animator2D

path = "/home/lilly/dev/data_comp_neighbour_selection_metacontrol/data_comparison_neighbour_selection_meta_control/0066_000_fov.json"

metric = Metrics.ORDER

params, simulationData = ServiceSavedModel.loadModel(path=path)
times, positions, orientations = simulationData

tmax = len(times)
n = 10

xlim = (0, tmax)
threshold = 0.01
if metric in [Metrics.ORDER, Metrics.DUAL_OVERLAY_ORDER_AND_PERCENTAGE]:
    ylim = (0, 1.1)
elif metric == Metrics.CLUSTER_NUMBER_WITH_RADIUS:
    ylim = (0, n)
    threshold = 0.995
else:
    ylim = (0, 50)

yAxisLabel = metric.label
startEval = time.time()
modelParams = [[None]]
switchTypes = []
evalInterval = 1

#paths.append(f"density-vs-noise_ORDER_mode-comparision_n={n}_k=1_radius=10_density={density}_noise={noisePercentage}%_hierarchical_clustering_threshold=0.01.png")
#createMultiPlotFromImages(title, numX, numY, rowLabels, colLabels, paths)
threshold = 0.01
evaluator = EvaluatorMultiComp.EvaluatorMultiAvgComp(metric=metric, modelParams=modelParams, simulationData=[[simulationData]], evaluationTimestepInterval=evalInterval, threshold=threshold)

labels = ["actual"]
savePath = f"{metric.val}_{path[-10:]}.jpeg"

evaluator.evaluateAndVisualize(labels=labels, xLabel="timesteps", yLabel=yAxisLabel, showVariance=True, xlim=xlim, ylim=ylim, savePath=savePath)    
endEval = time.time()
print(f"Duration eval {ServiceGeneral.formatTime(endEval-startEval)}") 
