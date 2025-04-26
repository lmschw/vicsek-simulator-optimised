import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import pandas as pd

import evaluators.VisualizerDependentInformationEval as vde
from evaluators.EvaluatorDependentInformation import EvaluatorDependentInformation
from enums.EnumMetrics import Metrics

# matplotlib default colours with corresponding colours that are 65% and 50% lighter
COLOURS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
BACKGROUND_COLOURS_65_PERCENT_LIGHTER = ['#a6d1f0', '#ffd2ab', '#abe8ab', '#f1b3b3', '#dacae8',
                                         '#dbc1bc', '#f5cfea', '#d2d2d2', '#eff0aa', '#a7eef5']
BACKGROUND_COLOURS_50_PERCENT_LIGHTER = ['#7fbee9', '#ffbf86', '#87de87', '#eb9293', '#c9b3de',
                                         '#cca69f', '#f1bbe0', '#bfbfbf', '#e8e985', '#81e7f1']
BACKGROUND_COLOURS = BACKGROUND_COLOURS_50_PERCENT_LIGHTER
class EvaluatorMultiDependentInformation(object):
    """
    Implementation of the evaluation mechanism for the Vicsek model for comparison of multiple models.
    """

    def __init__(self, metric, positions, orientations, domain_size, radius, threshold=0.01, use_agglomerative_clustering=True,
                 switch_values=[], target_switch_value=None, event_start=None, event_origin_point=(None,None), evaluationTimestepInterval=1):
        self.metric = metric
        self.positions = positions
        self.orientations = orientations
        self.domain_size = domain_size
        self.radius = radius
        self.threshold = threshold
        self.use_agglomerative_clustering = use_agglomerative_clustering
        self.switch_values = switch_values
        self.target_switch_value = target_switch_value
        self.event_start = event_start
        self.event_origin_point = event_origin_point
        self.evaluationTimestepInterval = evaluationTimestepInterval

    def evaluate(self):
        """
        Evaluates all models according to the metric specified for the evaluator.

        Parameters:
            None

        Returns:
            A dictionary with the results for each model at every time step.
        """
        dd = defaultdict(list)
        varianceData = []
        for model in range(len(self.positions)):
            results = []
            varianceDataModel = []
            for individualRun in range(len(self.positions[model])):
                evaluator = EvaluatorDependentInformation(self.metric, self.positions[model][individualRun], self.orientations[model][individualRun], self.domain_size, self.radius, self.threshold, self.use_agglomerative_clustering, [], self.target_switch_value, self.event_start, self.event_origin_point)
                result = evaluator.evaluate()
                results.append(result)
            
            ddi = defaultdict(list)
            for d in results: 
                for key, value in d.items():
                    ddi[key].append(value)
            for k in range(len(ddi.keys())):
                if ddi[k] == []:
                    continue
                dd[k].append(np.average(ddi[k]))
                varianceDataModel.append(np.array(ddi[k]))
            varianceData.append(varianceDataModel)
        dd2 = {k: dd[k][0] for k in dd.keys()}
        return dd2, varianceData

    def evaluateAndVisualize(self, xLabel=None, yLabel=None, subtitle=None, colourBackgroundForTimesteps=[], xlim=None, ylim=None, savePath=None, show=False):
        data, varianceData = self.evaluate()
        vde.visualize(self.metric, data, xLabel=xLabel, yLabel=yLabel, subtitle=subtitle, colourBackgroundForTimesteps=colourBackgroundForTimesteps, varianceData=varianceData, xlim=xlim, ylim=ylim, savePath=savePath, show=show)
