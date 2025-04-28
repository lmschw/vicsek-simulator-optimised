import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import pandas as pd

import evaluators.VisualizerDependentInformationEval as vde
import services.ServiceSavedModel as ssm
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

    def __init__(self, metric, base_paths, min_i, max_i, threshold=0.01, use_agglomerative_clustering=True,
                 switch_type=None, from_csv=False, target_switch_value=None, event_start=None, 
                 event_origin_point=(None,None), evaluationTimestepInterval=1):
        self.metric = metric
        self.base_paths = base_paths
        self.min_i = min_i
        self.max_i = max_i
        self.threshold = threshold
        self.use_agglomerative_clustering = use_agglomerative_clustering
        self.switch_type = switch_type
        self.from_csv = from_csv
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
        for base_path in self.base_paths:
            results = []
            varianceDataModel = []
            for individualRun in range(self.min_i, self.max_i):
                if self.from_csv:
                    path = f"{base_path}_{individualRun}.csv"
                    path_model_params = f"{base_path}_{individualRun}_modelParams.csv"
                    if self.switch_type:
                        model_params, simulation_data, switch_values = ssm.loadModelFromCsv(path, path_model_params, switchTypes=[self.switch_type])
                    else:
                        model_params, simulation_data = ssm.loadModelFromCsv(path, path_model_params)
                    switch_values = []
                else:
                    path = f"{base_path}_{individualRun}.json"
                    if self.switch_type:
                        model_params, simulation_data, switch_values = ssm.loadModel(path=path, switchTypes=[self.switch_type], loadSwitchValues=True)
                    else:
                        model_params, simulation_data = ssm.loadModel(path=path)
                        switch_values = []
                _, positions, orientations = simulation_data
                evaluator = EvaluatorDependentInformation(self.metric, positions, orientations, model_params["domainSize"], model_params["radius"], self.threshold, self.use_agglomerative_clustering, switch_values, self.target_switch_value, self.event_start, self.event_origin_point)
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
