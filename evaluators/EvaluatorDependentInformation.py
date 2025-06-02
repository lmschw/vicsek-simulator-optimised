import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx

import evaluators.VisualizerDependentInformationEval as vde
import services.ServiceSavedModel as ssm
import services.ServiceClusters as scl
import services.ServicePowerlaw as spl
import services.ServiceSwitchAnalysis as ssa
import services.ServiceNetwork as snw
from enums.EnumMetrics import TimeDependentMetrics
from enums.EnumEventSelectionType import EventSelectionType

class EvaluatorDependentInformation:

    def __init__(self, metric, positions, orientations, domain_size, radius, threshold=0.01, use_agglomerative_clustering=True,
                 switch_values=[], target_switch_value=None, event_start=None, event_origin_point=(None,None), 
                 event_selection_type=EventSelectionType.RANDOM, number_of_affected=None, include_affected=True, contribution_threshold=0):
        self.metric = metric
        self.positions = positions
        self.orientations = orientations
        self.domain_size = domain_size
        self.radius = radius
        self.threshold = threshold
        self.use_agglomerative_clustering = use_agglomerative_clustering
        self.alpha = None
        self.switch_values = switch_values
        self.target_switch_value = target_switch_value
        self.event_start = event_start
        self.event_origin_point = event_origin_point
        self.event_selection_type = event_selection_type
        self.number_of_affected = number_of_affected
        self.include_affected = include_affected
        self.contribution_threshold = contribution_threshold

    def evaluateAndVisualize(self, xLabel=None, yLabel=None, subtitle=None, colourBackgroundForTimesteps=(None,None), xlim=None, ylim=None, savePath=None, show=False):
        """
        Evaluates and subsequently visualises the results for multiple models.

        Parameters:
            - labels (array of strings): the label for each model
            - xLabel (string) [optional]: the label for the x-axis
            - yLabel (string) [optional]: the label for the y-axis
            - subtitle (string) [optional]: subtitle to be included in the title of the visualisation
            - colourBackgroundForTimesteps ([start, stop]) [optional]: the start and stop timestep for the background colouring for the event duration
            - showVariance (boolean) [optional]: whether the variance data should be added to the plot, by default False
            - xlim (float) [optional]: the x-limit for the plot
            - ylim (float) [optional]: the y-limit for the plot
            - savePath (string) [optional]: the location and name of the file where the model should be saved. Will not be saved unless a savePath is provided

        Returns:
            Nothing.
        """
        data = self.evaluate()
        vde.visualize(metric=self.metric, data=data, xLabel=xLabel, yLabel=yLabel, subtitle=subtitle, colourBackgroundForTimesteps=colourBackgroundForTimesteps, xlim=xlim, ylim=ylim, savePath=savePath, show=show)
        
    def evaluate(self):
        match self.metric:
            case TimeDependentMetrics.CLUSTER_DURATION:
                data, _ = scl.compute_cluster_durations(positions=self.positions,
                                                     orientations=self.orientations,
                                                     domain_size=self.domain_size,
                                                     radius=self.radius,
                                                     threshold=self.threshold,
                                                     use_agglomerative_clustering=self.use_agglomerative_clustering)
                sorted_dict = dict(sorted(data.items()))
                self.alpha = spl.determinePowerlaw(list(sorted_dict.values()))
            case TimeDependentMetrics.CLUSTER_DURATION_PER_STARTING_TIMESTEP:
                _, data = scl.compute_cluster_durations(positions=self.positions,
                                                     orientations=self.orientations,
                                                     domain_size=self.domain_size,
                                                     radius=self.radius,
                                                     threshold=self.threshold,
                                                     use_agglomerative_clustering=self.use_agglomerative_clustering)
            case TimeDependentMetrics.CLUSTER_TREE:
                data = scl.compute_cluster_graph(positions=self.positions,
                                               orientations=self.orientations,
                                               domain_size=self.domain_size,
                                               radius=self.radius,
                                               threshold=self.threshold,
                                               use_agglomerative_clustering=self.use_agglomerative_clustering)   
            case TimeDependentMetrics.TIME_TO_SWITCH:
                _, data = ssa.compute_time_between_exposure_and_switch(positions=self.positions,
                                                                    switch_values=self.switch_values,
                                                                    target_switch_value=self.target_switch_value,
                                                                    event_start=self.event_start,
                                                                    event_origin_point=self.event_origin_point,
                                                                    event_radius=self.radius,
                                                                    domain_size=self.domain_size)
            case TimeDependentMetrics.DISTRIBUTION_NETWORK:
                data = snw.computeInformationSpreadNetworkBasedOnContributions(positions=self.positions,
                                                          orientations=self.orientations,
                                                          switchValues=self.switch_values,
                                                          targetSwitchValue=self.target_switch_value,
                                                          domainSize=self.domain_size,
                                                          radius=self.radius,
                                                          eventSelectionType=self.event_selection_type,
                                                          numberOfAffected=self.number_of_affected,
                                                          eventOriginPoint=self.event_origin_point,
                                                          includeAffected=self.include_affected,
                                                          threshold=self.contribution_threshold)    
            case TimeDependentMetrics.SWITCH_PROBABILITY_DISTRIBUTION:
                data = snw.computeInformationSpreadProbabilities(positions=self.positions,
                                                          orientations=self.orientations,
                                                          switchValues=self.switch_values,
                                                          targetSwitchValue=self.target_switch_value,
                                                          domainSize=self.domain_size,
                                                          radius=self.radius,
                                                          eventSelectionType=self.event_selection_type,
                                                          numberOfAffected=self.number_of_affected,
                                                          eventOriginPoint=self.event_origin_point,
                                                          includeAffected=self.include_affected,
                                                          threshold=self.contribution_threshold)    
        return data
                

