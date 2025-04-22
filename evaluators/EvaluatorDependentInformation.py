import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx

import services.ServiceSavedModel as ssm
import services.ServiceClusters as scl
import services.ServicePowerlaw as spl
from enums.EnumMetrics import TimeDependentMetrics


COLOURS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
BACKGROUND_COLOURS_65_PERCENT_LIGHTER = ['#a6d1f0', '#ffd2ab', '#abe8ab', '#f1b3b3', '#dacae8',
                                        '#dbc1bc', '#f5cfea', '#d2d2d2', '#eff0aa', '#a7eef5']
BACKGROUND_COLOURS_50_PERCENT_LIGHTER = ['#7fbee9', '#ffbf86', '#87de87', '#eb9293', '#c9b3de',
                                        '#cca69f', '#f1bbe0', '#bfbfbf', '#e8e985', '#81e7f1']
BACKGROUND_COLOURS = BACKGROUND_COLOURS_50_PERCENT_LIGHTER

class EvaluatorDependentInformation:

    def __init__(self, metric, positions, orientations, domain_size, radius, threshold=0.01, use_agglomerative_clustering=True):
        self.metric = metric
        self.positions = positions
        self.orientations = orientations
        self.domain_size = domain_size
        self.radius = radius
        self.threshold = threshold
        self.use_agglomerative_clustering = use_agglomerative_clustering
        self.alpha = None

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
        self.visualize(data, xLabel=xLabel, yLabel=yLabel, subtitle=subtitle, colourBackgroundForTimesteps=colourBackgroundForTimesteps, xlim=xlim, ylim=ylim, savePath=savePath, show=show)
        
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
        return data
                

    def visualize(self, data, xLabel=None, yLabel=None, subtitle=None, colourBackgroundForTimesteps=None, varianceData=None, xlim=None, ylim=None, savePath=None, show=False):
        match self.metric:
            case TimeDependentMetrics.CLUSTER_DURATION:
                self.visualize_bars(data=data, 
                                    xLabel=xLabel, yLabel=yLabel, 
                                    subtitle=subtitle, 
                                    colourBackgroundForTimesteps=colourBackgroundForTimesteps, 
                                    varianceData=varianceData, 
                                    xlim=xlim, ylim=ylim,
                                    savePath=savePath, 
                                    show=show) 
            case TimeDependentMetrics.CLUSTER_DURATION_PER_STARTING_TIMESTEP:
                self.visualize_dots(data=data, 
                                    xLabel=xLabel, yLabel=yLabel, 
                                    subtitle=subtitle, 
                                    colourBackgroundForTimesteps=colourBackgroundForTimesteps, 
                                    varianceData=varianceData, 
                                    xlim=xlim, ylim=ylim,
                                    savePath=savePath, 
                                    show=show) 
            case TimeDependentMetrics.CLUSTER_TREE:
                self.visualize_tree(data=data, savePath=savePath, show=show)

    def visualize_bars(self, data, xLabel=None, yLabel=None, subtitle=None, colourBackgroundForTimesteps=None, varianceData=None, xlim=None, ylim=None, savePath=None, show=False):
        plt.bar(x=data.keys(), height=data.values())
        ax = plt.gca()
        # reset axis to start at (0.0)
        xlim = ax.get_xlim()
        ax.set_xlim((0, xlim[1]))
        ylim = ax.get_ylim()
        ax.set_ylim((0, ylim[1]))

        if xLabel != None:
            plt.xlabel(xLabel)
        if yLabel != None:
            plt.ylabel(yLabel)
        if subtitle != None:
            plt.title(f"""{subtitle}""")
        elif self.alpha:
            plt.title(f"""{r'$\alpha$'} = {self.alpha}""")
        if not any(ele is None for ele in colourBackgroundForTimesteps):
            ax = plt.gca()
            ylim = ax.get_ylim()
            y = np.arange(ylim[0], ylim[1], 0.01)
            ax.fill_betweenx(y, colourBackgroundForTimesteps[0], colourBackgroundForTimesteps[1], facecolor='green', alpha=0.2)
        if savePath != None:
            plt.savefig(savePath)
        if show:
            plt.show()
        plt.close()

    def visualize_dots(self, data, xLabel=None, yLabel=None, subtitle=None, colourBackgroundForTimesteps=None, varianceData=None, xlim=None, ylim=None, savePath=None, show=False):
        x = []
        y = []
        s = []
        for k in data.keys():
            for d in data[k].keys():
                x.append(k)
                y.append(d)
                s.append(data[k][d])
        plt.scatter(x, y, s)
        ax = plt.gca()
        # reset axis to start at (0.0)
        xlim = ax.get_xlim()
        ax.set_xlim((0, xlim[1]))
        ylim = ax.get_ylim()
        ax.set_ylim((0, ylim[1]))

        if xLabel != None:
            plt.xlabel(xLabel)
        if yLabel != None:
            plt.ylabel(yLabel)
        if subtitle != None:
            plt.title(f"""{subtitle}""")
        elif self.alpha:
            plt.title(f"""{r'$\alpha$'} = {self.alpha}""")
        if not any(ele is None for ele in colourBackgroundForTimesteps):
            ax = plt.gca()
            ylim = ax.get_ylim()
            y = np.arange(ylim[0], ylim[1], 0.01)
            ax.fill_betweenx(y, colourBackgroundForTimesteps[0], colourBackgroundForTimesteps[1], facecolor='green', alpha=0.2)
        if savePath != None:
            plt.savefig(savePath)
        if show:
            plt.show()
        plt.close()

    def visualize_tree(self, data, savePath, show=False):
        G, edge_labels = data
        print("Starting visualisation...")
        # For visualization purposes, layout the nodes in topological order
        for i, layer in enumerate(nx.topological_generations(G)):
            for n in layer:
                G.nodes[n]["layer"] = i
        pos = nx.multipartite_layout(G, subset_key="layer", align="horizontal")
        # Flip the layout so the root node is on top
        for k in pos:
            pos[k][-1] *= -1

        # Visualize the trie
        # nx.draw_networkx_nodes(G, pos)
        # nx.draw_networkx_edges(G, pos, alpha=0.5, width=6)
        nx.draw(G, pos, with_labels=True)
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
        )
        # Customize axes
        ax = plt.gca()
        ax.margins(0.11)
        plt.tight_layout()
        plt.axis("off")
        if savePath != None:
            plt.savefig(savePath)
        if show:
            plt.show()