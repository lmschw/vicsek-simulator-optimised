import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx
from collections import OrderedDict

import services.ServiceSavedModel as ssm
import services.ServiceClusters as scl
import services.ServicePowerlaw as spl
import services.ServiceSwitchAnalysis as ssa
from enums.EnumMetrics import TimeDependentMetrics

COLOURS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
BACKGROUND_COLOURS_65_PERCENT_LIGHTER = ['#a6d1f0', '#ffd2ab', '#abe8ab', '#f1b3b3', '#dacae8',
                                        '#dbc1bc', '#f5cfea', '#d2d2d2', '#eff0aa', '#a7eef5']
BACKGROUND_COLOURS_50_PERCENT_LIGHTER = ['#7fbee9', '#ffbf86', '#87de87', '#eb9293', '#c9b3de',
                                        '#cca69f', '#f1bbe0', '#bfbfbf', '#e8e985', '#81e7f1']
BACKGROUND_COLOURS = BACKGROUND_COLOURS_50_PERCENT_LIGHTER

DOT_FACTOR = 10

def visualize(metric, data, xLabel=None, yLabel=None, subtitle=None, colourBackgroundForTimesteps=[], varianceData=None, xlim=None, ylim=None, savePath=None, show=False):
    match metric:
        case TimeDependentMetrics.CLUSTER_DURATION:
            visualize_bars(data=data, 
                                xLabel=xLabel, yLabel=yLabel, 
                                subtitle=subtitle, 
                                colourBackgroundForTimesteps=colourBackgroundForTimesteps, 
                                varianceData=varianceData, 
                                xlim=xlim, ylim=ylim,
                                savePath=savePath, 
                                show=show) 
        case TimeDependentMetrics.CLUSTER_DURATION_PER_STARTING_TIMESTEP:
            visualize_dots_durations(data=data, 
                                xLabel=xLabel, yLabel=yLabel, 
                                subtitle=subtitle, 
                                colourBackgroundForTimesteps=colourBackgroundForTimesteps, 
                                varianceData=varianceData, 
                                xlim=xlim, ylim=ylim,
                                savePath=savePath, 
                                show=show) 
        case TimeDependentMetrics.CLUSTER_TREE:
            visualize_tree(data=data, savePath=savePath, show=show)
        case TimeDependentMetrics.TIME_TO_SWITCH:
            visualize_dots_time_to_switch(data=data, 
                                xLabel=xLabel, yLabel=yLabel, 
                                subtitle=subtitle, 
                                colourBackgroundForTimesteps=colourBackgroundForTimesteps, 
                                varianceData=varianceData, 
                                xlim=xlim, ylim=ylim,
                                savePath=savePath, 
                                show=show) 
        case TimeDependentMetrics.DISTRIBUTION_NETWORK:
            visualize_network(data=data, savePath=savePath, show=show)
        case TimeDependentMetrics.SWITCH_PROBABILITY_DISTRIBUTION:
            visualize_lines(data=data, savePath=savePath, show=show)
        case TimeDependentMetrics.NETWORK_HOP_DISTANCE:
            visualize_bars(data=data, 
                                xLabel=xLabel, yLabel=yLabel, 
                                subtitle=subtitle, 
                                colourBackgroundForTimesteps=colourBackgroundForTimesteps, 
                                varianceData=varianceData, 
                                xlim=xlim, ylim=ylim,
                                savePath=savePath, 
                                show=show) 
        case TimeDependentMetrics.NETWORK_HOP_STRENGTH:
            print(data)
            visualize_bars(data=data, 
                                xLabel=xLabel, yLabel=yLabel, 
                                subtitle=subtitle, 
                                colourBackgroundForTimesteps=colourBackgroundForTimesteps, 
                                varianceData=varianceData, 
                                xlim=xlim, ylim=ylim,
                                savePath=savePath, 
                                show=show) 

def visualize_lines(data, xLabel=None, yLabel=None, subtitle=None, colourBackgroundForTimesteps=[], varianceData=None, xlim=None, ylim=None, alpha=None, savePath=None, show=False):
    plt.plot(data[1])
    plt.plot(data[2])
    plt.plot(data[3])
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
    elif alpha:
        plt.title(f"""{r'$\alpha$'} = {alpha}""")
    if len(colourBackgroundForTimesteps) > 0:
        ax = plt.gca()
        ylim = ax.get_ylim()
        y = np.arange(ylim[0], ylim[1], 0.01)
        ax.fill_betweenx(y, colourBackgroundForTimesteps[0], colourBackgroundForTimesteps[1], facecolor='green', alpha=0.2)
    if savePath != None:
        plt.savefig(savePath)
    if show:
        plt.show()
    plt.close()

def visualize_bars(data, xLabel=None, yLabel=None, subtitle=None, colourBackgroundForTimesteps=[], varianceData=None, xlim=None, ylim=None, alpha=None, savePath=None, show=False):
    print(data.keys())

    d = OrderedDict(sorted(data.items()))
    
    plt.bar(x=d.keys(), height=d.values())
    ax = plt.gca()
    # reset axis to start at (0.0)
    # xlim = ax.get_xlim()
    # ax.set_xlim((0, xlim[1]))
    # ylim = ax.get_ylim()
    # ax.set_ylim((0, ylim[1]))

    if xLabel != None:
        plt.xlabel(xLabel)
    if yLabel != None:
        plt.ylabel(yLabel)
    if subtitle != None:
        plt.title(f"""{subtitle}""")
    elif alpha:
        plt.title(f"""{r'$\alpha$'} = {alpha}""")
    if len(colourBackgroundForTimesteps) > 0:
        ax = plt.gca()
        ylim = ax.get_ylim()
        y = np.arange(ylim[0], ylim[1], 0.01)
        ax.fill_betweenx(y, colourBackgroundForTimesteps[0], colourBackgroundForTimesteps[1], facecolor='green', alpha=0.2)
    if savePath != None:
        plt.savefig(savePath)
    if show:
        plt.show()
    plt.close()

def visualize_dots_durations(data, xLabel=None, yLabel=None, subtitle=None, colourBackgroundForTimesteps=[], varianceData=None, xlim=None, ylim=None, savePath=None, show=False):
    x = []
    y = []
    s = []
    for k in data.keys():
        for d in data[k].keys():
            x.append(k)
            y.append(d)
            s.append(DOT_FACTOR * data[k][d])
    visualize_dots(x=x,
                    y=y,
                    s=s,
                    xLabel=xLabel,
                    yLabel=yLabel,
                    subtitle=subtitle,
                    colourBackgroundForTimesteps=colourBackgroundForTimesteps,
                    varianceData=varianceData,
                    xlim=xlim,
                    ylim=ylim,
                    savePath=savePath,
                    show=show)
    

def visualize_dots_time_to_switch(data, xLabel=None, yLabel=None, subtitle=None, colourBackgroundForTimesteps=[], varianceData=None, xlim=None, ylim=None, savePath=None, show=False):
    x = []
    y = []
    s = []
    for k in data.keys():
        x.append(k)
        y.append(data[k])
        s.append(DOT_FACTOR)
    visualize_dots(x=x,
                    y=y,
                    s=s,
                    xLabel=xLabel,
                    yLabel=yLabel,
                    subtitle=subtitle,
                    colourBackgroundForTimesteps=colourBackgroundForTimesteps,
                    varianceData=varianceData,
                    xlim=xlim,
                    ylim=ylim,
                    savePath=savePath,
                    show=show)

def visualize_dots(x, y, s, xLabel=None, yLabel=None, subtitle=None, colourBackgroundForTimesteps=[], varianceData=None, xlim=None, ylim=None, alpha=None, savePath=None, show=False):
    plt.scatter(x, y, s)
    ax = plt.gca()

    if xLabel != None:
        plt.xlabel(xLabel)
    if yLabel != None:
        plt.ylabel(yLabel)
    if subtitle != None:
        plt.title(f"""{subtitle}""")
    elif alpha:
        plt.title(f"""{r'$\alpha$'} = {alpha}""")
    if len(colourBackgroundForTimesteps) > 0:
        ax = plt.gca()
        ylim = ax.get_ylim()
        y = np.arange(ylim[0], ylim[1], 0.01)
        ax.fill_betweenx(y, colourBackgroundForTimesteps[0], colourBackgroundForTimesteps[1], facecolor='green', alpha=0.2)
    if savePath != None:
        plt.savefig(savePath)
    if show:
        plt.show()
    plt.close()

def visualize_tree(data, savePath, show=False):
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


def visualize_network(data, savePath, show=False):
    G, edge_labels = data
    print("Starting visualisation...")

    pos = nx.spring_layout(G)
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