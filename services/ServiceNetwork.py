import numpy as np
import random
from scipy.stats import linregress
import matplotlib.pyplot as plt

import services.ServiceVicsekHelper as svh
import services.ServiceOrientations as so


def buildConnectionNetwork(positions, domainSize, radius):
    connections = []
    for t in range(len(positions)):
        connections.append(svh.getNeighbours(positions=positions[t], domainSize=domainSize, radius=radius))
    return connections

def buildContagionDictionaries(switchValues, targetSwitchValue, tStart, positions, domainSize, radius):
    connectionNetwork = buildConnectionNetwork(positions=positions, domainSize=domainSize, radius=radius)
    informedBy = {}
    originalInitiator = {}
    affectedAt = {}
    lags = {}
    fullySpread = False
    t = tStart
    while not fullySpread and t < len(switchValues):
        svt = np.array(switchValues[t])
        alreadyInformedOnce = np.full(len(svt), False)
        alreadyInformedOnce[list(lags.keys())] = True
        newlyInformed = svt != np.array(switchValues[t-1]) & (alreadyInformedOnce == False)
        newlyInformedIndices = np.nonzero(newlyInformed)[0]
        informed = (svt == targetSwitchValue) & (newlyInformed == False) 
        for i in newlyInformedIndices:
            affectedAt[i] = t
            informedNeighbours = connectionNetwork[t][i] & informed
            if np.count_nonzero(informedNeighbours) > 0:
                informer = random.choice(np.nonzero(informedNeighbours)[0])
                informedBy[i] = informer
                lags[i] = lags[informer] + 1
                if informer in originalInitiator:
                    originalInitiator[i] = originalInitiator[informer]
                else:
                    originalInitiator[i] = informer
            else:
                lags[i] = 1
        if np.count_nonzero(switchValues[t] == targetSwitchValue) == len(switchValues[t]):
            fullySpread = True
        t += 1

    return informedBy, originalInitiator, affectedAt, lags, fullySpread


def computeRanks(lags):
    maxLag = max(lags)
    ranks = {}
    for i in range(len(lags)):
        ranks[i] = maxLag - lags[i]
    return ranks

def measureInformationTransferSpeedViaInformationTransferDistance(switchValues, targetSwitchValue, eventStart, positions, domainSize, radius):
    """
    lag = time delay until the individual is informed
    rank = ranking by lag (short lag = low rank)
    The information transfer distance is the Euclidean distance between the individual and the initially 
    affected individual which has started the chain that caused the individual to change its mind.
    The information transfer speed is then the slope of the curve between the lag and the distance.

    If an individual has more than one informed neighbour, one is picked randomly
    """

    if eventStart == 0:
        tStart = eventStart + 1
    tStart = eventStart
    n = len(switchValues[0])
    informedBy, originalInitiator, affectedAt, lags, fullySpread = buildContagionDictionaries(switchValues=switchValues[0]['ks'], 
                                                                            targetSwitchValue=targetSwitchValue, 
                                                                            tStart=tStart, 
                                                                            positions=positions, 
                                                                            domainSize=domainSize, 
                                                                            radius=radius)
    #ranks = computeRanks(lags)
    distances = [np.linalg.norm(positions[affectedAt[i]][i], positions[tStart][originalInitiator[i]]) for i in range(n) if i in originalInitiator]

    result = linregress(lags.values(), distances)

    return result.slope, fullySpread

    
def measureInformationTransferViaOrientationsSpread(orientations, interval=1, startTimestep=0, endTimestep=None, savePath=None, show=False):
    """
    measure the spread of orientations and plot over time
    """
    if endTimestep == None:
        endTimestep = len(orientations)

    angles = so.computeAnglesForOrientations(orientations=orientations)
    x = []
    y = []
    for t in range(startTimestep, endTimestep, interval):
        angle = angles[t]
        x.extend([t for i in range(len(angle))])
        y.extend(angle)
    plt.scatter(x, y)
    plt.xlabel("time steps")
    plt.ylabel("orientation")
    if savePath:
        plt.savefig(savePath)
    if show:
        plt.show()


def measureInformationSpreadViaNumberOfNeighboursBasedProbability(switchValues, positions, domainSize, radius, eventStart, targetSwitchValue, savePath=None, show=False):
    """
    measure the number of neighbours with the target switch value and plot the probability for switching based on that
    """
    connections = buildConnectionNetwork(positions=positions, domainSize=domainSize, radius=radius)
    switches = {i: 0 for i in range(len(switchValues[0])+1)}
    for t in range(eventStart, len(switchValues)):
        svt = np.array(switchValues[t])
        newlyInformed = (svt != np.array(switchValues[t-1])) & (svt == targetSwitchValue)
        newlyInformedIndices = np.nonzero(newlyInformed)[0]
        for i in newlyInformedIndices:
            numAffectedNeighs = np.count_nonzero(connections[t][i] == targetSwitchValue)
            switches[numAffectedNeighs] += 1
    print(f"affected without affected neighbours: {switches[0]}")
    #del switches[0]
    plt.bar(switches.keys(), switches.values())
    plt.xlabel("number of neighbours with target value")
    plt.ylabel ("number of instances")
    if savePath:
        plt.savefig(savePath)
    if show:
        plt.show()


def measureInformationSpreadViaPercentageOfNeighboursBasedProbability(switchValues, positions, domainSize, radius, eventStart, targetSwitchValue, savePath=None, show=False):
    """
    measure the number of neighbours with the target switch value and plot the probability for switching based on that
    """
    connections = buildConnectionNetwork(positions=positions, domainSize=domainSize, radius=radius)
    switches = {i: 0 for i in range(101)}
    for t in range(eventStart, len(switchValues)):
        svt = np.array(switchValues[t])
        newlyInformed = (svt != np.array(switchValues[t-1])) & (svt == targetSwitchValue)
        newlyInformedIndices = np.nonzero(newlyInformed)[0]
        for i in newlyInformedIndices:
            numAffectedNeighs = np.count_nonzero(connections[t][i] == targetSwitchValue)
            percNeigh = int((numAffectedNeighs / len(connections[t][i])) * 100)
            switches[percNeigh] += 1
        
    print(f"affected without affected neighbours: {switches[0]}")
    #del switches[0]
    plt.bar(switches.keys(), switches.values())
    plt.xlabel("percentage of neighbours with target value")
    plt.ylabel("number of instances")
    if savePath:
        plt.savefig(savePath)
    if show:
        plt.show()