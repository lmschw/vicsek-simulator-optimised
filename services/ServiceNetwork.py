import numpy as np
import random
from scipy.stats import linregress

import services.ServiceVicsekHelper as svh


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


def computeContributionRateByTargetSwitchValue(positions, orientations, switchValues, targetSwitchValue, domainSize, radius):
    individualContributions = computeIndividualContributions(positions=positions, 
                                                             orientations=orientations, 
                                                             switchValues=switchValues,
                                                             targetSwitchValue=targetSwitchValue,
                                                             domainSize=domainSize,
                                                             radius=radius)
    

def computeIndividualContributions(positions, orientations, switchValues, targetSwitchValue, domainSize, radius):
    """
    The switching decision is made based on the local order.
    Local order is computed on the basis of the orientations of all neighbours.
    Each neighbour's orientation contributes to the local order and thus to the decision of the individual.
    We can compute this contribution by projecting it by multiplying its orientation with the combined orientation from the local order.
    We can then compare the sum of the absolute values for each switch value. If the contribution of the target switch value is higher, then we can assume that the information has spread
    """    
    contributions = {}       

    for t in range(len(positions)):
        neighbours = svh.getNeighbours(positions=positions[t], domainSize=domainSize, radius=radius)
        V = np.sum(neighbours[:,:,np.newaxis]*orientations[np.newaxis,:,:],axis=1)
        for i in range(len(V)):
            V_hat = V[i] / np.linalg.norm(V[i])
            contributions = np.dot(orientations[neighbours[i]], V_hat)
            print()

        

