import numpy as np
import random, copy
from scipy.stats import linregress

import services.ServiceMetric as sm
import services.ServiceVicsekHelper as svh
import services.ServiceEvent as se
import services.ServiceThresholdEvaluation as ste
from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism


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


def computeContributionRateByTargetSwitchValue(positions, orientations, switchValues, targetSwitchValue, domainSize, radius, eventSelectionType, eventOriginPoint, numberOfAffected=None):
    individualContributions = computeIndividualContributions(positions=positions, 
                                                             orientations=orientations, 
                                                             switchValues=switchValues,
                                                             targetSwitchValue=targetSwitchValue,
                                                             domainSize=domainSize,
                                                             radius=radius,
                                                             eventSelectionType=eventSelectionType,
                                                             eventOriginPoint=eventOriginPoint,
                                                             numberOfAffected=numberOfAffected)
    

def computeIndividualContributions(positions, orientations, switchValues, targetSwitchValue, domainSize, radius, eventSelectionType, eventOriginPoint, numberOfAffected=None):
    """
    The switching decision is made based on the local order.
    Local order is computed on the basis of the orientations of all neighbours.
    Each neighbour's orientation contributes to the local order and thus to the decision of the individual.
    We can compute this contribution by projecting it by multiplying its orientation with the combined orientation from the local order.
    We can then compare the sum of the absolute values for each switch value. If the contribution of the target switch value is higher, then we can assume that the information has spread
    """    
    contributions = {}  

    influenced = 0
    noninfluenced = 0
    tgts = []
    for t in range(len(positions)):
        influenced_t = 0
        noninfluenced_t = 0
        #neighbours = svh.getNeighbours(positions=positions[t], domainSize=domainSize, radius=radius)
        neighbours = svh.getNeighboursWithLimitedVision(positions=positions[t], orientations=orientations[t], domainSize=domainSize,
                                                                            radius=radius, degreesOfVision=np.pi*2)
            
        affected = se.selectAffected(eventSelectionType=eventSelectionType,
                                     totalNumberOfParticles=len(positions[t]),
                                     positions=positions[t],
                                     originPoint=eventOriginPoint,
                                     domainSize=domainSize,
                                     radius=radius,
                                     numberOfAffected=numberOfAffected)
        
        orients = neighbours[:,:,np.newaxis]*orientations[np.newaxis,t,:]
        for i in range(1, len(orients)):
            #if t > 20 and t < 120:
            if switchValues[t-1][i] != targetSwitchValue.value and switchValues[t][i] == targetSwitchValue.value:
                contributions = projected_contributions(orients[i])
                tgt_mask = np.where(neighbours[i] & ((switchValues[t] == np.full(len(switchValues[t]), targetSwitchValue)) | affected), True, False)
                non_tgt_mask = np.where(neighbours[i] & np.invert(tgt_mask), True, False)
                tgt = np.sum(tgt_mask*contributions) / np.count_nonzero(contributions)
                non_tgt = np.sum(non_tgt_mask*contributions) / np.count_nonzero(contributions)
                
                tgts.append(tgt)
                if tgt > non_tgt:
                    influenced_t += 1
                else:
                    noninfluenced_t += 1
                total = tgt + non_tgt
                print(f"affected: {affected[i]}, switched: {switchValues[t-1][i] != targetSwitchValue and switchValues[t][i] == targetSwitchValue}, lo: {ste.computeLocalOrders(orientations[t], neighbours)[i]} tgt: {tgt/total}, nontgt: {non_tgt/total}")
        if influenced_t != 0 or noninfluenced_t != 0:
            print(f"{t}: infl: {influenced_t}, ninfl: {noninfluenced_t}")
            influenced += influenced_t
            noninfluenced += noninfluenced_t
    print(f"overall: infl: {influenced}, noninfl: {noninfluenced}, mintgt={np.min(tgts)}, avgtgt={np.average(tgts)}, maxtgt={np.max(tgts)}")

def projected_contributions(vectors):
    """Compute projected contribution of each agent."""
    V = np.sum(vectors, axis=0)
    V_hat = V / np.linalg.norm(V)
    return np.dot(vectors, V_hat)

def projected_contributions_agent_based(vectors, agent_idx):
    """Compute projected contribution of each agent."""
    V = np.sum(vectors, axis=0)
    V_hat = V / np.linalg.norm(vectors[agent_idx])
    return np.dot(vectors, V_hat)

def leave_one_out_contributions(vectors, neighbours):
    """Compute leave-one-out order parameter change for each agent."""
    N = len(vectors)
    full_order = sm.computeLocalOrder(vectors, neighbours)
    loo_contribs = []

    for i in range(N):
        if neighbours[i]:
            neighbours[i] = False
            loo_order = sm.computeLocalOrder(vectors, neighbours)
            neighbours[i] = True
            delta_phi = full_order - loo_order
            loo_contribs.append(delta_phi[0])
        else:
            loo_contribs.append(0)
    
    return np.array(loo_contribs)

