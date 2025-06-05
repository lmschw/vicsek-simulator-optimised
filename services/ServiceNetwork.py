import numpy as np
import random, copy
from scipy.stats import linregress
import networkx as nx

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

def computeInformationSpreadNetworkBasedOnContributions(positions, orientations, switchValues, targetSwitchValue, domainSize, radius, eventSelectionType, eventOriginPoint, numberOfAffected=None, includeAffected=True, threshold=0):
    """
    Computes the network based on contributions. 

    The switching decision is made based on the local order.
    Local order is computed on the basis of the orientations of all neighbours.
    Each neighbour's orientation contributes to the local order and thus to the decision of the individual.
    We can compute this contribution by projecting it by multiplying its orientation with the combined orientation from the local order.
    We can then compare the sum of the absolute values for each switch value. If the contribution of the target switch value is higher, then we can assume that the information has spread
    """    
    G = nx.DiGraph()
    edge_labels = {}
    contributions_dict = {}
    affected_timesteps = {i: [] for i in range(len(positions[0]))}
    switches = {i: [] for i in range(len(positions[0]))}
    for t in range(len(positions)):
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

        for i in range(len(orients)):
            if affected[i]:
                affected_timesteps[i].append(t)
            contributions = projected_contributions(orients[i])
            if switchValues[t-1][i] != targetSwitchValue.value and switchValues[t][i] == targetSwitchValue.value:
                switches[i].append(t)
                target_mask = np.where(neighbours[i] & ((switchValues[t] == np.full(len(switchValues[t]), targetSwitchValue)) | (affected * np.full(affected.shape, includeAffected))), True, False)
                G.add_nodes_from([f"{i}"])
                for j in range(len(target_mask)):
                    if target_mask[j] and np.absolute(contributions[j]) > threshold:
                        G.add_edge(f"{j}", f"{i}")
                        edge_labels[(f"{j}", f"{i}")] = t
                        if (f"{t}", f"{i}") not in contributions_dict.keys():
                            contributions_dict[(f"{t}", f"{i}")] = []
                        contributions_dict[(f"{t}", f"{i}")].append((j, contributions[j]))

    return (G, edge_labels), contributions_dict, affected_timesteps, switches


def findPathLengthsAndStrengths(contributions, affected, i, t):
    lengths = []
    strengths = []
    if t in affected[i] or t < 0:
        return [0], [1]
    
    nt = t-1
    
    if not (f"{t}", f"{i}") in contributions.keys():
        return findPathLengthsAndStrengths(contributions, affected, i, nt)
    
    feeders = contributions[(f"{t}", f"{i}")]
    for feeder in feeders:
        j, cont = feeder
        l, s =  findPathLengthsAndStrengths(contributions, affected, j, nt)
        for ln in l:
            lengths.append(ln + 1)
        for st in s:
            strengths.append(np.absolute(st * cont))
    return lengths, strengths

def computeInformationHopDistanceAndStrength(positions, orientations, switchValues, targetSwitchValue, domainSize, radius, 
                                             eventSelectionType, eventOriginPoint, numberOfAffected=None, includeAffected=True, 
                                             threshold=0, return_max_lengths=True, return_all=False, strength_decimals=1):
    _, contributions, affected, switches = computeInformationSpreadNetworkBasedOnContributions(positions=positions,
                                                                                                  orientations=orientations,
                                                                                                  switchValues=switchValues,
                                                                                                  targetSwitchValue=targetSwitchValue,
                                                                                                  domainSize=domainSize,
                                                                                                  radius=radius,
                                                                                                  eventSelectionType=eventSelectionType,
                                                                                                  eventOriginPoint=eventOriginPoint,
                                                                                                  numberOfAffected=numberOfAffected,
                                                                                                  includeAffected=includeAffected,
                                                                                                  threshold=threshold)
    hop_durations = {0: 0}
    hop_path_strengths = {0: 0}
    for i, switch_times in switches.items():
        for t in switch_times:
            if t in affected[i]:
                hop_durations[0] += 1
                hop_path_strengths[0] += 1
            else:
                lengths, strengths = findPathLengthsAndStrengths(contributions, affected, i, t)
                if return_all:
                    for l in lengths:
                        if l in hop_durations.keys():
                            hop_durations[l] += 1
                        else:
                            hop_durations[l] = 1
                    for st in strengths:
                        s = np.round(st, strength_decimals)
                        if s in hop_path_strengths.keys():
                            hop_path_strengths[s] += 1
                        else:
                            hop_path_strengths[s] = 1
                else: 
                    if return_max_lengths:
                        idx = np.argmax(lengths)
                    else:
                        idx = np.argmax(strengths)
                    if lengths[idx] in hop_durations.keys():
                        hop_durations[lengths[idx]] += 1
                    else:
                        hop_durations[lengths[idx]] = 1
                    s = np.round(strengths[idx], strength_decimals)
                    if s in hop_path_strengths.keys():
                        hop_path_strengths[s] += 1
                    else:
                        hop_path_strengths[s] = 1

    return hop_durations, hop_path_strengths

def computeInformationSpreadProbabilities(positions, orientations, switchValues, targetSwitchValue, domainSize, radius, eventSelectionType, eventOriginPoint, numberOfAffected=None, includeAffected=True, threshold=0):
    """
    Computes and returns the switching probability for every individual at every timestep as well as the min, average and max probability 
    and the standard deviation to switch for target value.

    The switching decision is made based on the local order.
    Local order is computed on the basis of the orientations of all neighbours.
    Each neighbour's orientation contributes to the local order and thus to the decision of the individual.
    We can compute this contribution by projecting it by multiplying its orientation with the combined orientation from the local order.
    We can then compare the sum of the absolute values for each switch value. If the contribution of the target switch value is higher, then we can assume that the information has spread
    """    
    switch_probabilities = []
    switch_probabilities_switch = []
    switch_probabilities_nonswitch = []
    influenced = 0
    noninfluenced = 0
    tgts = []
    target_ratios = []
    target_switch_ratios = []
    target_nonswitch_ratios = []
    target_counts = []
    target_switched_counts = []
    target_nonswitch_counts = []
    probs = []
    probs_switched = []
    probs_nonswitched = []
    los = []
    for t in range(len(positions)):
        probabilities = []
        probabilities_switch = []
        probabilities_nonswitch = []
        influenced_t = 0
        noninfluenced_t = 0
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

        localOrders = ste.computeLocalOrders(orientations[t], neighbours)

        for i in range(len(orients)):
            contributions = projected_contributions(orients[i])
            target_mask = np.where(neighbours[i] & ((switchValues[t] == np.full(len(switchValues[t]), targetSwitchValue)) | (affected * np.full(affected.shape, includeAffected))), True, False)
            non_target_mask = np.where(neighbours[i] & np.invert(target_mask), True, False)
            target_contribution = np.sum(np.absolute(target_mask*contributions)) / np.count_nonzero(contributions)
            non_target_contribution = np.sum(np.absolute(non_target_mask*contributions)) / np.count_nonzero(contributions)
            target_ratios.append(np.absolute(target_contribution)/(np.absolute(target_contribution) + np.absolute(non_target_contribution)))
            target_counts.append(np.count_nonzero(target_mask)/len(target_mask))
            local_order = localOrders[i]
            contribution_localorder_ratio = target_contribution/local_order
            absolute_contribution_ratio = np.absolute(target_contribution)/(np.absolute(target_contribution) + np.absolute(non_target_contribution))
            target_contributors_count_ratio = np.count_nonzero(target_mask)/len(target_mask)
            target_adoption_probability = 2*(absolute_contribution_ratio*target_contributors_count_ratio)
            if target_adoption_probability > 1:
                target_adoption_probability = 1
            probabilities.append(target_adoption_probability)
            print(f"{i} through all: lo:{local_order}, tgt:{target_contribution}, c:{contribution_localorder_ratio}, d={absolute_contribution_ratio}, cr:{target_contributors_count_ratio}, ccr={contribution_localorder_ratio*target_contributors_count_ratio}, dcr={absolute_contribution_ratio**target_contributors_count_ratio}")
            probs.append(target_adoption_probability)
            if switchValues[t-1][i] != targetSwitchValue.value and switchValues[t][i] == targetSwitchValue.value:
                probabilities_switch.append(target_adoption_probability)
                target_switch_ratios.append(np.absolute(target_contribution)/(np.absolute(target_contribution) + np.absolute(non_target_contribution)))
                target_switched_counts.append(np.count_nonzero(target_mask)/len(target_mask))
                probs_switched.append(target_adoption_probability)
                for j in range(len(target_mask)):
                    if target_mask[j] and np.absolute(contributions[j]) > threshold:
                        print(f"{i} through {j}: c={contributions[j]}, cr={np.count_nonzero(target_mask)/len(target_mask)}, ccr={contributions[j]*(np.count_nonzero(target_mask)/len(target_mask))}")
                    elif np.absolute(contributions[j]) < threshold:
                        print(t, j, contributions[j])
                tgts.append(target_contribution)
                if target_contribution > non_target_contribution:
                    influenced_t += 1
                else:
                    noninfluenced_t += 1
                total = target_contribution + non_target_contribution
                print(f"affected: {affected[i]}, switched: {switchValues[t-1][i] != targetSwitchValue and switchValues[t][i] == targetSwitchValue}, lo: {ste.computeLocalOrders(orientations[t], neighbours)[i]} tgt: {target_contribution/total}, nontgt: {non_target_contribution/total}")
            elif switchValues[t-1][i] == targetSwitchValue.value and switchValues[t][i] != targetSwitchValue.value:
                target_nonswitch_ratios.append(np.absolute(target_contribution)/(np.absolute(target_contribution) + np.absolute(non_target_contribution)))
                target_nonswitch_counts.append(np.count_nonzero(target_mask)/len(target_mask))
                probs_nonswitched.append(target_adoption_probability)
                probabilities_nonswitch.append(target_adoption_probability)
            else:
                probabilities_nonswitch.append(target_adoption_probability)
        if influenced_t != 0 or noninfluenced_t != 0:
            print(f"{t}: infl: {influenced_t}, ninfl: {noninfluenced_t}")
            influenced += influenced_t
            noninfluenced += noninfluenced_t
        switch_probabilities.append(np.average(probabilities))
        if len(probabilities_switch) > 0:
            switch_probabilities_switch.append(np.average(probabilities_switch))
        else:
            switch_probabilities_switch.append(0)
        if len(probabilities_nonswitch):
            switch_probabilities_nonswitch.append(np.average(probabilities_nonswitch))
        else:
            switch_probabilities_nonswitch.append(0)
        los.append(np.average(localOrders))
    print(f"overall: infl: {influenced}, noninfl: {noninfluenced}, mintgt={np.min(tgts)}, avgtgt={np.average(tgts)}, maxtgt={np.max(tgts)}")
    print(f"tgt ratio: min: {np.min(target_ratios)}, avg:{np.average(target_ratios)}, max: {np.max(target_ratios)}, std: {np.std(target_ratios)}")
    print(f"tgt switched ratio: min: {np.min(target_switch_ratios)}, avg:{np.average(target_switch_ratios)}, max: {np.max(target_switch_ratios)}, std: {np.std(target_switch_ratios)}")
    print(f"tgt no switch ratio: min: {np.min(target_nonswitch_ratios)}, avg:{np.average(target_nonswitch_ratios)}, max: {np.max(target_nonswitch_ratios)}, std: {np.std(target_nonswitch_ratios)}")
    print(f"tgt count ratio: min: {np.min(target_counts)}, avg:{np.average(target_counts)}, max: {np.max(target_counts)}, std: {np.std(target_counts)}")
    print(f"tgt switched count ratio: min: {np.min(target_switched_counts)}, avg:{np.average(target_switched_counts)}, max: {np.max(target_switched_counts)}, std: {np.std(target_switched_counts)}")
    print(f"tgt nonswitch count ratio: min: {np.min(target_nonswitch_counts)}, avg:{np.average(target_nonswitch_counts)}, max: {np.max(target_nonswitch_counts)}, std: {np.std(target_nonswitch_counts)}")
    print(f"p ratio: min: {np.min(probs)}, avg:{np.average(probs)}, max: {np.max(probs)}, std: {np.std(probs)}")
    print(f"psw ratio: min: {np.min(probs_switched)}, avg:{np.average(probs_switched)}, max: {np.max(probs_switched)}, std: {np.std(probs_switched)}")
    print(f"pnsw ratio: min: {np.min(probs_nonswitched)}, avg:{np.average(probs_nonswitched)}, max: {np.max(probs_nonswitched)}, std: {np.std(probs_nonswitched)}")

    print(f"switch prob: min={np.min(switch_probabilities)},avg={np.average(switch_probabilities)},max={np.max(switch_probabilities)}") 
    print(f"switch prob switched: min={np.min(switch_probabilities_switch)},avg={np.average(switch_probabilities_switch)},max={np.max(switch_probabilities_switch)}")
    print(f"switch prob switched: min={np.min(switch_probabilities_nonswitch)},avg={np.average(switch_probabilities_nonswitch)},max={np.max(switch_probabilities_nonswitch)}")

    return (switch_probabilities, switch_probabilities_switch, switch_probabilities_nonswitch, los)

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

