import tensorflow as tf
import pandas as pd
import numpy as np
import random

from EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from EnumSwitchType import SwitchType

def generateNoise():
    return np.random.normal(scale=noise, size=(n, len(domainSize)))

def calculateMeanOrientations(orientations, neighbours):
    summedOrientations = np.sum(neighbours[:,:,np.newaxis]*orientations[np.newaxis,:,:],axis=1)
    return normalizeOrientations(summedOrientations)

def normalizeOrientations(orientations):
    return orientations/(np.sqrt(np.sum(orientations**2,axis=1))[:,np.newaxis])

def getPositionDifferences(positions):
    rij=positions[:,np.newaxis,:]-positions
    #rij=rij[~np.eye(rij.shape[0],dtype=bool),:].reshape(rij.shape[0],rij.shape[0]-1,-1) #remove i<>i interaction
        
    rij = rij - domainSize*np.rint(rij/domainSize) #minimum image convention

    rij2 = np.sum(rij**2,axis=2)
    return rij2

def getNeighbours(positions):
    rij2 = getPositionDifferences(positions)

    neighbours = (rij2 <= radius**2)
    np.fill_diagonal(neighbours, False)
    return neighbours

def pickPositionNeighbours(k, positions, neighbours, isMin=True):
    posDiff = getPositionDifferences(positions)
    if isMin == True:
        #neighbourDiffs = np.where(neighbours == True, posDiff, maxSq)
        fillValue = maxSq
    else:
        #neighbourDiffs = np.where(neighbours == True, posDiff, minSq)
        fillValue = minSq

    #a = np.sort(posDiff, axis=1)
    #mask = neighbours.nonzero()
    #a = np.argsort(neighbourDiffs, axis=1)
    
    minusOnes = np.full((n,k), -1)
    trues = np.full((n,n), True)
    falses = np.full((n,n), False)

    maskedArray = np.ma.MaskedArray(posDiff, mask=neighbours==False, fill_value=fillValue)
    sortedIndices = maskedArray.argsort(axis=1)
    if isMin == False:
        sortedIndices = np.flip(sortedIndices)
    candidates = sortedIndices[:, :k]
    
    pickedDistances = np.take_along_axis(posDiff, candidates, axis=1)
    # filter on actual neighbours, e.g. by replacing indices that aren't neighbours by the diagonal index (the agent's own index)
    #diagonalIndices = np.diag_indices(n)[0]
    picked = np.where(((pickedDistances == 0) | (pickedDistances > radius**2)), minusOnes, candidates)

    # TODO replace loop
    ns = np.full((n,n), False)
    for i in range(n):
        for j in range(n):
            if j in picked[i]:
                ns[i][j] = True

    #mask = np.zeros((n,n), dtype=np.bool_)
    #mask[picked] = True
    #mask[picked[:,0], picked[:,1]]=True
    #mask = []
    return ns

def computeNewOrientation(nsm, k, neighbours, positions, orientations, vals):

    """
    match switchType:
        case SwitchType.NEIGHBOUR_SELECTION_MODE:
            valsDf = pd.DataFrame(vals)
            valsDf["val"] = valsDf["val"].case_when([(, valsB),
                                (((valsDf["localOrder"] <= threshold) & (valsDf["previousLocalOrder"] >= threshold)), valsA),
            ])
    """
    match nsm:
        case NeighbourSelectionMechanism.NEAREST:
            pickedNeighbours = pickPositionNeighbours(k, positions, neighbours, maxSq)


    orientations = calculateMeanOrientations(orientations, pickedNeighbours)
    orientations = normalizeOrientations(orientations+generateNoise())
    return orientations

def getDistancesPosition(positions):
    print("placeholder")

def getDistancesOrientation(orientations):
    print("placeholder")

def getLocalOrders(orientations, neighbours):
    sumOrientation = np.sum(neighbours[:,:,np.newaxis]*orientations[np.newaxis,:,:],axis=1)
    localOrders = np.divide(np.sqrt(np.sum(sumOrientation**2,axis=1)), tf.math.count_nonzero(neighbours, axis=1))
    return localOrders
    

def getDecisions(t, previousSteps, localOrders, previousLocalOrders, vals, threshold=0.1):
    """
    Computes whether the individual chooses to use option A or option B as its value based on the local order, the average previous local order and a threshold.
    """
    prev = np.average(previousLocalOrders[max(t-previousSteps, 0):t+1], axis=0)
    valsDf = pd.DataFrame(vals)
    valsDf["localOrder"] = localOrders
    valsDf["previousLocalOrder"] = prev
    valsDf["val"] = valsDf["val"].case_when([(((valsDf["localOrder"] >= 1-threshold) & (valsDf["previousLocalOrder"] <= 1-threshold)), valsB),
                           (((valsDf["localOrder"] <= threshold) & (valsDf["previousLocalOrder"] >= threshold)), valsA),
    ])
    return pd.DataFrame(valsDf["val"])

    

noise = 0.063
density = 0.09
domainSize = (25, 25)
n = 56

maxSq = domainSize[0] * domainSize[1] + 1
minSq = -1

radius = 5
n = 5
k = 2
nsm = NeighbourSelectionMechanism.NEAREST

indices = []
for i in range(n):
    indices.append(tf.range(n))

valsA = tf.zeros([n])
valsB = tf.ones([n])


#indices = tf.constant(indices)

d1pos = [10, 10]
d1ori = [14, -14]
d2pos = [15, 15]
d2ori = [-0.5, -10]
d3pos = [20, 20]
d3ori = [0.2, 0.2]
d4pos = [5, 5]
d4ori = [-0.2, 20]
d5pos = [12, 12]
d5ori = [7, 4]

"""
d1ori = [1,1]
d2ori = [1,1]
d3ori = [1,1]

d1ori = [0.38252475, 0.37564835]
d2ori = [0.38252475, 0.37564835]
d3ori = [0.38252475, 0.37564835]
"""

vals = tf.constant([0, 1, 1, 0, 0])

vals = pd.DataFrame(vals, columns=['val'])

positions = tf.constant([d1pos, d2pos, d3pos, d4pos, d5pos])
orientations = tf.constant([d1ori, d2ori, d3ori, d4ori, d5ori])
orientations = normalizeOrientations(orientations+generateNoise())

localOrdersHistory = []


for t in range(20):
    #print("orientations:")
    #print(orientations)


    neighbours = getNeighbours(positions)
    #print(neighbours)


    #neighbour_indices = tf.boolean_mask(indices, neighbours)
    #print(neighbour_indices)

        
    localOrders = getLocalOrders(orientations, neighbours)
    localOrdersHistory.append(localOrders)

    print("localorders")
    print(localOrders)

    vals = getDecisions(t, 10, localOrders, localOrdersHistory, vals)
    print("vals:")
    print(vals)
    orientations = computeNewOrientation(nsm, k, neighbours, positions, orientations, vals)
    print("orientations:")
    print(orientations)
