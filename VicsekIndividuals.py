import tensorflow as tf
import pandas as pd
import numpy as np
import random

from EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from EnumSwitchType import SwitchType

import ServiceOrientations
import DefaultValues as dv

class VicsekWithNeighbourSelection:

    def __init__(self, domainSize, radius, noise, numberOfParticles, k, neighbourSelectionMechanism,
                 speed=dv.DEFAULT_SPEED, switchType=None, switchValues=(None, None), 
                 orderThresholds=None, numberPreviousStepsForThreshold=10, switchingActive=True):
        self.domainSize = domainSize
        self.radius = radius
        self.noise = noise
        self.numberOfParticles = numberOfParticles
        self.k = k
        self.neighbourSelectionMechanism = neighbourSelectionMechanism
        self.speed = speed
        self.switchType = switchType
        self.orderSwitchValue, self.disorderSwitchValue = switchValues
        self.orderThresholds = orderThresholds
        self.numberPreviousStepsForThreshold = numberPreviousStepsForThreshold
        self.switchingActive = switchingActive

        self.minReplacementValue = -1
        self.maxReplacementValue = domainSize[0] * domainSize[1] + 1

    def __initializeState(self, domainSize, numberOfParticles):
        positions = domainSize*np.random.rand(numberOfParticles,len(domainSize))
        orientations = ServiceOrientations.normalizeOrientations(np.random.rand(numberOfParticles, len(domainSize))-0.5)
        match self.switchType:
            case SwitchType.NEIGHBOUR_SELECTION_MODE:
                switchTypeValues = numberOfParticles * [self.neighbourSelectionMode]
            case SwitchType.K:
                switchTypeValues = numberOfParticles * [self.k]
            case _:
                switchTypeValues = numberOfParticles * [None]
        return positions, orientations, switchTypeValues

    def generateNoise(self):
        return np.random.normal(scale=self.noise, size=(self.numberOfParticles, len(self.domainSize)))

    def calculateMeanOrientations(self, orientations, neighbours):
        summedOrientations = np.sum(neighbours[:,:,np.newaxis]*orientations[np.newaxis,:,:],axis=1)
        return ServiceOrientations.normalizeOrientations(summedOrientations)

    """
    def normalizeOrientations(self, orientations):
        return orientations/(np.sqrt(np.sum(orientations**2,axis=1))[:,np.newaxis])
    """
    def getPositionDifferences(self, positions):
        rij=positions[:,np.newaxis,:]-positions
        #rij=rij[~np.eye(rij.shape[0],dtype=bool),:].reshape(rij.shape[0],rij.shape[0]-1,-1) #remove i<>i interaction
            
        rij = rij - self.domainSize*np.rint(rij/self.domainSize) #minimum image convention

        rij2 = np.sum(rij**2,axis=2)
        return rij2

    def getNeighbours(self, positions):
        rij2 = self.getPositionDifferences(positions)

        neighbours = (rij2 <= self.radius**2)
        np.fill_diagonal(neighbours, False)
        return neighbours

    def pickPositionNeighbours(self, positions, neighbours, isMin=True):
        posDiff = self.getPositionDifferences(positions)
        if isMin == True:
            #neighbourDiffs = np.where(neighbours == True, posDiff, maxSq)
            fillValue = self.maxReplacementValue
        else:
            #neighbourDiffs = np.where(neighbours == True, posDiff, minSq)
            fillValue = self.minReplacementValue

        #a = np.sort(posDiff, axis=1)
        #mask = neighbours.nonzero()
        #a = np.argsort(neighbourDiffs, axis=1)
        
        minusOnes = np.full((self.numberOfParticles,self.k), -1)
        #trues = np.full((n,n), True)
        #falses = np.full((n,n), False)

        maskedArray = np.ma.MaskedArray(posDiff, mask=neighbours==False, fill_value=fillValue)
        sortedIndices = maskedArray.argsort(axis=1)
        if isMin == False:
            sortedIndices = np.flip(sortedIndices)
        candidates = sortedIndices[:, :self.k]
        
        pickedDistances = np.take_along_axis(posDiff, candidates, axis=1)
        # filter on actual neighbours, e.g. by replacing indices that aren't neighbours by the diagonal index (the agent's own index)
        #diagonalIndices = np.diag_indices(n)[0]
        picked = np.where(((pickedDistances == 0) | (pickedDistances > self.radius**2)), minusOnes, candidates)

        # TODO replace loop
        ns = np.full((self.numberOfParticles,self.numberOfParticles), False)
        for i in range(self.numberOfParticles):
            for j in range(self.numberOfParticles):
                if j in picked[i]:
                    ns[i][j] = True

        #mask = np.zeros((n,n), dtype=np.bool_)
        #mask[picked] = True
        #mask[picked[:,0], picked[:,1]]=True
        #mask = []
        return ns

    def computeNewOrientations(self, neighbours, positions, orientations, vals):

        """
        match switchType:
            case SwitchType.NEIGHBOUR_SELECTION_MODE:
                valsDf = pd.DataFrame(vals)
                valsDf["val"] = valsDf["val"].case_when([(, valsB),
                                    (((valsDf["localOrder"] <= threshold) & (valsDf["previousLocalOrder"] >= threshold)), valsA),
                ])
        """
        match self.neighbourSelectionMechanism:
            case NeighbourSelectionMechanism.NEAREST:
                pickedNeighbours = self.pickPositionNeighbours(positions, neighbours, isMin=True)
            case NeighbourSelectionMechanism.FARTHEST:
                pickedNeighbours = self.pickPositionNeighbours(positions, neighbours, isMin=False)

        orientations = self.calculateMeanOrientations(orientations, pickedNeighbours)
        orientations = ServiceOrientations.normalizeOrientations(orientations+self.generateNoise())
        return orientations

    def getLocalOrders(self, orientations, neighbours):
        sumOrientation = np.sum(neighbours[:,:,np.newaxis]*orientations[np.newaxis,:,:],axis=1)
        localOrders = np.divide(np.sqrt(np.sum(sumOrientation**2,axis=1)), tf.math.count_nonzero(neighbours, axis=1))
        return localOrders
    
    def __getLowerAndUpperThreshold(self):
        if len(self.orderThresholds) == 1:
            switchDifferenceThresholdLower = self.orderThresholds[0]
            switchDifferenceThresholdUpper = 1 - self.orderThresholds[0]
        else:
            switchDifferenceThresholdLower = self.orderThresholds[0]
            switchDifferenceThresholdUpper = self.orderThresholds[1]
        return switchDifferenceThresholdLower, switchDifferenceThresholdUpper
        
    def getDecisions(self, t, localOrders, previousLocalOrders, vals):
        """
        Computes whether the individual chooses to use option A or option B as its value based on the local order, the average previous local order and a threshold.
        """
        switchDifferenceThresholdLower, switchDifferenceThresholdUpper = self.__getLowerAndUpperThreshold()

        prev = np.average(previousLocalOrders[max(t-self.numberPreviousStepsForThreshold, 0):t+1], axis=0)
        valsDf = pd.DataFrame(vals)
        valsDf["localOrder"] = localOrders
        valsDf["previousLocalOrder"] = prev
        valsDf["val"] = valsDf["val"].case_when([(((valsDf["localOrder"] >= switchDifferenceThresholdUpper) & (valsDf["previousLocalOrder"] <= switchDifferenceThresholdUpper)), self.orderSwitchValue),
                            (((valsDf["localOrder"] <= switchDifferenceThresholdLower) & (valsDf["previousLocalOrder"] >= switchDifferenceThresholdLower)), self.disorderSwitchValue),
        ])
        return pd.DataFrame(valsDf["val"])


    def simulate(self, initialState=(None, None, None), dt=None, tmax=None):

        # Preparations and setting of parameters if they are not passed to the method
        positions, orientations, vals = initialState
        
        if any(ele is None for ele in initialState):
            positions, orientations, vals = self.__initializeState(self.domainSize, self.numberOfParticles)
            
        if dt is None and tmax is not None:
            dt = 1
        
        if tmax is None:
            tmax = (10**3)*dt
            dt = 10**(-2)*(np.max(self.domainSize)/self.speed)

        self.tmax = tmax
        self.dt = dt

        # Initialisations for the loop and the return variables
        t=0
        numIntervals=int(tmax/dt+1)

        localOrdersHistory = []    

        """
        print(f"t=prestart")
        print("pos")
        print(positions)
        print("ori")
        print(orientations)
        """
        for t in range(numIntervals):

            neighbours = self.getNeighbours(positions)
                
            localOrders = self.getLocalOrders(orientations, neighbours)
            localOrdersHistory.append(localOrders)

            #vals = self.getDecisions(t, localOrders, localOrdersHistory, vals)

            positions += dt*(self.speed*orientations)
            positions += (-self.domainSize[0], -self.domainSize[1])*np.floor(positions/self.domainSize)

            orientations = self.computeNewOrientations(neighbours, positions, orientations, vals)
            if t >= tmax-5:
                print(f"t={t}")
                print("pos")
                print(positions)
                print("ori")
                print(orientations)
