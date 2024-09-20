import pandas as pd
import numpy as np
import random
import time

from EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from EnumSwitchType import SwitchType

import ServiceOrientations
import ServiceGeneral

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
        self.events = None

    def getParameterSummary(self, asString=False):
        """
        Creates a summary of all the model parameters ready for use for conversion to JSON or strings.

        Parameters:
            - asString (bool, default False) [optional]: if the summary should be returned as a dictionary or as a single string
        
        Returns:
            A dictionary or a single string containing all model parameters.
        """
        summary = {"n": self.numberOfParticles,
                    "k": self.k,
                    "noise": self.noise,
                    "speed": self.speed,
                    "radius": self.radius,
                    "neighbourSelectionMechanism": self.neighbourSelectionMechanism.name,
                    "domainSize": self.domainSize,
                    "tmax": self.tmax,
                    "dt": self.dt,
                    "thresholds": self.orderThresholds,
                    "previousSteps": self.numberPreviousStepsForThreshold,
                    "switchingActive": self.switchingActive,
                    }
        if self.switchingActive:
            summary["switchType"] = self.switchType.name
            summary["orderValue"] = self.orderSwitchValue
            summary["disorderValue"] = self.disorderSwitchValue

        if self.events:
            eventsSummary = []
            for event in self.events:
                eventsSummary.append(event.getParameterSummary())
            summary["events"] = eventsSummary

        if asString:
            strPrep = [tup[0] + ": " + tup[1] for tup in summary.values()]
            return ", ".join(strPrep)
        return summary


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
        #ts = time.time()
        a =  np.random.normal(scale=self.noise, size=(self.numberOfParticles, len(self.domainSize)))

        #te = time.time()
        #ServiceGeneral.logWithTime(f"duration generateNoise(): {ServiceGeneral.formatTime(te-ts)}")
        return a

    def calculateMeanOrientations(self, orientations, neighbours):
        #ts = time.time()

        summedOrientations = np.sum(neighbours[:,:,np.newaxis]*orientations[np.newaxis,:,:],axis=1)
        a =  ServiceOrientations.normalizeOrientations(summedOrientations)
        #te = time.time()
        #ServiceGeneral.logWithTime(f"duration calculateMeanOrientations(): {ServiceGeneral.formatTime(te-ts)}")
        return a

    """
    def normalizeOrientations(self, orientations):
        return orientations/(np.sqrt(np.sum(orientations**2,axis=1))[:,np.newaxis])
    """
    def getPositionDifferences(self, positions):

        #ts = time.time()

        rij=positions[:,np.newaxis,:]-positions
        #rij=rij[~np.eye(rij.shape[0],dtype=bool),:].reshape(rij.shape[0],rij.shape[0]-1,-1) #remove i<>i interaction
            
        rij = rij - self.domainSize*np.rint(rij/self.domainSize) #minimum image convention

        rij2 = np.sum(rij**2,axis=2)
        #te = time.time()
        #ServiceGeneral.logWithTime(f"duration getPositionDifferences(): {ServiceGeneral.formatTime(te-ts)}")
        return rij2

    def getNeighbours(self, positions):
        #ts = time.time()

        rij2 = self.getPositionDifferences(positions)

        neighbours = (rij2 <= self.radius**2)
        np.fill_diagonal(neighbours, False)
        #te = time.time()
        #ServiceGeneral.logWithTime(f"duration getNeighbours(): {ServiceGeneral.formatTime(te-ts)}")
        return neighbours

    def pickPositionNeighbours(self, positions, neighbours, isMin=True):
        #ts = time.time()
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
        """
        pickedDistances = np.take_along_axis(posDiff, candidates, axis=1)
        # filter on actual neighbours, e.g. by replacing indices that aren't neighbours by the diagonal index (the agent's own index)
        #diagonalIndices = np.diag_indices(n)[0]
        picked = np.where(((pickedDistances == 0) | (pickedDistances > self.radius**2)), minusOnes, candidates)
        """
        # TODO replace loop
        
        """
        for i in range(self.numberOfParticles):
            for j in range(self.numberOfParticles):
                if j in candidates[i] and (posDiff[i][j] != 0 and posDiff[i][j]> self.radius**2):
                    ns[i][j] = True
        """
        pickedDistances = np.take_along_axis(posDiff, candidates, axis=1)
        picked = np.where(((pickedDistances == 0) | (pickedDistances > self.radius**2)), minusOnes, candidates)
        
        ns = np.full((self.numberOfParticles,self.numberOfParticles+1), False) # add extra dimension to catch indices that are not applicable
        pickedValues = np.full((self.numberOfParticles, self.k), True)
        #fillers = np.full((self.numberOfParticles, self.numberOfParticles-self.k), -1)
        np.put_along_axis(ns, picked, pickedValues, axis=1)
        ns = ns[:, :-1] # remove extra dimension to catch indices that are not applicable

        #mask = np.zeros((n,n), dtype=np.bool_)
        #mask[picked] = True
        #mask[picked[:,0], picked[:,1]]=True
        #mask = []
        #te = time.time()
        #ServiceGeneral.logWithTime(f"duration pickPositionNeighbours(): {ServiceGeneral.formatTime(te-ts)}")
        return ns

    def computeNewOrientations(self, neighbours, positions, orientations, switchTypeValues):
        #ts = time.time()
        """
        match switchType:
            case SwitchType.NEIGHBOUR_SELECTION_MODE:
                switchTypeValuesDf = pd.DataFrame(switchTypeValues)
                switchTypeValuesDf["val"] = switchTypeValuesDf["val"].case_when([(, switchTypeValuesB),
                                    (((switchTypeValuesDf["localOrder"] <= threshold) & (switchTypeValuesDf["previousLocalOrder"] >= threshold)), switchTypeValuesA),
                ])
        """
        match self.neighbourSelectionMechanism:
            case NeighbourSelectionMechanism.NEAREST:
                pickedNeighbours = self.pickPositionNeighbours(positions, neighbours, isMin=True)
            case NeighbourSelectionMechanism.FARTHEST:
                pickedNeighbours = self.pickPositionNeighbours(positions, neighbours, isMin=False)

        np.fill_diagonal(pickedNeighbours, True)
        orientations = self.calculateMeanOrientations(orientations, pickedNeighbours)
        orientations = ServiceOrientations.normalizeOrientations(orientations+self.generateNoise())
        #te = time.time()
        #ServiceGeneral.logWithTime(f"duration computeNewOrientations(): {ServiceGeneral.formatTime(te-ts)}")
        return orientations

    def getLocalOrders(self, orientations, neighbours):
        #ts = time.time()
        sumOrientation = np.sum(neighbours[:,:,np.newaxis]*orientations[np.newaxis,:,:],axis=1)
        localOrders = np.divide(np.sqrt(np.sum(sumOrientation**2,axis=1)), np.count_nonzero(neighbours, axis=1))
        #te = time.time()
        #ServiceGeneral.logWithTime(f"duration getLocalOrders(): {ServiceGeneral.formatTime(te-ts)}")
        return localOrders
    
    def __getLowerAndUpperThreshold(self):
        #ts = time.time()
        if len(self.orderThresholds) == 1:
            switchDifferenceThresholdLower = self.orderThresholds[0]
            switchDifferenceThresholdUpper = 1 - self.orderThresholds[0]
        else:
            switchDifferenceThresholdLower = self.orderThresholds[0]
            switchDifferenceThresholdUpper = self.orderThresholds[1]
        #te = time.time()
        #ServiceGeneral.logWithTime(f"duration getLowerAndUpperThreshold(): {ServiceGeneral.formatTime(te-ts)}")
        return switchDifferenceThresholdLower, switchDifferenceThresholdUpper
        
    def getDecisions(self, t, localOrders, previousLocalOrders, switchTypeValues):
        """
        Computes whether the individual chooses to use option A or option B as its value based on the local order, the average previous local order and a threshold.
        """
        #ts = time.time()
        switchDifferenceThresholdLower, switchDifferenceThresholdUpper = self.__getLowerAndUpperThreshold()

        prev = np.average(previousLocalOrders[max(t-self.numberPreviousStepsForThreshold, 0):t+1], axis=0)
        switchTypeValuesDf = pd.DataFrame(switchTypeValues)
        switchTypeValuesDf["localOrder"] = localOrders
        switchTypeValuesDf["previousLocalOrder"] = prev
        switchTypeValuesDf["val"] = switchTypeValuesDf["val"].case_when([(((switchTypeValuesDf["localOrder"] >= switchDifferenceThresholdUpper) & (switchTypeValuesDf["previousLocalOrder"] <= switchDifferenceThresholdUpper)), self.orderSwitchValue),
                            (((switchTypeValuesDf["localOrder"] <= switchDifferenceThresholdLower) & (switchTypeValuesDf["previousLocalOrder"] >= switchDifferenceThresholdLower)), self.disorderSwitchValue),
        ])
        a = pd.DataFrame(switchTypeValuesDf["val"])

        #te = time.time()
        #ServiceGeneral.logWithTime(f"duration getDecisions(): {ServiceGeneral.formatTime(te-ts)}")
        return a


    def simulate(self, initialState=(None, None, None), dt=None, tmax=None):
        #ts = time.time()
        # Preparations and setting of parameters if they are not passed to the method
        positions, orientations, switchTypeValues = initialState
        
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
        positionsHistory = np.zeros((numIntervals,self.numberOfParticles,len(self.domainSize)))
        orientationsHistory = np.zeros((numIntervals,self.numberOfParticles,len(self.domainSize)))  
        switchTypeValuesHistory = numIntervals * [self.numberOfParticles * [None]]

        positionsHistory[0,:,:]=positions
        orientationsHistory[0,:,:]=orientations
        switchTypeValuesHistory[0]=switchTypeValues

        """
        print(f"t=prestart")
        print("pos")
        print(positions)
        print("ori")
        print(orientations)
        """
        for t in range(numIntervals):

            if t % 1000 == 0:
                print(f"t={t}/{tmax}")

            neighbours = self.getNeighbours(positions)
                
            localOrders = self.getLocalOrders(orientations, neighbours)
            localOrdersHistory.append(localOrders)

            #switchTypeValues = self.getDecisions(t, localOrders, localOrdersHistory, switchTypeValues)

            positions += dt*(self.speed*orientations)
            positions += (-self.domainSize[0], -self.domainSize[1])*np.floor(positions/self.domainSize)

            orientations = self.computeNewOrientations(neighbours, positions, orientations, switchTypeValues)

            positionsHistory[t,:,:]=positions
            orientationsHistory[t,:,:]=orientations
            switchTypeValuesHistory[t]=switchTypeValues
            
            """
            print(f"t={t}")
            print("pos")
            print(positions)
            print("ori")
            print(orientations)
            """
            
        #te = time.time()
        #ServiceGeneral.logWithTime(f"duration simulate(): {ServiceGeneral.formatTime(te-ts)}")

        return (dt*np.arange(numIntervals), positionsHistory, orientationsHistory), switchTypeValuesHistory
