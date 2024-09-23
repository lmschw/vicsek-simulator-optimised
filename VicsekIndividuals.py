import pandas as pd
import numpy as np

from EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from EnumSwitchType import SwitchType

import ServiceOrientations

import DefaultValues as dv

class VicsekWithNeighbourSelection:

    def __init__(self, domainSize, radius, noise, numberOfParticles, k, neighbourSelectionMechanism,
                 speed=dv.DEFAULT_SPEED, switchType=None, switchValues=(None, None), 
                 orderThresholds=None, numberPreviousStepsForThreshold=10, switchingActive=True):
        self.domainSize = np.asarray(domainSize)
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
                    "domainSize": self.domainSize.tolist(),
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


    def __initializeState(self):
        """
        Initialises the state of the swarm at the start of the simulation.

        Params:
            None
        
        Returns:
            Arrays of positions, orientations and switchTypeValues containing values for every individual within the system
        """
        positions = self.domainSize*np.random.rand(self.numberOfParticles,len(self.domainSize))
        orientations = ServiceOrientations.normalizeOrientations(np.random.rand(self.numberOfParticles, len(self.domainSize))-0.5)
        match self.switchType:
            case SwitchType.NEIGHBOUR_SELECTION_MODE:
                switchTypeValues = self.numberOfParticles * [self.neighbourSelectionMechanism]
            case SwitchType.K:
                switchTypeValues = self.numberOfParticles * [self.k]
            case _:
                switchTypeValues = self.numberOfParticles * [None]
        return positions, orientations, switchTypeValues

    def generateNoise(self):
        """
        Generates some noise based on the noise amplitude set at creation.

        Params:
            None

        Returns:
            An array with the noise to be added to each individual
        """
        return np.random.normal(scale=self.noise, size=(self.numberOfParticles, len(self.domainSize)))

    def calculateMeanOrientations(self, orientations, neighbours):
        """
        Computes the average of the orientations of all selected neighbours for every individual.

        Params:
            - orientations (array of floats): the orientation of every individual at the current timestep
            - neighbours (array of arrays of booleans): the identity of every neighbour of every individual

        Returns:
            An array of floats containing the new, normalised orientations of every individual
        """
        summedOrientations = np.sum(neighbours[:,:,np.newaxis]*orientations[np.newaxis,:,:],axis=1)
        return ServiceOrientations.normalizeOrientations(summedOrientations)
    
    def __getDifferences(self, array):
        """
        Computes the differences between all individuals for the values provided by the array.

        Params:
            - array (array of floats): the values to be compared

        Returns:
            An array of arrays of floats containing the difference between each pair of values.
        """
        rij=array[:,np.newaxis,:]-array   
        rij = rij - self.domainSize*np.rint(rij/self.domainSize) #minimum image convention
        return np.sum(rij**2,axis=2)

    def getOrientationDifferences(self, orientations):
        """
        Helper method to gloss over identical differences implementation for position and orientation. 
        """
        return self.__getDifferences(orientations)
    
    def getPositionDifferences(self, positions):
        """
        Helper method to gloss over identical differences implementation for position and orientation. 
        """
        return self.__getDifferences(positions)

    def getNeighbours(self, positions):
        """
        Determines all the neighbours for each individual.

        Params:
            - positions (array of floats): the position of every individual at the current timestep

        Returns:
            An array of arrays of booleans representing whether or not any two individuals are neighbours
        """
        rij2 = self.getPositionDifferences(positions)
        return (rij2 <= self.radius**2)
    
    def __getPickedNeighboursFromMaskedArray(self, maskedArray, posDiff, ks, isMin):
        """
        Determines which neighbours the individuals should considered based on a preexisting maskedArray, the neighbour selection mechanism and k.

        Params:
            - maskedArray (MaskedArray): masked array containing the values for consideration
            - posDiff (array of arrays of floats): the distance from every individual to all other individuals
            - ks (Dataframe): which value of k every individual observes (in column "val")
            - isMin (boolean) [optional, default=True]: whether to take the nearest or farthest neighbours

        Returns:
            An array of arrays of booleans representing the selected neighbours
        """
        kMax = np.max(ks)
        minusDiff = np.full((self.numberOfParticles,kMax-np.min(ks)), -1)

        sortedIndices = maskedArray.argsort(axis=1)
        if isMin == False:
            sortedIndices = np.flip(sortedIndices, axis=1)
        
        candidatesOrder = sortedIndices[:, :self.orderSwitchValue]
        if self.orderSwitchValue < kMax:
            candidatesOrder = np.concatenate((candidatesOrder, minusDiff), axis=1)

        candidatesDisorder = sortedIndices[:, :self.disorderSwitchValue]
        if self.disorderSwitchValue < kMax:
            candidatesDisorder = np.concatenate((candidatesDisorder, minusDiff), axis=1)

        candidates = np.where(((ks == self.orderSwitchValue)), candidatesOrder, candidatesDisorder)

        # exclude any individuals that are not neighbours
        pickedDistances = np.take_along_axis(posDiff, candidates, axis=1)
        minusOnes = np.full((self.numberOfParticles,kMax), -1)
        picked = np.where(((candidates == -1) | (pickedDistances == 0) | (pickedDistances > self.radius**2)), minusOnes, candidates)

        # create the boolean mask
        ns = np.full((self.numberOfParticles,self.numberOfParticles+1), False) # add extra dimension to catch indices that are not applicable
        pickedValues = np.full((self.numberOfParticles, self.k), True)
        np.put_along_axis(ns, picked, pickedValues, axis=1)
        ns = ns[:, :-1] # remove extra dimension to catch indices that are not applicable
        return ns        
            
    def pickPositionNeighbours(self, positions, neighbours, ks, isMin=True):
        """
        Determines which neighbours the individuals should considered based on the neighbour selection mechanism and k.

        Params:
            - positions (array of floats): the position of every individual at the current timestep
            - neighbours (array of arrays of booleans): the identity of every neighbour of every 
            - ks (Dataframe): which value of k every individual observes (in column "val")
            - isMin (boolean) [optional, default=True]: whether to take the nearest or farthest neighbours

        Returns:
            An array of arrays of booleans representing the selected neighbours
        """
        posDiff = self.getPositionDifferences(positions)
        if isMin == True:
            fillValue = self.maxReplacementValue
        else:
            fillValue = self.minReplacementValue

        # select the best candidates
        maskedArray = np.ma.MaskedArray(posDiff, mask=neighbours==False, fill_value=fillValue)
        return self.__getPickedNeighboursFromMaskedArray(maskedArray=maskedArray, posDiff=posDiff, ks=ks, isMin=isMin)
    
    def pickOrientationNeighbours(self, positions, orientations, neighbours, ks, isMin=True):
        """
        Determines which neighbours the individuals should considered based on the neighbour selection mechanism and k.

        Params:
            - positions (array of floats): the position of every individual at the current timestep
            - orientations (array of floats): the orientation of every individual at the current timestep
            - neighbours (array of arrays of booleans): the identity of every neighbour of every individual
            - ks (Dataframe): which value of k every individual observes (in column "val")
            - isMin (boolean) [optional, default=True]: whether to take the least orientionally different or most orientationally different neighbours

        Returns:
            An array of arrays of booleans representing the selected neighbours
        """
        posDiff = self.getPositionDifferences(positions)
        orientDiff = self.getOrientationDifferences(orientations)

        if isMin == True:
            fillValue = self.maxReplacementValue
        else:
            fillValue = self.minReplacementValue

        # select the best candidates
        maskedArray = np.ma.MaskedArray(orientDiff, mask=neighbours==False, fill_value=fillValue)
        return self.__getPickedNeighboursFromMaskedArray(maskedArray=maskedArray, posDiff=posDiff, ks=ks, isMin=isMin)

    def computeNewOrientations(self, neighbours, positions, orientations, switchTypeValues):
        """
        Computes the new orientation of every individual based on the neighbour selection mechanism, k and Vicsek-like 
        averaging.

        Params:
            - neighbours (array of arrays of booleans): the identity of every neighbour of every individual
            - positions (array of floats): the position of every individual at the current timestep
            - orientations (array of floats): the orientation of every individual at the current timestep
            - switchTypeValues (array of ints): the chosen value for every individual at the current timestep

        Returns:
            An array of floats representing the orientations of all individuals after the current timestep
        """

        ks = np.array(self.numberOfParticles * [self.k])
        nsms = np.array(self.numberOfParticles * [self.neighbourSelectionMechanism])
        if self.switchingActive:
            match self.switchType:
                case SwitchType.NEIGHBOUR_SELECTION_MODE:
                    nsms = switchTypeValues
                case SwitchType.K:
                    ks = switchTypeValues

        match self.neighbourSelectionMechanism:
            case NeighbourSelectionMechanism.NEAREST:
                pickedNeighbours = self.pickPositionNeighbours(positions, neighbours, ks, isMin=True)
            case NeighbourSelectionMechanism.FARTHEST:
                pickedNeighbours = self.pickPositionNeighbours(positions, neighbours, ks, isMin=False)
            case NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE:
                pickedNeighbours = self.pickOrientationNeighbours(positions, orientations, neighbours, ks, isMin=True)
            case NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE:
                pickedNeighbours = self.pickOrientationNeighbours(positions, orientations, neighbours, ks, isMin=False)
            case NeighbourSelectionMechanism.ALL:
                pickedNeighbours = neighbours

        np.fill_diagonal(pickedNeighbours, True)

        orientations = self.calculateMeanOrientations(orientations, pickedNeighbours)
        orientations = ServiceOrientations.normalizeOrientations(orientations+self.generateNoise())

        return orientations

    def getLocalOrders(self, orientations, neighbours):
        """
        Computes the local order for every individual.

        Params: 
            - orientations (array of floats): the orientation of every individual at the current timestep
            - neighbours (array of arrays of booleans): the identity of every neighbour of every individual

        Returns:
            An array of floats representing the local order for every individual at the current time step (values between 0 and 1)
        """
        sumOrientation = np.sum(neighbours[:,:,np.newaxis]*orientations[np.newaxis,:,:],axis=1)
        localOrders = np.divide(np.sqrt(np.sum(sumOrientation**2,axis=1)), np.count_nonzero(neighbours, axis=1))
        return localOrders
    
    def __getLowerAndUpperThreshold(self):
        """
        Determines the lower and upper thresholds for hysteresis.

        Params:
            None

        Returns:
            Two floats representing the lower and upper threshold respectively
        """
        if len(self.orderThresholds) == 1:
            switchDifferenceThresholdLower = self.orderThresholds[0]
            switchDifferenceThresholdUpper = 1 - self.orderThresholds[0]
        else:
            switchDifferenceThresholdLower = self.orderThresholds[0]
            switchDifferenceThresholdUpper = self.orderThresholds[1]
        return switchDifferenceThresholdLower, switchDifferenceThresholdUpper
        
    def getDecisions(self, t, localOrders, previousLocalOrders, switchTypeValues):
        """
        Computes whether the individual chooses to use option A or option B as its value based on the local order, 
        the average previous local order and a threshold.

        Params:
            - t (int): the current timestep
            - localOrders (array of floats): the local order from the point of view of every individual
            - previousLocalOrders (array of arrays of floats): the local order for every individual at every previous time step
            - switchTypeValues (array of ints): the current switchTypeValue selection for every individual

        Returns:
            A pandas Dataframe containing the switchTypeValues for every individual
        """
        switchDifferenceThresholdLower, switchDifferenceThresholdUpper = self.__getLowerAndUpperThreshold()

        prev = np.average(previousLocalOrders[max(t-self.numberPreviousStepsForThreshold, 0):t+1], axis=0)
        switchTypeValuesDf = pd.DataFrame(switchTypeValues)
        switchTypeValuesDf["localOrder"] = localOrders
        switchTypeValuesDf["previousLocalOrder"] = prev
        switchTypeValuesDf["val"] = switchTypeValuesDf["val"].case_when([(((switchTypeValuesDf["localOrder"] >= switchDifferenceThresholdUpper) & (switchTypeValuesDf["previousLocalOrder"] <= switchDifferenceThresholdUpper)), self.orderSwitchValue),
                            (((switchTypeValuesDf["localOrder"] <= switchDifferenceThresholdLower) & (switchTypeValuesDf["previousLocalOrder"] >= switchDifferenceThresholdLower)), self.disorderSwitchValue),
        ])
        return pd.DataFrame(switchTypeValuesDf["val"])
    
    def computeOrder(self, orientations):
        # TODO remove this method. This is only here to make debugging easier
        """
        Computes the order within the provided orientations. 
        Can also be called for a subsection of all particles by only providing their orientations.

        Params:
            - orientations (array of (u,v)-coordinates): the orientation of all particles that should be included
        
        Returns:
            A float representing the order in the given orientations
        """
        sumOrientation = [0,0]
        for j in range(len(orientations)):
            sumOrientation += orientations[j]
        return np.sqrt(sumOrientation[0]**2 + sumOrientation[1]**2) / len(orientations)

    def simulate(self, initialState=(None, None, None), dt=None, tmax=None):
        """
        Runs the simulation experiment.
        First the parameters are computed if they are not passed. 
        Then the positions, orientations and colours are computed for each particle at each time step.

        Params:
            - initialState (tuple of arrays) [optional]: A tuple containing the initial positions of all particles, their initial orientations and their initial switch type values
            - dt (int) [optional]: time step
            - tmax (int) [optional]: the total number of time steps of the experiment

        Returns:
            times, positionsHistory, orientationsHistory, coloursHistory, switchTypeValueHistory. All of them as ordered arrays so that they can be matched by index matching
        """
        # Preparations and setting of parameters if they are not passed to the method
        positions, orientations, switchTypeValues = initialState
        
        if any(ele is None for ele in initialState):
            positions, orientations, switchTypeValues = self.__initializeState()

        switchTypeValues = pd.DataFrame(switchTypeValues, columns=["val"])
        switchTypeValues["val"][0] = 1
        switchTypeValues["val"][4] = 1
        switchTypeValues["val"][-1] = 1
            
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

        for t in range(numIntervals):

            if t % 1000 == 0:
                print(f"t={t}/{tmax}")

            # all neighbours (including self)
            neighbours = self.getNeighbours(positions)

            if self.switchingActive:
                localOrders = self.getLocalOrders(orientations, neighbours)
                localOrdersHistory.append(localOrders)
            
                switchTypeValues = self.getDecisions(t, localOrders, localOrdersHistory, switchTypeValues)

            orientations = self.computeNewOrientations(neighbours, positions, orientations, switchTypeValues)

            positions += dt*(self.speed*orientations)
            positions += -self.domainSize*np.floor(positions/self.domainSize)


            positionsHistory[t,:,:]=positions
            orientationsHistory[t,:,:]=orientations
            switchTypeValuesHistory[t]=switchTypeValues

            if t % 1000 == 0:
                print(f"t={t}, order={self.computeOrder(orientations)}")
            
        return (dt*np.arange(numIntervals), positionsHistory, orientationsHistory), switchTypeValuesHistory
