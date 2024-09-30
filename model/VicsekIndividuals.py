import pandas as pd
import numpy as np

from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from enums.EnumSwitchType import SwitchType

import services.ServiceOrientations as ServiceOrientations
import services.ServiceVicsekHelper as ServiceVicsekHelper
import services.ServiceMetric as ServiceMetric

import DefaultValues as dv

class VicsekWithNeighbourSelection:

    def __init__(self, domainSize, radius, noise, numberOfParticles, k, neighbourSelectionMechanism,
                 speed=dv.DEFAULT_SPEED, speeds=None, switchType=None, switchValues=(None, None), 
                 orderThresholds=None, numberPreviousStepsForThreshold=10, switchingActive=True,
                 events=None):
        self.domainSize = np.asarray(domainSize)
        self.radius = radius
        self.noise = noise
        self.numberOfParticles = numberOfParticles
        self.k = k
        self.neighbourSelectionMechanism = neighbourSelectionMechanism
        self.speeds = speeds
        if speed and speeds == None:
            self.speeds = np.full(numberOfParticles, speed)

        self.switchType = switchType
        self.orderSwitchValue, self.disorderSwitchValue = switchValues
        self.orderThresholds = orderThresholds
        self.numberPreviousStepsForThreshold = numberPreviousStepsForThreshold
        self.switchingActive = switchingActive

        self.minReplacementValue = -1
        self.maxReplacementValue = domainSize[0] * domainSize[1] + 1
        self.events = events

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
                    "speeds": self.speeds.tolist(),
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
            case SwitchType.NEIGHBOUR_SELECTION_MECHANISM:
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
        
        kMaxPresent = np.max(ks)

        sortedIndices = maskedArray.argsort(axis=1)
        if isMin == False:
            sortedIndices = np.flip(sortedIndices, axis=1)
        
        if self.switchingActive and self.switchType == SwitchType.K:
            kMin = np.min([self.orderSwitchValue, self.disorderSwitchValue])
            kMax = np.max([self.orderSwitchValue, self.disorderSwitchValue])
            minusDiff = np.full((self.numberOfParticles,kMax-kMin), -1)
            candidatesOrder = sortedIndices[:, :self.orderSwitchValue]
            if self.orderSwitchValue < kMax and kMax == kMaxPresent:
                candidatesOrder = np.concatenate((candidatesOrder, minusDiff), axis=1)

            candidatesDisorder = sortedIndices[:, :self.disorderSwitchValue]
            if self.disorderSwitchValue < kMax and kMax == kMaxPresent:
                candidatesDisorder = np.concatenate((candidatesDisorder, minusDiff), axis=1)

            candidates = np.where(((ks == self.orderSwitchValue)[:, None]), candidatesOrder, candidatesDisorder)
        else:
            candidates = sortedIndices[:, :self.k]

        # exclude any individuals that are not neighbours
        pickedDistances = np.take_along_axis(posDiff, candidates, axis=1)
        minusOnes = np.full((self.numberOfParticles,kMaxPresent), -1)
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
        posDiff = ServiceVicsekHelper.getPositionDifferences(positions, self.domainSize)
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
        posDiff = ServiceVicsekHelper.getPositionDifferences(positions, self.domainSize)
        orientDiff = ServiceVicsekHelper.getOrientationDifferences(orientations, self.domainSize)

        if isMin == True:
            fillValue = self.maxReplacementValue
        else:
            fillValue = self.minReplacementValue

        # select the best candidates
        maskedArray = np.ma.MaskedArray(orientDiff, mask=neighbours==False, fill_value=fillValue)
        return self.__getPickedNeighboursFromMaskedArray(maskedArray=maskedArray, posDiff=posDiff, ks=ks, isMin=isMin)
    

    def getPickedNeighboursForNeighbourSelectionMechanism(self, neighbourSelectionMechanism, positions, orientations, neighbours, ks):
        match neighbourSelectionMechanism:
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
        return pickedNeighbours

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

        if self.switchingActive and self.switchType == SwitchType.K:
            ks = switchTypeValues
        else:
            ks = np.array(self.numberOfParticles * [self.k])

        if self.switchingActive and self.switchType == SwitchType.NEIGHBOUR_SELECTION_MECHANISM:
            neighboursOrder = self.getPickedNeighboursForNeighbourSelectionMechanism(neighbourSelectionMechanism=self.orderSwitchValue,
                                                                                     positions=positions,
                                                                                     orientations=orientations,
                                                                                     neighbours=neighbours,
                                                                                     ks=ks)
            neighboursDisorder = self.getPickedNeighboursForNeighbourSelectionMechanism(neighbourSelectionMechanism=self.disorderSwitchValue,
                                                                                    positions=positions,
                                                                                    orientations=orientations,
                                                                                    neighbours=neighbours,
                                                                                    ks=ks)
            pickedNeighbours = np.where(((switchTypeValues == self.orderSwitchValue)), neighboursDisorder, neighboursOrder)

            
        else:
            pickedNeighbours = self.getPickedNeighboursForNeighbourSelectionMechanism(neighbourSelectionMechanism=self.neighbourSelectionMechanism,
                                                                                      positions=positions, 
                                                                                      orientations=orientations, 
                                                                                      neighbours=neighbours,
                                                                                      ks=ks)

        np.fill_diagonal(pickedNeighbours, True)

        orientations = self.calculateMeanOrientations(orientations, pickedNeighbours)
        orientations = ServiceOrientations.normalizeOrientations(orientations+self.generateNoise())

        return orientations
    
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
        switchTypeValuesDf = pd.DataFrame(switchTypeValues, columns=["val"])
        switchTypeValuesDf["localOrder"] = localOrders
        switchTypeValuesDf["previousLocalOrder"] = prev
        switchTypeValuesDf["val"] = switchTypeValuesDf["val"].case_when([(((switchTypeValuesDf["localOrder"] >= switchDifferenceThresholdUpper) & (switchTypeValuesDf["previousLocalOrder"] <= switchDifferenceThresholdUpper)), self.orderSwitchValue),
                            (((switchTypeValuesDf["localOrder"] <= switchDifferenceThresholdLower) & (switchTypeValuesDf["previousLocalOrder"] >= switchDifferenceThresholdLower)), self.disorderSwitchValue),
        ])
        return np.array(switchTypeValuesDf["val"])
    
    def prepareSimulation(self, initialState, dt, tmax):
         # Preparations and setting of parameters if they are not passed to the method
        positions, orientations, switchTypeValues = initialState
        
        if any(ele is None for ele in initialState):
            positions, orientations, switchTypeValues = self.__initializeState()

        print(f"t=pre, order={ServiceMetric.computeGlobalOrder(orientations)}")

        if dt is None and tmax is not None:
            dt = 1
        
        if tmax is None:
            tmax = (10**3)*dt
            dt = np.average(10**(-2)*(np.max(self.domainSize)/self.speeds))

        self.tmax = tmax
        self.dt = dt

        # Initialisations for the loop and the return variables
        self.numIntervals=int(tmax/dt+1)

        self.localOrdersHistory = []  
        self.positionsHistory = np.zeros((self.numIntervals,self.numberOfParticles,len(self.domainSize)))
        self.orientationsHistory = np.zeros((self.numIntervals,self.numberOfParticles,len(self.domainSize)))  
        self.switchTypeValuesHistory = []

        self.positionsHistory[0,:,:]=positions
        self.orientationsHistory[0,:,:]=orientations
        self.switchTypeValuesHistory.append(switchTypeValues)

        return positions, orientations, switchTypeValues
    
    def handleEvents(self, t, positions, orientations):
        if self.events != None:
                for event in self.events:
                    orientations = event.check(self.numberOfParticles, t, positions, orientations)
        return orientations

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
       
        positions, orientations, switchTypeValues = self.prepareSimulation(initialState=initialState, dt=dt, tmax=tmax)
        for t in range(self.numIntervals):
            if t % 1000 == 0:
                print(f"t={t}/{self.tmax}")

            orientations = self.handleEvents(t, positions, orientations)

            # all neighbours (including self)
            neighbours = ServiceVicsekHelper.getNeighbours(positions, self.domainSize, self.radius)

            if self.switchingActive:
                localOrders = ServiceMetric.computeLocalOrders(orientations, neighbours)
                self.localOrdersHistory.append(localOrders)
            
                switchTypeValues = self.getDecisions(t, localOrders, self.localOrdersHistory, switchTypeValues)

                if self.switchType == SwitchType.SPEED:
                    self.speeds = switchTypeValues

            orientations = self.computeNewOrientations(neighbours, positions, orientations, switchTypeValues)

            positions += self.dt*(orientations.T * self.speeds).T
            positions += -self.domainSize*np.floor(positions/self.domainSize)

            self.positionsHistory[t,:,:]=positions
            self.orientationsHistory[t,:,:]=orientations
            self.switchTypeValuesHistory.append(switchTypeValues)

            if t % 1000 == 0:
                print(f"t={t}, order={ServiceMetric.computeGlobalOrder(orientations)}")
            
        return (self.dt*np.arange(self.numIntervals), self.positionsHistory, self.orientationsHistory), np.array(self.switchTypeValuesHistory)
