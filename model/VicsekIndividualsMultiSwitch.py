import pandas as pd
import numpy as np

from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from enums.EnumSwitchType import SwitchType

import services.ServiceOrientations as ServiceOrientations
import services.ServiceVicsekHelper as ServiceVicsekHelper
import services.ServiceMetric as ServiceMetric
import services.ServiceVision as ServiceVision

import model.SwitchInformation as SwitchInformation

import DefaultValues as dv

class VicsekWithNeighbourSelection:

    def __init__(self, domainSize, radius, noise, numberOfParticles, k, neighbourSelectionMechanism,
                 speed=dv.DEFAULT_SPEED, switchSummary=None, events=None, degreesOfVision=dv.DEFAULT_DEGREES_OF_VISION):
        self.domainSize = np.asarray(domainSize)
        self.radius = radius
        self.noise = noise
        self.numberOfParticles = numberOfParticles
        self.k = k
        self.neighbourSelectionMechanism = neighbourSelectionMechanism
        self.speed = speed

        self.switchSummary = switchSummary

        self.minReplacementValue = -1
        self.maxReplacementValue = domainSize[0] * domainSize[1] + 1
        self.disorderPlaceholder = -1
        self.orderPlaceholder = -2

        self.events = events
        self.degreesOfVision = degreesOfVision

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
                    "radius": self.radius,
                    "neighbourSelectionMechanism": self.neighbourSelectionMechanism.name,
                    "domainSize": self.domainSize.tolist(),
                    "tmax": self.tmax,
                    "dt": self.dt,
                    "degreesOfVision": self.degreesOfVision
                    }
        if self.switchSummary != None:
            summary["switchSummary"] = self.switchSummary.getParameterSummary()

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

        return positions, orientations
    
    def initialiseSwitchingValues(self):
        nsms = np.full(self.numberOfParticles, self.orderPlaceholder)
        nsmsDf = pd.DataFrame(nsms, columns=["nsms"])
        nsmsDf["nsms"] = nsmsDf["nsms"].replace(self.orderPlaceholder, self.neighbourSelectionMechanism)
        nsms = np.array(nsmsDf["nsms"])

        ks = np.array(self.numberOfParticles * [self.k])
        speeds = np.full(self.numberOfParticles, self.speed)

        if self.switchSummary != None:
            info = self.switchSummary.getBySwitchType(SwitchType.NEIGHBOUR_SELECTION_MECHANISM)
            if info != None and info.initialValues != None:
                nsms = info.initialValues
            info = self.switchSummary.getBySwitchType(SwitchType.K)
            if info != None and info.initialValues != None:
                ks = info.initialValues
            info = self.switchSummary.getBySwitchType(SwitchType.SPEED)
            if info != None and info.initialValues != None:
                speeds = info.initialValues
        return nsms, ks, speeds

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
    
    def __getPickedNeighboursFromMaskedArray(self, posDiff, candidates, ks, isMin):
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

        sortedIndices = candidates.argsort(axis=1)
        if isMin == False:
            sortedIndices = np.flip(sortedIndices, axis=1)
        
        if self.switchSummary != None and self.switchSummary.isActive(SwitchType.K):
            kSwitch = self.switchSummary.getBySwitchType(SwitchType.K)
            kMin = np.min([kSwitch.orderSwitchValue, kSwitch.disorderSwitchValue])
            kMax = np.max([kSwitch.orderSwitchValue, kSwitch.disorderSwitchValue])
            minusDiff = np.full((self.numberOfParticles,kMax-kMin), -1)
            candidatesOrder = sortedIndices[:, :kSwitch.orderSwitchValue]
            if kSwitch.orderSwitchValue < kMax and kMax == kMaxPresent:
                candidatesOrder = np.concatenate((candidatesOrder, minusDiff), axis=1)

            candidatesDisorder = sortedIndices[:, :kSwitch.disorderSwitchValue]
            if kSwitch.disorderSwitchValue < kMax and kMax == kMaxPresent:
                candidatesDisorder = np.concatenate((candidatesDisorder, minusDiff), axis=1)

            candidates = np.where(((ks == kSwitch.orderSwitchValue)[:, None]), candidatesOrder, candidatesDisorder)
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

        fillVals = np.full((self.numberOfParticles,self.numberOfParticles), fillValue)
        candidates = np.where((neighbours), posDiff, fillVals)

        # select the best candidates
        return self.__getPickedNeighboursFromMaskedArray(posDiff=posDiff, candidates=candidates, ks=ks, isMin=isMin)
    
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

        fillVals = np.full((self.numberOfParticles,self.numberOfParticles), fillValue)
        candidates = np.where((neighbours), orientDiff, fillVals)

        # select the best candidates
        return self.__getPickedNeighboursFromMaskedArray(posDiff=posDiff, candidates=candidates, ks=ks, isMin=isMin)
    

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

    def computeNewOrientations(self, neighbours, positions, orientations, nsms, ks):
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

        if self.switchSummary != None and self.switchSummary.isActive(SwitchType.K):
            ks = ks
        else:
            ks = np.array(self.numberOfParticles * [self.k])

        if self.switchSummary != None and self.switchSummary.isActive(SwitchType.NEIGHBOUR_SELECTION_MECHANISM):
            nsmsSwitch = self.switchSummary.getBySwitchType(SwitchType.NEIGHBOUR_SELECTION_MECHANISM)
            neighboursOrder = self.getPickedNeighboursForNeighbourSelectionMechanism(neighbourSelectionMechanism=nsmsSwitch.orderSwitchValue,
                                                                                     positions=positions,
                                                                                     orientations=orientations,
                                                                                     neighbours=neighbours,
                                                                                     ks=ks)
            neighboursDisorder = self.getPickedNeighboursForNeighbourSelectionMechanism(neighbourSelectionMechanism=nsmsSwitch.disorderSwitchValue,
                                                                                    positions=positions,
                                                                                    orientations=orientations,
                                                                                    neighbours=neighbours,
                                                                                    ks=ks)
            pickedNeighbours = np.where(((nsms == nsmsSwitch.orderSwitchValue)), neighboursDisorder, neighboursOrder)
            
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
     
    def getDecisions(self, t, localOrders, previousLocalOrders, switchType, switchTypeValues, blocked):
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
        switchInfo = self.switchSummary.getBySwitchType(switchType)
        switchDifferenceThresholdLower = switchInfo.lowerThreshold
        switchDifferenceThresholdUpper = switchInfo.upperThreshold

        prev = np.average(previousLocalOrders[max(t-switchInfo.numberPreviousStepsForThreshold, 0):t+1], axis=0)
        switchTypeValuesDf = pd.DataFrame(switchTypeValues, columns=["val"])
        switchTypeValuesDf["localOrder"] = localOrders
        switchTypeValuesDf["previousLocalOrder"] = prev
        switchTypeValuesDf["val"] = switchTypeValuesDf["val"].case_when([(((switchTypeValuesDf["localOrder"] >= switchDifferenceThresholdUpper) & (switchTypeValuesDf["previousLocalOrder"] <= switchDifferenceThresholdUpper) & (blocked != True)), self.orderPlaceholder),
                            (((switchTypeValuesDf["localOrder"] <= switchDifferenceThresholdLower) & (switchTypeValuesDf["previousLocalOrder"] >= switchDifferenceThresholdLower) & (blocked != True)), self.disorderPlaceholder),
        ])
        switchTypeValuesDf["val"] = switchTypeValuesDf["val"].replace(self.orderPlaceholder, switchInfo.orderSwitchValue)
        switchTypeValuesDf["val"] = switchTypeValuesDf["val"].replace(self.disorderPlaceholder, switchInfo.disorderSwitchValue)

        return np.array(switchTypeValuesDf["val"])
    
    def appendSwitchValues(self, nsms, ks, speeds):
        if self.switchSummary == None:
            return
        if self.switchSummary.isActive(SwitchType.NEIGHBOUR_SELECTION_MECHANISM):
            self.switchTypeValuesHistory['nsms'].append(nsms)
        if self.switchSummary.isActive(SwitchType.K):
            self.switchTypeValuesHistory['ks'].append(ks)
        if self.switchSummary.isActive(SwitchType.SPEED):
            self.switchTypeValuesHistory['speeds'].append(speeds)
    
    def prepareSimulation(self, initialState, dt, tmax):
         # Preparations and setting of parameters if they are not passed to the method
        
        if any(ele is None for ele in initialState):
            positions, orientations = self.__initializeState()
        else:
            positions, orientations = initialState

        nsms, ks, speeds = self.initialiseSwitchingValues()

        print(f"t=pre, order={ServiceMetric.computeGlobalOrder(orientations)}")

        if dt is None and tmax is not None:
            dt = 1
        
        if tmax is None:
            tmax = (10**3)*dt
            dt = np.average(10**(-2)*(np.max(self.domainSize)/speeds))

        self.tmax = tmax
        self.dt = dt

        # Initialisations for the loop and the return variables
        self.numIntervals=int(tmax/dt+1)

        self.localOrdersHistory = []  
        self.positionsHistory = np.zeros((self.numIntervals,self.numberOfParticles,len(self.domainSize)))
        self.orientationsHistory = np.zeros((self.numIntervals,self.numberOfParticles,len(self.domainSize)))  
        self.switchTypeValuesHistory = {'nsms': [], 'ks': [], 'speeds': []}

        self.positionsHistory[0,:,:]=positions
        self.orientationsHistory[0,:,:]=orientations
        self.appendSwitchValues(nsms, ks, speeds)

        return positions, orientations, nsms, ks, speeds
    
    def handleEvents(self, t, positions, orientations, nsms, ks, speeds):
        blocked = np.full(self.numberOfParticles, False)
        if self.events != None:
                for event in self.events:
                    orientations, nsms, ks, speeds, blocked = event.check(self.numberOfParticles, t, positions, orientations, nsms, ks, speeds)
        return orientations, nsms, ks, speeds, blocked

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
       
        positions, orientations, nsms, ks, speeds = self.prepareSimulation(initialState=initialState, dt=dt, tmax=tmax)
        for t in range(self.numIntervals):
            if t % 1000 == 0:
                print(f"t={t}/{self.tmax}")

            orientations, nsms, ks, speeds, blocked = self.handleEvents(t, positions, orientations, nsms, ks, speeds)

            # all neighbours (including self)
            neighbours = ServiceVicsekHelper.getNeighboursWithLimitedVision(positions=positions, orientations=orientations, domainSize=self.domainSize,
                                                                            radius=self.radius, degreesOfVision=self.degreesOfVision)

            if self.switchSummary != None:
                localOrders = ServiceMetric.computeLocalOrders(orientations, neighbours)
                self.localOrdersHistory.append(localOrders)
            
                if SwitchType.NEIGHBOUR_SELECTION_MECHANISM in self.switchSummary.switches.keys():
                    nsms = self.getDecisions(t, localOrders, self.localOrdersHistory, SwitchType.NEIGHBOUR_SELECTION_MECHANISM, nsms, blocked)
                if SwitchType.K in self.switchSummary.switches.keys():
                    ks = self.getDecisions(t, localOrders, self.localOrdersHistory, SwitchType.K, ks, blocked)
                if SwitchType.SPEED in self.switchSummary.switches.keys():
                    speeds = self.getDecisions(t, localOrders, self.localOrdersHistory, SwitchType.SPEED, speeds, blocked)

            orientations = self.computeNewOrientations(neighbours, positions, orientations, nsms, ks)

            positions += self.dt*(orientations.T * speeds).T
            positions += -self.domainSize*np.floor(positions/self.domainSize)

            self.positionsHistory[t,:,:]=positions
            self.orientationsHistory[t,:,:]=orientations
            self.appendSwitchValues(nsms, ks, speeds)

            if t % 500 == 0:
                print(f"t={t}, order={ServiceMetric.computeGlobalOrder(orientations)}")
                print(neighbours)
            
        return (self.dt*np.arange(self.numIntervals), self.positionsHistory, self.orientationsHistory), np.array(self.switchTypeValuesHistory)
