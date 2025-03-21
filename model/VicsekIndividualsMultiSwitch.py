import pandas as pd
import numpy as np

from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from enums.EnumSwitchType import SwitchType
from enums.EnumColourType import ColourType
from enums.EnumThresholdEvaluationMethod import ThresholdEvaluationMethod

import services.ServiceOrientations as ServiceOrientations
import services.ServiceVicsekHelper as ServiceVicsekHelper
import services.ServiceSavedModel as ServiceSavedModel
import services.ServiceThresholdEvaluation as ServiceThresholdEvaluation

import model.SwitchInformation as SwitchInformation

import DefaultValues as dv

class VicsekWithNeighbourSelection():

    def __init__(self, domainSize, radius, noise, numberOfParticles, k, neighbourSelectionMechanism,
                 speed=dv.DEFAULT_SPEED, switchSummary=None, events=None, degreesOfVision=dv.DEFAULT_DEGREES_OF_VISION, 
                 activationTimeDelays=[], isActivationTimeDelayRelevantForEvents=False, colourType=None, 
                 thresholdEvaluationMethod=ThresholdEvaluationMethod.LOCAL_ORDER, updateIfNoNeighbours=True, returnHistories=True,
                 logPath=None, logInterval=1):
        """
        Params:
            - domainSize (tuple of floats): the size of the domain
            - radius (float): the perception radius of the individuals
            - noise (float): the noise in the environment that is applied to the orientation of each particle at every timestep
            - numberOfParticles (int): how many particles are in the domain
            - k (int): how many neighbours should each individual consider (start value if k-switching is active)
            - neighbourSelectonMechansim (NeighbourSelectionMechanism): which neighbours each individual should consider (start value if nsm-switching is active)
            - speed (float) [optional]: the speed at which the particles move
            - switchSummary (SwitchSummary) [optional]: The switches that are available to the particles
            - events (list of BaseEvents or child classes) [optional]: the events that occur within the domain during the simulation
            - degreesOfVision (float, range(0, 2pi)) [optional]: how much of their surroundings each individual is able to see. By default 2pi
            - activationTimeDelays (array of int) [optional]: how often each individual updates its orientation
            - isActivationTimeDelayRelevantForEvents (boolean) [optional]: whether an individual should also ignore events when it is not ready to update its orientation
            - colourType (ColourType) [optional]: if and how individuals should be coloured for future rendering
        """

        self.domainSize = np.asarray(domainSize)
        self.radius = radius
        self.noise = noise
        self.numberOfParticles = numberOfParticles
        self.k = k
        self.neighbourSelectionMechanism = neighbourSelectionMechanism
        self.speed = speed

        self.switchSummary = switchSummary

        self.events = events
        self.degreesOfVision = degreesOfVision
        self.activationTimeDelays = np.array(activationTimeDelays)
        self.isActivationTimeDelayRelevantForEvents = isActivationTimeDelayRelevantForEvents
        self.colourType = colourType
        self.thresholdEvaluationMethod = thresholdEvaluationMethod
        self.updateIfNoNeighbours = updateIfNoNeighbours
        self.returnHistories = returnHistories
        self.logPath = logPath
        self.logInterval = logInterval

        # Preparation of constants
        self.minReplacementValue = -1
        self.maxReplacementValue = domainSize[0] * domainSize[1] + 1
        self.disorderPlaceholder = -1
        self.orderPlaceholder = -2

        # Preparation of active switchTypes
        self.switchTypes = [k for k, v in self.switchSummary.actives.items() if v == True]

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
                    "degreesOfVision": self.degreesOfVision,
                    "activationTimeDelays": self.activationTimeDelays.tolist(),
                    "isActivationTimeDelayRelevantForEvents": self.isActivationTimeDelayRelevantForEvents,
                    }

        if self.colourType != None:
            summary["colourType"] = self.colourType.value
            if self.exampleId != None:
                summary["exampleId"] = self.exampleId.tolist()

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


    def initializeState(self):
        """
        Initialises the state of the swarm at the start of the simulation.

        Params:
            None
        
        Returns:
            Arrays of positions and orientations containing values for every individual within the system
        """
        positions = self.domainSize*np.random.rand(self.numberOfParticles,len(self.domainSize))
        orientations = ServiceOrientations.normalizeOrientations(np.random.rand(self.numberOfParticles, len(self.domainSize))-0.5)

        return positions, orientations
    
    def initialiseSwitchingValues(self):
        """
        Initialises the valus that may be affected by switching: neighbour selection mechanisms, ks, speeds, time delays.

        Params:
            None

        Returns:
            Numpy arrays containing the neighbour selection mechanisms, ks, speeds and time delays for each particle.
        """

        nsms = np.full(self.numberOfParticles, self.orderPlaceholder)
        nsmsDf = pd.DataFrame(nsms, columns=["nsms"])
        nsmsDf["nsms"] = nsmsDf["nsms"].replace(self.orderPlaceholder, self.neighbourSelectionMechanism.value)
        nsms = np.array(nsmsDf["nsms"])

        ks = np.array(self.numberOfParticles * [self.k])
        speeds = np.full(self.numberOfParticles, self.speed)

        activationTimeDelays = np.ones(self.numberOfParticles)

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
            info = self.switchSummary.getBySwitchType(SwitchType.ACTIVATION_TIME_DELAY)
            if info != None and info.initialValues != None:
                activationTimeDelays = info.initialValues

        if len(self.activationTimeDelays) > 0:
            activationTimeDelays = self.activationTimeDelays

        return nsms, ks, speeds, activationTimeDelays

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
    
    def __getPickedNeighbourIndices(self, sortedIndices, kMaxPresent, ks):
        """
        Chooses the indices of the neighbours that will be considered for updates.

        Params:
            - sortedIndices (arrays of ints): the sorted indices of all neighbours
            - kMaxPresent (int): what is the highest value of k present in the current values of k
            - ks (array of int): the current values of k for every individual

        Returns:
            Array containing the selected indices for each individual.
        """
        if self.switchSummary != None and self.switchSummary.isActive(SwitchType.K):
            kSwitch = self.switchSummary.getBySwitchType(SwitchType.K)
            kMin, kMax = self.switchSummary.getMinMaxValuesForKSwitchIfPresent()
            
            candidatesOrder = sortedIndices[:, :kSwitch.orderSwitchValue]
            if kSwitch.orderSwitchValue < kMax:
                candidatesOrder = ServiceVicsekHelper.padArray(candidatesOrder, self.numberOfParticles, kMin=kMin, kMax=kMax)

            candidatesDisorder = sortedIndices[:, :kSwitch.disorderSwitchValue]
            if kSwitch.disorderSwitchValue < kMax:
                candidatesDisorder = ServiceVicsekHelper.padArray(candidatesDisorder, self.numberOfParticles, kMin=kMin, kMax=kMax)

            candidates = np.where(((ks == kSwitch.orderSwitchValue)[:, None]), candidatesOrder, candidatesDisorder)
        else:
            candidates = sortedIndices[:, :self.k]
        return candidates
    
    def __checkPickedForNeighbourhood(self, posDiff, candidates, kMaxPresent):
        """
        Verifies that all the selected neighbours are within the perception radius.

        Params:
            - posDiff (array of arrays of float): the position difference between every pair of individuals
            - candidates (array of int): the indices of the selected neighbours
            - kMaxPresent (int): waht is the highest value of k present in the current values of k

        Returns:
            An array of int indices of the selected neighbours that are actually within the neighbourhood.
        """
        if len(candidates) == 0 or len(candidates[0]) == 0:
            return candidates
        # exclude any individuals that are not neighbours
        pickedDistances = np.take_along_axis(posDiff, candidates, axis=1)
        minusOnes = np.full((self.numberOfParticles,kMaxPresent), -1)
        picked = np.where(((candidates == -1) | (pickedDistances > self.radius**2)), minusOnes, candidates)
        return picked
    
    def __createBooleanMaskFromPickedNeighbourIndices(self, picked, kMax):
        """
        Creates a boolean mask from the indices of the selected neighbours.

        Params:
            - picked (array of array of int): the selected indices for each individual

        Returns:
            An array of arrays of booleans representing which neighbours have been selected by each individual.
        """
        if len(picked) == 0 or len(picked[0]) == 0:
            return np.full((self.numberOfParticles, self.numberOfParticles), False)
        # create the boolean mask
        ns = np.full((self.numberOfParticles,self.numberOfParticles+1), False) # add extra dimension to catch indices that are not applicable
        pickedValues = np.full((self.numberOfParticles, kMax), True)
        np.put_along_axis(ns, picked, pickedValues, axis=1)
        ns = ns[:, :-1] # remove extra dimension to catch indices that are not applicable
        return ns
    
    def __getPickedNeighbours(self, posDiff, candidates, ks, isMin):
        """
        Determines which neighbours the individuals should consider.

        Params:
            - posDiff (array of arrays of floats): the distance from every individual to all other individuals
            - candidates (array of arrays of floats): represents either the position distance between each pair of individuals or a fillValue if they are not neighbours  
            - ks (array of ints): which value of k every individual observes
            - isMin (boolean) [optional, default=True]: whether to take the nearest or farthest neighbours

        Returns:
            An array of arrays of booleans representing the selected neighbours
        """
        
        kMax = np.max(ks)

        if self.switchSummary != None and self.switchSummary.isActive(SwitchType.K):
            _, kMax = self.switchSummary.getMinMaxValuesForKSwitchIfPresent()

        sortedIndices = candidates.argsort(axis=1)
        if isMin == False:
            sortedIndices = np.flip(sortedIndices, axis=1)
        
        picked = self.__getPickedNeighbourIndices(sortedIndices=sortedIndices, kMaxPresent=kMax, ks=ks)
        picked = self.__checkPickedForNeighbourhood(posDiff=posDiff, candidates=picked, kMaxPresent=kMax)
        mask = self.__createBooleanMaskFromPickedNeighbourIndices(picked, kMax)
        return mask        
            
    def pickPositionNeighbours(self, positions, neighbours, ks, isMin=True):
        """
        Determines which neighbours the individuals should considered based on the neighbour selection mechanism and k with regard to position.

        Params:
            - positions (array of floats): the position of every individual at the current timestep
            - neighbours (array of arrays of booleans): the identity of every neighbour of every 
            - ks (array of ints): which value of k every individual observes
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
        return self.__getPickedNeighbours(posDiff=posDiff, candidates=candidates, ks=ks, isMin=isMin)
    
    def pickOrientationNeighbours(self, positions, orientations, neighbours, ks, isMin=True):
        """
        Determines which neighbours the individuals should consider based on the neighbour selection mechanism and k with regard to orientation.

        Params:
            - positions (array of floats): the position of every individual at the current timestep
            - orientations (array of floats): the orientation of every individual at the current timestep
            - neighbours (array of arrays of booleans): the identity of every neighbour of every individual
            - ks (array of ints): which value of k every individual observes
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
        return self.__getPickedNeighbours(posDiff=posDiff, candidates=candidates, ks=ks, isMin=isMin)
    
    def pickRandomNeighbours(self, positions, neighbours, ks):
        """
        Determines which neighbours the individuals should consider based on random selection.

        Params:
            - positions (array of floats): the position of every individual at the current timestep
            - neighbours (array of arrays of booleans): the identity of every neighbour of every individual
            - ks (array of ints): which value of k every individual observes

        Returns:
            An array of arrays of booleans representing the selected neighbours
        """
        np.fill_diagonal(neighbours, False)
        posDiff = ServiceVicsekHelper.getPositionDifferences(positions, self.domainSize)
        kMax = np.max(ks)
        
        candidateIndices = ServiceVicsekHelper.getIndicesForTrueValues(neighbours, paddingType='repetition')
        rng = np.random.default_rng()
        rng.shuffle(candidateIndices, axis=1)
        if self.switchSummary != None and self.switchSummary.isActive(SwitchType.K):
            kMin, kMax = self.switchSummary.getMinMaxValuesForKSwitchIfPresent()
            if len(candidateIndices[0]) < kMax:
                candidateIndices = ServiceVicsekHelper.padArray(candidateIndices, self.numberOfParticles, kMin, kMax)
        elif kMax < self.k:
            candidateIndices = ServiceVicsekHelper.padArray(candidateIndices, self.numberOfParticles, kMax, self.k)
        elif len(candidateIndices[0]) < kMax:
            candidateIndices = ServiceVicsekHelper.padArray(candidateIndices, self.numberOfParticles, len(candidateIndices[0]), kMax)
        picked = self.__getPickedNeighbourIndices(sortedIndices=candidateIndices, kMaxPresent=kMax, ks=ks)
        picked = self.__checkPickedForNeighbourhood(posDiff=posDiff, candidates=picked, kMaxPresent=kMax)
        selection = self.__createBooleanMaskFromPickedNeighbourIndices(picked, kMax)
        np.fill_diagonal(selection, True)
        return selection

    def getPickedNeighboursForNeighbourSelectionMechanism(self, neighbourSelectionMechanism, positions, orientations, neighbours, ks):
        """
        Determines which neighbours should be considered by each individual.

        Params:
            - neighbourSelectionMechanism (NeighbourSelectionMechanism): how the neighbours should be selected
            - positions (array of floats): the position of every individual at the current timestep
            - orientations (array of floats): the orientation of every individual at the current timestep
            - neighbours (array of arrays of booleans): the identity of every neighbour of every individual
            - ks (array of ints): which value of k every individual observes
        """
        match neighbourSelectionMechanism:
            case NeighbourSelectionMechanism.NEAREST:
                pickedNeighbours = self.pickPositionNeighbours(positions, neighbours, ks, isMin=True)
            case NeighbourSelectionMechanism.FARTHEST:
                pickedNeighbours = self.pickPositionNeighbours(positions, neighbours, ks, isMin=False)
            case NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE:
                pickedNeighbours = self.pickOrientationNeighbours(positions, orientations, neighbours, ks, isMin=True)
            case NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE:
                pickedNeighbours = self.pickOrientationNeighbours(positions, orientations, neighbours, ks, isMin=False)
            case NeighbourSelectionMechanism.RANDOM:
                pickedNeighbours = self.pickRandomNeighbours(positions, neighbours, ks)
            case NeighbourSelectionMechanism.ALL:
                pickedNeighbours = neighbours
        return pickedNeighbours

    def computeNewOrientations(self, neighbours, positions, orientations, nsms, ks, activationTimeDelays):
        """
        Computes the new orientation of every individual based on the neighbour selection mechanisms, ks, time delays and Vicsek-like 
        averaging.
        Also sets the colours for ColourType.EXAMPLE.

        Params:
            - neighbours (array of arrays of booleans): the identity of every neighbour of every individual
            - positions (array of floats): the position of every individual at the current timestep
            - orientations (array of floats): the orientation of every individual at the current timestep
            - nsms (array of NeighbourSelectionMechanism): the neighbour selection mechanism used by every individual at the current timestep
            - ks (array of ints): the number of neighbours k used by every individual at the current timestep
            - activationTimeDelays (array of ints): at what rate updates are possible for every individual at the current timestep

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
            pickedNeighbours = np.where(((nsms == nsmsSwitch.orderSwitchValue.value)), neighboursOrder, neighboursDisorder)
            
        else:
            pickedNeighbours = self.getPickedNeighboursForNeighbourSelectionMechanism(neighbourSelectionMechanism=self.neighbourSelectionMechanism,
                                                                                      positions=positions, 
                                                                                      orientations=orientations, 
                                                                                      neighbours=neighbours,
                                                                                      ks=ks)

        np.fill_diagonal(pickedNeighbours, True)

        oldOrientations = np.copy(orientations)

        orientations = self.calculateMeanOrientations(orientations, pickedNeighbours)
        orientations = ServiceOrientations.normalizeOrientations(orientations+self.generateNoise())
        
        orientations = ServiceVicsekHelper.revertTimeDelayedChanges(self.t, oldOrientations, orientations, activationTimeDelays)

        if self.colourType == ColourType.EXAMPLE and self.exampleId != None:
            self.colours = np.full(self.numberOfParticles, 'k')
            self.colours[pickedNeighbours[self.exampleId][0]] = 'y'
            self.colours[self.exampleId] = 'r'

        return orientations
     
    def getDecisions(self, t, neighbours, thresholdEvaluationChoiceValues, previousthresholdEvaluationChoiceValues, switchType, switchTypeValues, blocked):
        """
        Computes whether the individual chooses to use option A or option B as its value based on the local order, 
        the average previous local order and a threshold.

        Params:
            - t (int): the current timestep
            - thresholdEvaluationChoiceValues (array of floats): the local order from the point of view of every individual
            - previousthresholdEvaluationChoiceValues of arrays of floats): the local order for every individual at every previous time step
            - switchType (SwitchType): the property that the values are assigned to
            - switchTypeValues (array of ints): the current switchTypeValue selection for every individual
            - blocked (array of booleans): whether updates are possible

        Returns:
            Numpy array containing the updated switchTypeValues for every individual for the given switchType.
        """
        switchInfo = self.switchSummary.getBySwitchType(switchType)
        switchDifferenceThresholdLower = switchInfo.lowerThreshold
        switchDifferenceThresholdUpper = switchInfo.upperThreshold

        prev = np.average(previousthresholdEvaluationChoiceValues[max(t-switchInfo.numberPreviousStepsForThreshold, 0):t+1], axis=0)

        oldWithNewOrderValues = np.where(((thresholdEvaluationChoiceValues >= switchDifferenceThresholdUpper) & (prev <= switchDifferenceThresholdUpper) & (blocked != True)), np.full(len(switchTypeValues), switchInfo.getOrderValue()), switchTypeValues)
        updatedSwitchValues = np.where(((thresholdEvaluationChoiceValues <= switchDifferenceThresholdLower) & (prev >= switchDifferenceThresholdLower) & (blocked != True)), np.full(len(switchTypeValues), switchInfo.getDisorderValue()), oldWithNewOrderValues)
        if self.updateIfNoNeighbours == False:
            neighbour_counts = np.count_nonzero(neighbours, axis=1)
            updatedSwitchValues = np.where((neighbour_counts <= 1), switchTypeValues, updatedSwitchValues)
        return updatedSwitchValues
    
    def appendSwitchValues(self, nsms, ks, speeds, activationTimeDelays):
        """
        Appends all relevant switch type values to the history.

        Params:
            - nsms (array of NeighbourSelectionMechanism): how each individual selects its neighbours
            - ks (array of ints): how many neighbours each individual considers
            - speeds (array of floats): how fast each agent moves
            - activationTimeDelays (array of ints): how long each agent waits until it is ready to update its orientation again

        Returns:
            Nothing.
        """
        if self.switchSummary == None:
            return
        if self.switchSummary.isActive(SwitchType.NEIGHBOUR_SELECTION_MECHANISM):
            self.switchTypeValuesHistory['nsms'].append(nsms)
        if self.switchSummary.isActive(SwitchType.K):
            self.switchTypeValuesHistory['ks'].append(ks)
        if self.switchSummary.isActive(SwitchType.SPEED):
            self.switchTypeValuesHistory['speeds'].append(speeds)
        if self.switchSummary.isActive(SwitchType.ACTIVATION_TIME_DELAY):
            self.switchTypeValuesHistory['activationTimeDelays'].append(activationTimeDelays)
    
    def prepareSimulation(self, initialState, dt, tmax):
        """
        Prepares the simulation by initialising all necessary properties.

        Params:
            - initialState (tuple of arrays) [optional]: A tuple containing the initial positions of all particles, their initial orientations and their initial switch type values
            - dt (int) [optional]: time step
            - tmax (int) [optional]: the total number of time steps of the experiment

        Returns:
            Arrays containing the positions, orientations, neighbour selection mechanisms, ks, speeds and time delays.
        """
         # Preparations and setting of parameters if they are not passed to the method
        
        if any(ele is None for ele in initialState):
            positions, orientations = self.initializeState()
        else:
            positions, orientations = initialState

        nsms, ks, speeds, activationTimeDelays = self.initialiseSwitchingValues()

        #print(f"t=pre, order={ServiceMetric.computeGlobalOrder(orientations)}")

        if dt is None and tmax is not None:
            dt = 1
        
        if tmax is None:
            tmax = (10**3)*dt
            dt = np.average(10**(-2)*(np.max(self.domainSize)/speeds))

        self.tmax = tmax
        self.dt = dt

        # Initialisations for the loop and the return variables
        self.numIntervals=int(tmax/dt+1)

        self.thresholdEvaluationChoiceValuesHistory = []  
        if self.returnHistories:
            self.positionsHistory = np.zeros((self.numIntervals,self.numberOfParticles,len(self.domainSize)))
            self.orientationsHistory = np.zeros((self.numIntervals,self.numberOfParticles,len(self.domainSize)))  
            self.switchTypeValuesHistory = {'nsms': [], 'ks': [], 'speeds': [], 'activationTimeDelays': []}
            if self.colourType != None:
                self.coloursHistory = self.numIntervals * [self.numberOfParticles * ['k']]

            self.positionsHistory[0,:,:]=positions
            self.orientationsHistory[0,:,:]=orientations
            self.appendSwitchValues(nsms, ks, speeds, activationTimeDelays)

        if self.logPath:
            ServiceSavedModel.logModelParams(path=f"{self.logPath}_modelParams", modelParamsDict=self.getParameterSummary())
            ServiceSavedModel.initialiseCsvFileHeaders(path=self.logPath)

        return positions, orientations, nsms, ks, speeds, activationTimeDelays
    
    def handleEvents(self, t, positions, orientations, nsms, ks, speeds, activationTimeDelays):
        """
        Handles all types of events.

        Params:
            - t (int): the current timestep
            - positions (array of (x,y)-coordinates): the position of every particle at the current timestep
            - orientations (array of (u,v)-coordinates): the orientation of every particle at the current timestep
            - nsms (array of NeighbourSelectionMechanism): how every particle selects its neighbours at the current timestep
            - ks (array of ints): how many neighbours each particle considers at the current timestep
            - speeds (array of floats): how fast each particle moves at the current timestep
            - activationTimeDelays (array of ints): how often a particle is ready to update its orientation

        Returns:
            Arrays containing the updates orientations, neighbour selecton mechanisms, ks, speeds, which particles are blocked from updating and the colours assigned to each particle.
        """
        blocked = np.full(self.numberOfParticles, False)
        colours = np.full(self.numberOfParticles, 'k')
        if self.events != None:
                for event in self.events:
                    orientations, nsms, ks, speeds, blocked, colours = event.check(self.numberOfParticles, t, positions, orientations, nsms, ks, speeds, self.dt, activationTimeDelays, self.isActivationTimeDelayRelevantForEvents, self.colourType)
        return orientations, nsms, ks, speeds, blocked, colours

    def simulate(self, initialState=(None, None, None), dt=None, tmax=None):
        """
        Runs the simulation experiment.
        First the parameters are computed if they are not passed. 
        Then the positions and orientations are computed for each particle at each time step.

        Params:
            - initialState (tuple of arrays) [optional]: A tuple containing the initial positions of all particles, their initial orientations and their initial switch type values
            - dt (int) [optional]: time step
            - tmax (int) [optional]: the total number of time steps of the experiment

        Returns:
            (times, positionsHistory, orientationsHistory), the history of the switchValues as a dictionary  and optionally coloursHistory. All except the switchValueHistory as ordered arrays so that they can be matched by index matching
        """
       
        positions, orientations, nsms, ks, speeds, activationTimeDelays = self.prepareSimulation(initialState=initialState, dt=dt, tmax=tmax)
        if self.colourType == ColourType.EXAMPLE:
            self.exampleId = np.random.choice(self.numberOfParticles, 1)
        for t in range(self.numIntervals):
            self.t = t
            if t % 5000 == 0:
                print(f"t={t}/{self.tmax}")
            # if self.t % 100 == 0:
            #     print(f"{t}: {ServiceMetric.computeGlobalOrder(orientations)}")

            # all neighbours (including self)
            neighbours = ServiceVicsekHelper.getNeighboursWithLimitedVision(positions=positions, orientations=orientations, domainSize=self.domainSize,
                                                                            radius=self.radius, degreesOfVision=self.degreesOfVision)
            
            orientations, nsms, ks, speeds, blocked, self.colours = self.handleEvents(t, positions, orientations, nsms, ks, speeds, activationTimeDelays)


            if self.switchSummary != None:
                thresholdEvaluationChoiceValues = ServiceThresholdEvaluation.getThresholdEvaluationValuesForChoice(thresholdEvaluationMethod=self.thresholdEvaluationMethod, positions=positions, orientations=orientations, neighbours=neighbours, domainSize=self.domainSize)

                self.thresholdEvaluationChoiceValuesHistory.append(thresholdEvaluationChoiceValues)
            
                if SwitchType.NEIGHBOUR_SELECTION_MECHANISM in self.switchSummary.switches.keys():
                    nsms = self.getDecisions(t, neighbours, thresholdEvaluationChoiceValues, self.thresholdEvaluationChoiceValuesHistory, SwitchType.NEIGHBOUR_SELECTION_MECHANISM, nsms, blocked)
                if SwitchType.K in self.switchSummary.switches.keys():
                    ks = self.getDecisions(t, neighbours, thresholdEvaluationChoiceValues, self.thresholdEvaluationChoiceValuesHistory, SwitchType.K, ks, blocked)
                if SwitchType.SPEED in self.switchSummary.switches.keys():
                    speeds = self.getDecisions(t, neighbours, thresholdEvaluationChoiceValues, self.thresholdEvaluationChoiceValuesHistory, SwitchType.SPEED, speeds, blocked)
                if SwitchType.ACTIVATION_TIME_DELAY in self.switchSummary.switches.keys():
                    activationTimeDelays = self.getDecisions(t, neighbours, thresholdEvaluationChoiceValues, self.thresholdEvaluationChoiceValuesHistory, SwitchType.ACTIVATION_TIME_DELAY, activationTimeDelays, blocked)

            orientations = self.computeNewOrientations(neighbours, positions, orientations, nsms, ks, activationTimeDelays)

            positions += self.dt*(orientations.T * speeds).T
            positions += -self.domainSize*np.floor(positions/self.domainSize)

            if self.returnHistories:
                self.positionsHistory[t,:,:]=positions
                self.orientationsHistory[t,:,:]=orientations
                self.appendSwitchValues(nsms, ks, speeds, activationTimeDelays)
                if self.colourType != None:
                    self.coloursHistory[t] = self.colours
            
            if self.logPath and t % self.logInterval == 0:
                switchValues = {'nsms': nsms, 'ks': ks, 'speeds': speeds, 'activationTimeDelays': activationTimeDelays}
                ServiceSavedModel.saveModelTimestep(timestep=t, 
                                                    positions=positions, 
                                                    orientations=orientations,
                                                    colours=self.colours,
                                                    path=self.logPath,
                                                    switchValues=switchValues,
                                                    switchTypes=self.switchTypes)
            

            # if t % 500 == 0:
            #     print(f"t={t}, th={self.thresholdEvaluationMethod.name}, order={ServiceMetric.computeGlobalOrder(orientations)}")
            
        if self.returnHistories:
            if self.colourType == None:
                return (self.dt*np.arange(self.numIntervals), self.positionsHistory, self.orientationsHistory), self.switchTypeValuesHistory
            else:
                return (self.dt*np.arange(self.numIntervals), self.positionsHistory, self.orientationsHistory), self.switchTypeValuesHistory, self.coloursHistory
        else:
            if self.colourType == None:
                return (None, None, None), None
            else:
                return (None, None, None), None, None
        