import pandas as pd
import numpy as np

from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from enums.EnumSwitchType import SwitchType
from enums.EnumColourType import ColourType
from enums.EnumThresholdEvaluationMethod import ThresholdEvaluationMethod

import services.ServiceOrientations as ServiceOrientations
import services.ServiceVicsekHelper as ServiceVicsekHelper
import services.ServiceMetric as ServiceMetric
import services.ServiceThresholdEvaluation as ServiceThresholdEvaluation

import model.SwitchInformation as SwitchInformation
from model.VicsekIndividualsMultiSwitch  import VicsekWithNeighbourSelection

import DefaultValues as dv

class VicsekWithNeighbourSelectionOscillation(VicsekWithNeighbourSelection):

    def __init__(self, domainSize, radius, noise, numberOfParticles, k, neighbourSelectionMechanism,
                 speed=dv.DEFAULT_SPEED, switchSummary=None, events=None, degreesOfVision=dv.DEFAULT_DEGREES_OF_VISION, 
                 activationTimeDelays=[], isActivationTimeDelayRelevantForEvents=False, colourType=None, 
                 thresholdEvaluationMethod=ThresholdEvaluationMethod.LOCAL_ORDER, updateIfNoNeighbours=True,
                 individualistic_stress_delta=0.01, social_stress_delta=0.01, stress_num_neighbours=2):
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
        super().__init__(domainSize=domainSize,
                         radius=radius,
                         noise=noise,
                         numberOfParticles=numberOfParticles,
                         k=k,
                         neighbourSelectionMechanism=neighbourSelectionMechanism,
                         speed=speed,
                         switchSummary=switchSummary,
                         events=events,
                         degreesOfVision=degreesOfVision,
                         activationTimeDelays=activationTimeDelays,
                         isActivationTimeDelayRelevantForEvents=isActivationTimeDelayRelevantForEvents,
                         colourType=colourType,
                         thresholdEvaluationMethod=thresholdEvaluationMethod,
                         updateIfNoNeighbours=updateIfNoNeighbours)

        self.individualistic_stress_delta = individualistic_stress_delta
        self.social_stress_delta = social_stress_delta
        self.stress_num_neighbours = stress_num_neighbours

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
                    "individualistic_stress_delta": self.individualistic_stress_delta,
                    "social_stress_delta": self.social_stress_delta,
                    "stress_num_neighbours": self.stress_num_neighbours
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
     
    def getDecisions(self, t, neighbours, thresholdEvaluationChoiceValues, previousthresholdEvaluationChoiceValues, switchType, switchTypeValues, blocked, stressLevels):
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

        stressCorrectedEvaluationValues = thresholdEvaluationChoiceValues + stressLevels
        oldWithNewOrderValues = np.where(((stressCorrectedEvaluationValues >= switchDifferenceThresholdUpper) & (prev <= switchDifferenceThresholdUpper) & (blocked != True)), np.full(len(switchTypeValues), switchInfo.getOrderValue()), switchTypeValues)
        updatedSwitchValues = np.where(((stressCorrectedEvaluationValues <= switchDifferenceThresholdLower) & (prev >= switchDifferenceThresholdLower) & (blocked != True)), np.full(len(switchTypeValues), switchInfo.getDisorderValue()), oldWithNewOrderValues)
        if self.updateIfNoNeighbours == False:
            neighbour_counts = np.count_nonzero(neighbours, axis=1)
            updatedSwitchValues = np.where((neighbour_counts <= 1), switchTypeValues, updatedSwitchValues)
        return updatedSwitchValues
    
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

        stressLevels = np.zeros(self.numberOfParticles)

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
        self.positionsHistory = np.zeros((self.numIntervals,self.numberOfParticles,len(self.domainSize)))
        self.orientationsHistory = np.zeros((self.numIntervals,self.numberOfParticles,len(self.domainSize)))  
        self.stressLevelsHistory = np.zeros((self.numIntervals,self.numberOfParticles))
        self.switchTypeValuesHistory = {'nsms': [], 'ks': [], 'speeds': [], 'activationTimeDelays': []}
        if self.colourType != None:
            self.coloursHistory = self.numIntervals * [self.numberOfParticles * ['k']]

        self.positionsHistory[0,:,:]=positions
        self.orientationsHistory[0,:,:]=orientations
        self.stressLevelsHistory[0,:]=stressLevels
        self.appendSwitchValues(nsms, ks, speeds, activationTimeDelays)

        return positions, orientations, nsms, ks, speeds, activationTimeDelays, stressLevels
    
    def updateStressLevels(self, stressLevels, neighbours):
        neighbour_counts = np.count_nonzero(neighbours, axis=1)
        if self.t % 500 == 0:
            print(f"avg neighbours = {np.average(neighbour_counts)}")
        stressLevels = np.where((neighbour_counts > self.stress_num_neighbours), stressLevels - self.social_stress_delta, stressLevels)
        stressLevels = np.where((neighbour_counts < self.stress_num_neighbours), stressLevels + self.individualistic_stress_delta, stressLevels)
        return stressLevels

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
       
        positions, orientations, nsms, ks, speeds, activationTimeDelays, stressLevels = self.prepareSimulation(initialState=initialState, dt=dt, tmax=tmax)
        if self.colourType == ColourType.EXAMPLE:
            self.exampleId = np.random.choice(self.numberOfParticles, 1)
        for t in range(self.numIntervals):
            self.t = t
            # if t % 5000 == 0:
            #     print(f"t={t}/{self.tmax}")
            # if self.t % 100 == 0:
            #     print(f"{t}: {ServiceMetric.computeGlobalOrder(orientations)}")

            # all neighbours (including self)
            neighbours = ServiceVicsekHelper.getNeighboursWithLimitedVision(positions=positions, orientations=orientations, domainSize=self.domainSize,
                                                                            radius=self.radius, degreesOfVision=self.degreesOfVision)
            stressLevels = self.updateStressLevels(stressLevels, neighbours)
            orientations, nsms, ks, speeds, blocked, self.colours = self.handleEvents(t, positions, orientations, nsms, ks, speeds, activationTimeDelays)

            if self.switchSummary != None:
                thresholdEvaluationChoiceValues = ServiceThresholdEvaluation.getThresholdEvaluationValuesForChoice(thresholdEvaluationMethod=self.thresholdEvaluationMethod, positions=positions, orientations=orientations, neighbours=neighbours, domainSize=self.domainSize)

                self.thresholdEvaluationChoiceValuesHistory.append(thresholdEvaluationChoiceValues)
            
                if SwitchType.NEIGHBOUR_SELECTION_MECHANISM in self.switchSummary.switches.keys():
                    nsms = self.getDecisions(t, neighbours, thresholdEvaluationChoiceValues, self.thresholdEvaluationChoiceValuesHistory, SwitchType.NEIGHBOUR_SELECTION_MECHANISM, nsms, blocked, stressLevels)
                if SwitchType.K in self.switchSummary.switches.keys():
                    ks = self.getDecisions(t, neighbours, thresholdEvaluationChoiceValues, self.thresholdEvaluationChoiceValuesHistory, SwitchType.K, ks, blocked, stressLevels)
                if SwitchType.SPEED in self.switchSummary.switches.keys():
                    speeds = self.getDecisions(t, neighbours, thresholdEvaluationChoiceValues, self.thresholdEvaluationChoiceValuesHistory, SwitchType.SPEED, speeds, blocked, stressLevels)
                if SwitchType.ACTIVATION_TIME_DELAY in self.switchSummary.switches.keys():
                    activationTimeDelays = self.getDecisions(t, neighbours, thresholdEvaluationChoiceValues, self.thresholdEvaluationChoiceValuesHistory, SwitchType.ACTIVATION_TIME_DELAY, activationTimeDelays, blocked, stressLevels)

            orientations = self.computeNewOrientations(neighbours, positions, orientations, nsms, ks, activationTimeDelays)

            positions += self.dt*(orientations.T * speeds).T
            positions += -self.domainSize*np.floor(positions/self.domainSize)

            self.positionsHistory[t,:,:]=positions
            self.orientationsHistory[t,:,:]=orientations
            self.stressLevelsHistory[t,:]=stressLevels

            self.appendSwitchValues(nsms, ks, speeds, activationTimeDelays)
            if self.colourType != None:
                self.coloursHistory[t] = self.colours

            # if t % 500 == 0:
            #     print(f"t={t}, th={self.thresholdEvaluationMethod.name}, order={ServiceMetric.computeGlobalOrder(orientations)}")
            
        if self.colourType == None:
            return (self.dt*np.arange(self.numIntervals), self.positionsHistory, self.orientationsHistory), self.switchTypeValuesHistory, self.stressLevelsHistory
        else:
            return (self.dt*np.arange(self.numIntervals), self.positionsHistory, self.orientationsHistory), self.switchTypeValuesHistory, self.coloursHistory, self.stressLevelsHistory
