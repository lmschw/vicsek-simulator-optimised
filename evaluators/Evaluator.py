import services.ServiceSavedModel as ServiceSavedModel
import services.ServiceMetric as ServiceMetric
from enums.EnumMetrics import Metrics

import numpy as np

class Evaluator(object):
    """
    Implementation of the evaluation mechanism for the Vicsek model for a single model.
    """
    def __init__(self, modelParams, metric, simulationData=None, evaluationTimestepInterval=1, threshold=0.01, switchTypeValues=np.array([None]), switchTypeOptions=(None, None)):
        """
        Initialises the evaluator.

        Parameters:
            - modelParams (array of dictionaries): contains the model parameters for the current model
            - metric (EnumMetrics.Metrics) [optional]: the metric according to which the models' performances should be evaluated
            - simulationData (array of (time array, positions array, orientation array, colours array)) [optional]: contains all the simulation data
            - evaluationTimestepInterval (int) [optional]: the interval of the timesteps to be evaluated. By default, every time step is evaluated
            - threshold (float) [optional]: the threshold for the AgglomerativeClustering cutoff
            - switchTypeValues (array of arrays of switchTypeValues) [optional]: the switch type value of every particle at every timestep
            - switchTypeOptions (tuple) [optional]: the two possible values for the switch type value
        
        Returns:
            Nothing.
        """
        if simulationData != None:
            self.time, self.positions, self.orientations = simulationData
        self.modelParams = modelParams
        self.metric = metric
        self.evaluationTimestepInterval = evaluationTimestepInterval
        self.threshold = threshold
        self.switchTypeValues = switchTypeValues
        self.switchTypeOptions = switchTypeOptions
        self.domainSize = np.array(modelParams["domainSize"])

        if metric in [Metrics.CLUSTER_NUMBER, 
                      Metrics.CLUSTER_SIZE, 
                      Metrics.CLUSTER_NUMBER_WITH_RADIUS,
                      Metrics.AVERAGE_NUMBER_NEIGHBOURS,
                      Metrics.MIN_AVG_MAX_NUMBER_NEIGHBOURS,
                      Metrics.AVG_DISTANCE_NEIGHBOURS]:
            self.radius = modelParams["radius"]
        else:
            self.radius = None
            

    def evaluate(self, startTimestep=0, endTimestep=None, saveTimestepsResultsPath=None):
        """
        Evaluates the model according to the metric specified for the evaluator.

        Returns:
            A dictionary with the results for the model at every time step.
        """
        if len(self.time) < 1:
            print("ERROR: cannot evaluate without simulationData. Please supply simulationData, modelParams and metric at Evaluator instantiation.")
            return
        if endTimestep == None:
            endTimestep = len(self.time)
        valuesPerTimeStep = {}
        for i in range(len(self.time)):
            #if i % 100 == 0:
                #print(f"evaluating {i}/{len(self.time)}")
            if i % self.evaluationTimestepInterval == 0 and i >= startTimestep and i <= endTimestep:
                if any(ele is None for ele in self.switchTypeValues):
                    valuesPerTimeStep[self.time[i]] = ServiceMetric.evaluateSingleTimestep(positions=self.positions[i], orientations=self.orientations[i], metric=self.metric, domainSize=self.domainSize, radius=self.radius, threshold=self.threshold)
                else:
                    valuesPerTimeStep[self.time[i]] = ServiceMetric.evaluateSingleTimestep(positions=self.positions[i], orientations=self.orientations[i], metric=self.metric, domainSize=self.domainSize, radius=self.radius, threshold=self.threshold, switchTypeValues=self.switchTypeValues[i], switchTypeOptions=self.switchTypeOptions)

        #print("Evaluation completed.")
        if saveTimestepsResultsPath != None:
            ServiceSavedModel.saveTimestepsResults(valuesPerTimeStep, saveTimestepsResultsPath, self.modelParams)
        return valuesPerTimeStep
    