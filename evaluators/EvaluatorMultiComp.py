import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import pandas as pd

import evaluators.Evaluator as Evaluator
import services.ServiceSavedModel as ssm
from enums.EnumMetrics import Metrics

# matplotlib default colours with corresponding colours that are 65% and 50% lighter
COLOURS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
BACKGROUND_COLOURS_65_PERCENT_LIGHTER = ['#a6d1f0', '#ffd2ab', '#abe8ab', '#f1b3b3', '#dacae8',
                                         '#dbc1bc', '#f5cfea', '#d2d2d2', '#eff0aa', '#a7eef5']
BACKGROUND_COLOURS_50_PERCENT_LIGHTER = ['#7fbee9', '#ffbf86', '#87de87', '#eb9293', '#c9b3de',
                                         '#cca69f', '#f1bbe0', '#bfbfbf', '#e8e985', '#81e7f1']
BACKGROUND_COLOURS = BACKGROUND_COLOURS_50_PERCENT_LIGHTER
class EvaluatorMultiAvgComp(object):
    """
    Implementation of the evaluation mechanism for the Vicsek model for comparison of multiple models.
    """

    def __init__(self, metric, modelParams=None, simulationData=None, basePaths=[], runRange=[], from_csv=False,
                 evaluationTimestepInterval=1, threshold=0.01, switchTypeValues=None, 
                 switchType=None, switchTypeOptions=None, use_median=True):
        """
        Initialises the evaluator.

        Parameters:
            - modelParams (array of dictionaries): contains the model parameters for each model
            - metric (EnumMetrics.Metrics) [optional]: the metric according to which the models' performances should be evaluated
            - simulationData (array of (time array, positions array, orientation array, colours array)) [optional]: contains all the simulation data for each model
            - evaluationTimestepInterval (int) [optional]: the interval of the timesteps to be evaluated. By default, every time step is evaluated
            - threshold (float) [optional]: the threshold for the AgglomerativeClustering cutoff
            - switchTypeValues (array of arrays of switchTypeValues) [optional]: the switch type value of every particle at every timestep
            - switchType (SwitchType) [optional]: the switch type that was used
            - switchTypeOptions (tuple) [optional]: the two possible values for the switch type value
        
        Returns:
            Nothing.
        """
        self.simulationData = simulationData
        self.modelParams = modelParams
        self.basePaths = basePaths
        self.runRange = runRange
        self.from_csv = from_csv
        self.metric = metric
        self.evaluationTimestepInterval = evaluationTimestepInterval
        self.threshold = threshold
        self.switchTypeValues = switchTypeValues
        self.switchType = switchType
        self.switchTypeOptions = switchTypeOptions
        self.use_median = use_median

    def getResults(self, model):
        results = []
        if self.simulationData:
            run_range = (0, len(self.simulationData[model]))
        else:
            run_range = self.runRange
        for individualRun in range(run_range[0], run_range[1]):
            if self.simulationData:
            #print(f"step {individualRun}/{len(self.simulationData[model])}")
                if self.switchTypeValues == None or self.switchTypeValues == []:
                    evaluator = Evaluator.Evaluator(self.modelParams[model][individualRun], self.metric, self.simulationData[model][individualRun], self.evaluationTimestepInterval, self.threshold)
                else:    
                    evaluator = Evaluator.Evaluator(self.modelParams[model][individualRun], self.metric, self.simulationData[model][individualRun], self.evaluationTimestepInterval, self.threshold, self.switchTypeValues[model][individualRun], self.switchType, self.switchTypeOptions)
            else:
                if self.switchTypeValues == None or self.switchTypeValues == []:
                    switchTypes = [None]
                    if self.from_csv:
                        params, simulationData = ssm.loadModelFromCsv(filepathData=f"{self.basePaths[model]}_{individualRun}.csv", 
                                                                      filePathModelParams=f"{self.basePaths[model]}_{individualRun}_modelParams.csv")
                    else:
                        params, simulationData = ssm.loadModel(path=f"{self.basePaths[model]}_{individualRun}.json")
                else:
                    if self.from_csv:
                        params, simulationData, switchTypes = ssm.loadModelFromCsv(filepathData=f"{self.basePaths[model]}_{individualRun}.csv",
                                                                                   filePathModelParams=f"{self.basePaths[model]}_{individualRun}.csv",
                                                                                   switchTypes=[self.switchType])
                    else:
                        params, simulationData, switchTypes = ssm.loadModel(path=f"{self.basePaths[model]}_{individualRun}.json",
                                                                            switchTypes=[self.switchType],
                                                                            loadSwitchValues=True)
                evaluator = Evaluator.Evaluator(modelParams=params, metric=self.metric, simulationData=simulationData,
                                                evaluationTimestepInterval=self.evaluationTimestepInterval,
                                                threshold=self.threshold, switchTypeValues=switchTypes, switchType=self.switchType,
                                                switchTypeOptions=self.switchTypeOptions)
            result = evaluator.evaluate()
            results.append(result)
        return results

    def evaluate(self):
        """
        Evaluates all models according to the metric specified for the evaluator.

        Parameters:
            None

        Returns:
            A dictionary with the results for each model at every time step.
        """
        dd = defaultdict(list)
        varianceData = []
        if self.simulationData:
            num_models = len(self.simulationData)
        else:
            num_models = len(self.basePaths)
        for model in range(num_models):
            varianceDataModel = []
            #print(f"evaluating {model}/{len(self.simulationData)}")
            results = self.getResults(model)
            
            ddi = defaultdict(list)
            for d in results: 
                for key, value in d.items():
                    ddi[key].append(value)
            if self.metric == Metrics.DUAL_OVERLAY_ORDER_AND_PERCENTAGE:
                for m in range(len(ddi)):
                    idx = m * self.evaluationTimestepInterval
                    dd[idx].append(ddi[idx][0][0])
                    dd[idx].append(ddi[idx][0][1])
            elif self.metric == Metrics.MIN_AVG_MAX_NUMBER_NEIGHBOURS:
                for m in range(len(ddi)):
                    idx = m * self.evaluationTimestepInterval
                    dd[idx].append(ddi[idx][0][0])
                    dd[idx].append(ddi[idx][0][1])
                    dd[idx].append(ddi[idx][0][2])
            else:
                for m in range(len(ddi)):
                    idx = m * self.evaluationTimestepInterval
                    if self.metric == Metrics.CLUSTER_SIZE:
                        for i in range(len(ddi[idx])):
                            ddi[idx][i] = np.max(ddi[idx][i])
                    if self.use_median:
                        dd[idx].append(np.median(ddi[idx]))
                        quantile25 = np.quantile(ddi[idx], 0.25)
                        quantile75 = np.quantile(ddi[idx], 0.75)
                        varianceDataModel.append([quantile25, quantile75])
                    else:
                        dd[idx].append(np.average(ddi[idx]))
                        varianceDataModel.append(np.array(ddi[idx]))
            varianceData.append(varianceDataModel)
        return dd, varianceData

    
    def visualize(self, data, labels, xLabel=None, yLabel=None, subtitle=None, colourBackgroundForTimesteps=None, varianceData=None, xlim=None, ylim=None, savePath=None):
        """
        Visualizes and optionally saves the results of the evaluation as a graph.

        Parameters:
            - data (dictionary): a dictionary with the time step as key and an array of each model's result as values
            - labels (array of strings): the label for each model
            - xLabel (string) [optional]: the label for the x-axis
            - yLabel (string) [optional]: the label for the y-axis
            - subtitle (string) [optional]: subtitle to be included in the title of the visualisation
            - colourBackgroundForTimesteps ([start, stop]) [optional]: the start and stop timestep for the background colouring for the event duration
            - varianceData (array) [optional]: the variance for every timestep
            - xlim (float) [optional]: the x-limit for the plot
            - ylim (float) [optional]: the y-limit for the plot
            - savePath (string) [optional]: the location and name of the file where the model should be saved. Will not be saved unless a savePath is provided

        Returns:
            Nothing.
        """

        match self.metric:
            case Metrics.ORDER:
                if ylim == None:
                    ylim = (0, 1.1)
                self.__createStandardLineplot(data, labels, xlim=xlim, ylim=ylim)
            case Metrics.CLUSTER_NUMBER:
                self.__createStandardLineplot(data, labels, xlim=xlim, ylim=ylim)
            case Metrics.CLUSTER_NUMBER_WITH_RADIUS:
                self.__createStandardLineplot(data, labels, xlim=xlim, ylim=ylim)
            case Metrics.CLUSTER_SIZE:
                self.__createStandardLineplot(data, labels, xlim=xlim, ylim=ylim)
            case Metrics.ORDER_VALUE_PERCENTAGE:
                if ylim == None:
                    ylim = (0, 100.1)
                self.__createStandardLineplot(data, labels, xlim=xlim, ylim=ylim)
            case Metrics.DUAL_OVERLAY_ORDER_AND_PERCENTAGE:
                if ylim == None:
                    ylim = (0, 1.1)
                self.__createDualOrderPlot(data, labels=labels,
                                           xlim=xlim, ylim=ylim)
            case Metrics.AVERAGE_NUMBER_NEIGHBOURS:
                self.__createStandardLineplot(data, labels, xlim=xlim, ylim=ylim)
            case Metrics.MIN_AVG_MAX_NUMBER_NEIGHBOURS:
                self.__createMinAvgMaxLinePlot(data, labels, xlim=xlim, ylim=ylim)
            case Metrics.AVG_DISTANCE_NEIGHBOURS:
                self.__createStandardLineplot(data, labels, xlim=xlim, ylim=ylim)
            case Metrics.AVG_CENTROID_DISTANCE:
                self.__createStandardLineplot(data, labels, xlim=xlim, ylim=ylim)

        ax = plt.gca()
        # reset axis to start at (0.0)
        xlim = ax.get_xlim()
        ax.set_xlim((0, xlim[1]))
        ylim = ax.get_ylim()
        ax.set_ylim((0, ylim[1]))

        if varianceData != None and self.metric != Metrics.DUAL_OVERLAY_ORDER_AND_PERCENTAGE:
            xlim = ax.get_xlim()
            x = np.arange(start=0, stop=len(varianceData[0]), step=1)
            for i in range(len(varianceData)):
                ax.fill_between(x, np.array(varianceData[i])[:,0], np.array(varianceData[i])[:,1], color=COLOURS[i], alpha=0.2)

        if xLabel != None:
            plt.xlabel(xLabel)
        if yLabel != None:
            plt.ylabel(yLabel)
        if subtitle != None:
            plt.title(f"""{subtitle}""")
        if not any(ele is None for ele in colourBackgroundForTimesteps):
            ax = plt.gca()
            ylim = ax.get_ylim()
            y = np.arange(ylim[0], ylim[1], 0.01)
            ax.fill_betweenx(y, colourBackgroundForTimesteps[0], colourBackgroundForTimesteps[1], facecolor='green', alpha=0.2)
        if savePath != None:
            plt.savefig(savePath)
        plt.show()
        plt.close()

    def evaluateAndVisualize(self, labels, xLabel=None, yLabel=None, subtitle=None, colourBackgroundForTimesteps=(None,None), showVariance=False, xlim=None, ylim=None, savePath=None):
        """
        Evaluates and subsequently visualises the results for multiple models.

        Parameters:
            - labels (array of strings): the label for each model
            - xLabel (string) [optional]: the label for the x-axis
            - yLabel (string) [optional]: the label for the y-axis
            - subtitle (string) [optional]: subtitle to be included in the title of the visualisation
            - colourBackgroundForTimesteps ([start, stop]) [optional]: the start and stop timestep for the background colouring for the event duration
            - showVariance (boolean) [optional]: whether the variance data should be added to the plot, by default False
            - xlim (float) [optional]: the x-limit for the plot
            - ylim (float) [optional]: the y-limit for the plot
            - savePath (string) [optional]: the location and name of the file where the model should be saved. Will not be saved unless a savePath is provided

        Returns:
            Nothing.
        """
        data, varianceData = self.evaluate()
        if showVariance == False:
            varianceData = None
        self.visualize(data, labels, xLabel=xLabel, yLabel=yLabel, subtitle=subtitle, colourBackgroundForTimesteps=colourBackgroundForTimesteps, varianceData=varianceData, xlim=xlim, ylim=ylim, savePath=savePath)
        
    def __createStandardLineplot(self, data, labels, xlim=None, ylim=None):
        """
        Creates a line plot for every model at every timestep

        Parameters:
            - data (dictionary): a dictionary with the time step as its key and a list of the number of clusters for every model as its value
            - labels (list of strings): labels for the models
            - xlim (float) [optional]: the x-limit for the plot
            - ylim (float) [optional]: the y-limit for the plot

        Returns:
            Nothing.
        """
        sorted(data.items())
        df = pd.DataFrame(data, index=labels).T

        if xlim != None and ylim != None:
            df.plot.line(xlim=xlim, ylim=ylim)
        elif xlim != None:
            df.plot.line(xlim=xlim)
        elif ylim != None:
            df.plot.line(ylim=ylim)
        else:
            df.plot.line()

    def __createDualOrderPlot(self, data, labels=None, xlim=None, ylim=None):
        """
        Creates a line plot overlaying the percentage of particles choosing the order switch type value and the order value. 

        Parameters:
            - data (dictionary): a dictionary with the time step as its key and a list of the number of clusters for every model as its value
            - labels (list of strings) [optional]: labels for the models
            - xlim (float) [optional]: the x-limit for the plot
            - ylim (float) [optional]: the y-limit for the plot
            
        Returns:
            Nothing.
        """
        if labels == None:
            labels = ["order", "percentage of order value"]

        sorted(data.items())
        df = pd.DataFrame(data, index=labels).T
        if xlim != None and ylim != None:
            df.plot.line(xlim=xlim, ylim=ylim)
        elif xlim != None:
            df.plot.line(xlim=xlim)
        elif ylim != None:
            df.plot.line(ylim=ylim)
        else:
            df.plot.line()

    def __createMinAvgMaxLinePlot(self, data, xlim=None, ylim=None):
        """
        Creates a line plot overlaying minimum, average and maximum number of neighbours.

        Parameters:
            - data (dictionary): a dictionary with the time step as its key and a list of the min, avg and max number of neighbours for every model as its value
            - xlim (float) [optional]: the x-limit for the plot
            - ylim (float) [optional]: the y-limit for the plot

        Returns:
            Nothing.
        """
        sorted(data.items())
        df = pd.DataFrame(data, index=["min", "avg", "max"]).T
        if xlim != None and ylim != None:
            df.plot.line(xlim=xlim, ylim=ylim)
        elif xlim != None:
            df.plot.line(xlim=xlim)
        elif ylim != None:
            df.plot.line(ylim=ylim)
        else:
            df.plot.line()

    def getMinAvgMaxNumberOfNeighboursOverWholeRun(self):
        """
        Computes the minimum, average and maximum number of neighbours over the course of the whole run. Therefore, the resulting data
        is not useful to plot the tendency over time. Rather, it shows the boundaries of the number of neighbours any particle can see
        during the whole simulation.

        Parameters:
            None

        Returns:
            Three numbers representing the minimum, average and maximum number of neighbours for all particles during the whole simulation.
        """
        self.metric = Metrics.MIN_AVG_MAX_NUMBER_NEIGHBOURS
        dataDict = self.evaluate()
        mins = []
        avgs = []
        maxs = []
        for val in dataDict.values():
            mins.append(val[0])
            avgs.append(val[1])
            maxs.append(val[2])
        return np.min(mins), np.average(avgs), np.max(maxs)
