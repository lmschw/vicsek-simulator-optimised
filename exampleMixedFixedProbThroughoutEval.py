import time
import numpy as np

from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from enums.EnumMetrics import Metrics
from enums.EnumSwitchType import SwitchType
from enums.EnumDistributionType import DistributionType
from enums.EnumEventEffect import EventEffect

from evaluators.EvaluatorMultiComp import EvaluatorMultiAvgComp
import services.ServiceSavedModel as ServiceSavedModel
import services.ServicePreparation as ServicePreparation
import services.ServiceGeneral as ServiceGeneral


dataLocation = "J:/stagger/"
saveLocation = "results_stagger/"
iStart = 1
iStop = 11

def eval(density, n, radius, eventEffect, metrics, type, nsm=None, combo=None, evalInterval=1, tmax=15000, duration=1000, noisePercentage=1, enforceSplit=True, percentageFirstValue=0, onlyInitial=True):

    startEval = time.time()
    ServiceGeneral.logWithTime(f"d={density}, r={radius}, nsm={nsm}, type={type}, enforceSplit={enforceSplit}") 

    modelParams = []
    simulationData = []
    colours = []
    switchTypes = []

    for initialStateString in ["ordered", "random"]:
        baseFilename = f"{dataLocation}local_nosw_noev_stagger=({percentageFirstValue},{ks},{enforceSplit},{onlyInitial})_d={density}_r={radius}_{initialStateString}_nsm={nsm.value}_ks=1-2_noise={noisePercentage}"

        filenames = ServiceGeneral.createListOfFilenamesForI(baseFilename=baseFilename, minI=iStart, maxI=iStop, fileTypeString="json")
        #filenames = [f"{name}.csv" for name in filenames]

        modelParamsDensity, simulationDataDensity = ServiceSavedModel.loadModels(filenames, loadColours=False, loadSwitchValues=False,  loadFromCsv=False)
        modelParams.append(modelParamsDensity)
        simulationData.append(simulationDataDensity)

#paths.append(f"density-vs-noise_ORDER_mode-comparision_n={n}_k=1_radius=10_density={density}_noise={noisePercentage}%_hierarchical_clustering_threshold=0.01.png")
#createMultiPlotFromImages(title, numX, numY, rowLabels, colLabels, paths)
    
    for metric in metrics:
        ServiceGeneral.logWithTime(f"Starting metric = {metric.val}")
        xlim = (0, tmax)
        threshold = 0.01
        if metric in [Metrics.ORDER, Metrics.DUAL_OVERLAY_ORDER_AND_PERCENTAGE]:
            ylim = (0, 1.1)
        elif metric == Metrics.CLUSTER_NUMBER_WITH_RADIUS:
            ylim = (0, n)
            threshold = 0.995
        else:
            ylim = (0, 50)
   
        yAxisLabel = metric.label
        threshold = 0.01
        
        evaluator = EvaluatorMultiAvgComp(modelParams=modelParams, metric=metric, simulationData=
                                          simulationData, evaluationTimestepInterval=evalInterval, threshold=threshold, switchType=None, switchTypeValues=switchTypes, switchTypeOptions=combo)
        
        labels = ["ordered", "random"]
        if metric == Metrics.DUAL_OVERLAY_ORDER_AND_PERCENTAGE:
            labels = ["ordered - order", "ordered - percentage of order-inducing value", "disordered - order", "disordered - percentage of order-inducing value"]
            labels = ["order", "percentage of order-inducing value"]
        savePath = f"{saveLocation}{metric.val}local_nosw_noev_stagger=({percentageFirstValue},{ks},{enforceSplit},{onlyInitial})_d={density}_r={radius}_nsm={nsm.value}_ks=1-2_noise={noisePercentage}.svg"

        evaluator.evaluateAndVisualize(labels=labels, xLabel=xAxisLabel, yLabel=yAxisLabel, colourBackgroundForTimesteps=[eventStart, eventStart+duration], showVariance=True, xlim=xlim, ylim=ylim, savePath=savePath)    
        endEval = time.time()
        print(f"Duration eval {ServiceGeneral.formatTime(endEval-startEval)}") 

def getLabelsFromNoisePercentages(noisePercentages):
    return [f"{noisePercentage}% noise" for noisePercentage in noisePercentages]

def getLabelsFromKValues(ks):
    return [f"k={k}" for k in ks]

def getLabelsFromNeighbourSelectionModes(neighbourSelectionModes):
    return [neighbourSelectionMode.name for neighbourSelectionMode in neighbourSelectionModes]

def getLabelsFromEventEffects(eventEffects):
    return [eventEffect.label for eventEffect in eventEffects]

xLabel = "time steps"
# GENERAL
domainSize = (50, 50)
noisePercentage = 1
noise = ServicePreparation.getNoiseAmplitudeValueForPercentage(noisePercentage)
speed = 1
initialAngleX = 0.5
initialAngleY = 0.5
colourType = None
degreesOfVision = 2 * np.pi

tmax = 15000

# SWITCHING
threshold = [0.1]
numberOfPreviousSteps = 100
updateIfNoNeighbours = False

# EVENT
eventStart = 5000
eventDuration = 100
eventDistributionType = DistributionType.LOCAL_SINGLE_SITE
eventAngle = np.pi
eventNumberAffected = None

# TEST VALS
nsms = [
        NeighbourSelectionMechanism.NEAREST,
        NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE]

nsmsReduced = [NeighbourSelectionMechanism.NEAREST,
               NeighbourSelectionMechanism.FARTHEST,
               NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE,
               NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE]

ks = [1,2]

eventEffects = [EventEffect.ALIGN_TO_FIXED_ANGLE,
                EventEffect.AWAY_FROM_ORIGIN,
                EventEffect.RANDOM]

nsmCombos = [[NeighbourSelectionMechanism.FARTHEST, NeighbourSelectionMechanism.NEAREST],
             [NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE, NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE]]

kCombos = [[1,2]]

densities = [0.06]
radii = [10]
initialConditions = ["ordered", "random"]

evaluationInterval = 1

# K VS. START
metrics = [
           Metrics.ORDER
           ]
xAxisLabel = "timesteps"

noisePercentages = [1, 2, 3, 4, 5]

percentageFirstValue = 0

startTime = time.time()
startNoswnoev = time.time()
ServiceGeneral.logWithTime("Starting eval for nosw noev")
for noisePercentage in noisePercentages:
    noise = ServicePreparation.getNoiseAmplitudeValueForPercentage(noisePercentage)
    # ------------------------ FIXED STRATEGIES ---------------------------------
    enforceSplit = False
    for density in densities:
        n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
        for radius in radii:
            for nsm in nsms:
                eval(density=density, n=n, radius=radius, eventEffect=None, metrics=metrics, type="enfsplit", nsm=nsm,  
                    combo=None, evalInterval=evaluationInterval, tmax=tmax, noisePercentage=noisePercentage, percentageFirstValue=percentageFirstValue, enforceSplit=enforceSplit, onlyInitial=False)
endNoswnoev = time.time()
ServiceGeneral.logWithTime(f"Completed eval for enforceSplit={enforceSplit} in {ServiceGeneral.formatTime(endNoswnoev-startNoswnoev)}")

endTime = time.time()
print(f"Total duration: {ServiceGeneral.formatTime(endTime-startTime)}")
    

