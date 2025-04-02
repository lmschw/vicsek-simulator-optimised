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

def eval(density, n, radius, eventEffect, metrics, type, nsm=None, k=None, combo=None, evalInterval=1, tmax=15000, duration=1000, noisePercentage=1, enforceSplit=False):

    startEval = time.time()
    if type in ["noswnoev", "nsmswnoev", "kswnoev", "stagger"]:
        ServiceGeneral.logWithTime(f"d={density}, r={radius}, nsm={nsm}, k={k}, type={type}") 
    else:
        ServiceGeneral.logWithTime(f"d={density}, r={radius}, nsm={nsm}, k={k}, combo={combo}, eventEffect={eventEffect.val}, type={type}")
    modelParams = []
    simulationData = []
    colours = []
    switchTypes = []

    for initialStateString in ["ordered"]:
        if type in ["nsmswnoev", "nsmsw", "kswnoev", "ksw"]:
            orderValue, disorderValue = combo
        if type in ["nsmswnoev", "nsmsw"]: 
            if initialStateString == "ordered":
                nsm = orderValue
            else:
                nsm = disorderValue
        elif type in ["kswnoev", "ksw"]: 
            if initialStateString == "ordered":
                k = orderValue
            else:
                k = disorderValue

        if type == "stagger":
            baseFilename = f"{dataLocation}local_nosw_noev_stagger=(0.5,[1, 2],{enforceSplit})_d={density}_r={radius}_{initialStateString}_nsm={nsm.value}_ks=1-2_noise={noisePercentage}"
            sTypes = []


        
        filenames = ServiceGeneral.createListOfFilenamesForI(baseFilename=baseFilename, minI=iStart, maxI=iStop, fileTypeString="json")
        #filenames = [f"{name}.csv" for name in filenames]
        if type not in ["nosw", "noswnoev", "stagger"]:
            modelParamsDensity, simulationDataDensity, switchTypeValues = ServiceSavedModel.loadModels(filenames, loadColours=False, loadSwitchValues=True, switchTypes=sTypes, loadFromCsv=False)
            switchTypes.append([switchTypeValues[0][sTypes[0].switchTypeValueKey]])
        else:
            modelParamsDensity, simulationDataDensity = ServiceSavedModel.loadModels(filenames, loadColours=False, loadSwitchValues=False, switchTypes=sTypes, loadFromCsv=False)
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

        if len(sTypes) > 0:
            sType = sTypes[0]
        else:
            sType = None
        
        evaluator = EvaluatorMultiAvgComp(modelParams, metric, simulationData, evaluationTimestepInterval=evalInterval, threshold=threshold, switchType=sType, switchTypeValues=switchTypes, switchTypeOptions=combo)
        
        labels = ["ordered"]
        if metric == Metrics.DUAL_OVERLAY_ORDER_AND_PERCENTAGE:
            labels = ["ordered - order", "ordered - percentage of order-inducing value", "disordered - order", "disordered - percentage of order-inducing value"]
            labels = ["order", "percentage of order-inducing value"]
        if type == "stagger":
            savePath = f"{saveLocation}{metric.val}_local_nosw_noev_stagger=(0.5,[1, 2],{enforceSplit})_d={density}_r={radius}_nsm={nsm.value}_ks=1-2_noise={noisePercentage}.svg"

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
nsms = [NeighbourSelectionMechanism.ALL,
        NeighbourSelectionMechanism.RANDOM,
        NeighbourSelectionMechanism.NEAREST,
        NeighbourSelectionMechanism.FARTHEST,
        NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE,
        NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE]

nsmsReduced = [NeighbourSelectionMechanism.NEAREST,
               NeighbourSelectionMechanism.FARTHEST,
               NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE,
               NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE]

ks = [1,5]

eventEffects = [EventEffect.ALIGN_TO_FIXED_ANGLE,
                EventEffect.AWAY_FROM_ORIGIN,
                EventEffect.RANDOM]

nsmCombos = [[NeighbourSelectionMechanism.FARTHEST, NeighbourSelectionMechanism.NEAREST],
             [NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE, NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE]]

kCombos = [[1,5]]

densities = [0.06]
radii = [10]
initialConditions = ["ordered", "random"]

evaluationInterval = 1

# K VS. START
metrics = [
           Metrics.ORDER
           ]
xAxisLabel = "timesteps"

noisePercentages = [1,2,3,4,5]

startTime = time.time()
startNoswnoev = time.time()
ServiceGeneral.logWithTime("Starting eval for nosw noev")
for density in densities:
    n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
    for radius in radii:
        for noisePercentage in noisePercentages:
            for enforceSplit in [True, False]:
                for nsm in nsms:
                    eval(density=density, n=n, radius=radius, eventEffect=None, metrics=metrics, type="stagger", nsm=nsm, k=1, 
                        combo=None, evalInterval=evaluationInterval, tmax=tmax, noisePercentage=noisePercentage, enforceSplit=enforceSplit)
endNoswnoev = time.time()
ServiceGeneral.logWithTime(f"Completed eval for nosw noev in {ServiceGeneral.formatTime(endNoswnoev-startNoswnoev)}")

endTime = time.time()
print(f"Total duration: {ServiceGeneral.formatTime(endTime-startTime)}")
    

