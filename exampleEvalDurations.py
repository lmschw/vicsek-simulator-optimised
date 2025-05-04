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


dataLocation = "J:/duration_tests/"
saveLocation = "results_durations/"
iStart = 1
iStop = 11

def eval(density, n, radius, eventEffect, metrics, type, nsm=None, k=None, combo=None, evalInterval=1, tmax=15000, duration=1000, noisePercentage=1):

    startEval = time.time()
    if type in ["noswnoev", "nsmswnoev", "kswnoev"]:
        ServiceGeneral.logWithTime(f"d={density}, r={radius}, nsm={nsm}, k={k}, type={type}") 
    else:
        ServiceGeneral.logWithTime(f"d={density}, r={radius}, nsm={nsm}, k={k}, combo={combo}, eventEffect={eventEffect.val}, type={type}")
    modelParams = []
    simulationData = []
    colours = []
    switchTypes = []

    base_filenames = []
    for initialStateString in ["ordered", "random"]:
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

        if type == "noswnoev":
            baseFilename = f"{dataLocation}global_noev_nosw_{initialStateString}_st={nsm.value}_d={density}_n={n}_r={radius}_tmax={tmax}_k={k}_noise={noisePercentage}"
            sTypes = []
        elif type == "nosw":
            baseFilename = f"{dataLocation}local_nosw_1ev_d={density}_r={radius}_{initialStateString}_nsm={nsm.value}_k={k}_ee={eventEffect.val}"
            sTypes = []
        elif type == "nsmswnoev":
            baseFilename = f"{dataLocation}local_nsmsw_noev_d={density}_r={radius}_{initialStateString}_st={nsm.value}_nsmCombo={combo[0].value}-{combo[1].value}_k={k}"
            sTypes = [SwitchType.NEIGHBOUR_SELECTION_MECHANISM]
        elif type == "nsmsw":
            baseFilename = f"{dataLocation}local_1e_switchType=MODE_{initialStateString}_st={nsm}_o={orderValue.value}_do={disorderValue.value}_d={density}_n={n}_r={radius}_k={k}_noise={noisePercentage}_ee={eventEffect.val}_duration={duration}"
            sTypes = [SwitchType.NEIGHBOUR_SELECTION_MECHANISM]
        elif type == "kswnoev":
            baseFilename = f"{dataLocation}local_ksw_noev_d={density}_r={radius}_{initialStateString}_st={k}_nsm={nsm.value}_kCombo={combo[0]}-{combo[1]}"
            sTypes = [SwitchType.K]
        elif type == "ksw":
            baseFilename = f"{dataLocation}local_1e_switchType=K_{initialStateString}_st={k}_o={orderValue}_do={disorderValue}_d={density}_n={n}_r={radius}_nsm={nsm.value}_noise=1_ee={eventEffect.value}_duration={duration}"
            sTypes = [SwitchType.K]
        base_filenames.append(baseFilename)

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

        evaluator = EvaluatorMultiAvgComp(metric=metric,
                                          basePaths=base_filenames,
                                          runRange=(iStart, iStop),
                                          from_csv=True,
                                          evaluationTimestepInterval=evalInterval,
                                          threshold=threshold,
                                          switchType=sTypes,
                                          switchTypeOptions=combo)
        
        labels = ["ordered", "random"]

        if type == "ksw":
            savePath = f"{saveLocation}{metric.val}_local_1e_switchType=K_{initialStateString}_st={k}_o={orderValue}_do={disorderValue}_d={density}_n={n}_r={radius}_nsm={nsm.value}_noise=1_ee={eventEffect.val}_duration={duration}.svg"
        elif type == "nsmsw":
            savePath = f"{saveLocation}{metric.val}_local_1e_switchType=MODE_{initialStateString}_st={nsm}_o={orderValue.value}_do={disorderValue.value}_d={density}_n={n}_r={radius}_k={k}_noise={noisePercentage}_ee={eventEffect.val}_duration={duration}.svg"

        evaluator.evaluateAndVisualize(labels=labels, xLabel=xAxisLabel, yLabel=yAxisLabel, colourBackgroundForTimesteps=[eventStart, eventStart+duration], showVariance=True, xlim=xlim, ylim=ylim, savePath=savePath)    
        endEval = time.time()
        print(f"Duration eval {ServiceGeneral.formatTime(endEval-startEval)}") 

def getLabelsFromNoisePercentages(noisePercentages):
    return [f"{noisePercentage}% noise" for noisePercentage in noisePercentages]

def getLabelsFromKValues(ks):
    return [f"k={k}" for k in ks]

def getLabelsFromNeighbourSelectionMechanisms(NeighbourSelectionMechanisms):
    return [NeighbourSelectionMechanism.name for NeighbourSelectionMechanism in NeighbourSelectionMechanisms]

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

ks = [1]

eventEffects = [EventEffect.ALIGN_TO_FIXED_ANGLE,
                EventEffect.AWAY_FROM_ORIGIN,
                EventEffect.RANDOM]

nsmCombos = [[NeighbourSelectionMechanism.FARTHEST, NeighbourSelectionMechanism.NEAREST],
             [NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE, NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE]]

kCombos = [[1,5]]

densities = [0.09]
radii = [10]
initialConditions = ["ordered", "random"]

evaluationInterval = 1

# K VS. START
metrics = [
           Metrics.ORDER
           ]
xAxisLabel = "timesteps"

noisePercentage = 1

durations = [1, 2, 5, 10, 50, 100, 200, 500, 1000]

eventEffects = [EventEffect.ALIGN_TO_FIXED_ANGLE,
                EventEffect.AWAY_FROM_ORIGIN,
                EventEffect.RANDOM]

startTime = time.time()
startNoswnoev = time.time()
combo = [5,1]
ServiceGeneral.logWithTime("Starting eval for nosw noev")
for density in densities:
    n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
    for radius in radii:
        for duration in durations:
            for nsm in [NeighbourSelectionMechanism.NEAREST,
                        NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE]:
                for eventEffect in eventEffects:
                    eval(density=density, n=n, radius=radius, eventEffect=eventEffect, metrics=metrics, type="ksw", nsm=nsm, k=1, 
                        combo=combo, evalInterval=evaluationInterval, tmax=tmax, noisePercentage=noisePercentage, duration=duration)
endNoswnoev = time.time()
ServiceGeneral.logWithTime(f"Completed eval for nosw noev in {ServiceGeneral.formatTime(endNoswnoev-startNoswnoev)}")

startNoswnoev = time.time()
ServiceGeneral.logWithTime("Starting eval for nosw noev")
for density in densities:
    n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
    for radius in radii:
        for duration in durations:
            for nsmCombo in [[NeighbourSelectionMechanism.FARTHEST, NeighbourSelectionMechanism.NEAREST],
                        [NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE, NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE]]:
                for eventEffect in eventEffects:
                    eval(density=density, n=n, radius=radius, eventEffect=eventEffect, metrics=metrics, type="nsmsw", nsm=None, k=1, 
                        combo=nsmCombo, evalInterval=evaluationInterval, tmax=tmax, noisePercentage=noisePercentage, duration=duration)
endNoswnoev = time.time()
ServiceGeneral.logWithTime(f"Completed eval for nsm in {ServiceGeneral.formatTime(endNoswnoev-startNoswnoev)}")

endTime = time.time()
print(f"Total duration: {ServiceGeneral.formatTime(endTime-startTime)}")
    

