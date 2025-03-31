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


dataLocation = "J:/data4/"
saveLocation = "results_ns/"
iStart = 1
iStop = 11

def eval(density, n, radius, eventEffect, metrics, type, nsm=None, k=None, combo=None, evalInterval=1, tmax=15000, duration=1000):

    startEval = time.time()
    if type in ["noswnoev", "nsmswnoev", "kswnoev"]:
        ServiceGeneral.logWithTime(f"d={density}, r={radius}, nsm={nsm}, k={k}, type={type}") 
    else:
        ServiceGeneral.logWithTime(f"d={density}, r={radius}, nsm={nsm}, k={k}, combo={combo}, eventEffect={eventEffect.val}, type={type}")
    modelParams = []
    simulationData = []
    colours = []
    switchTypes = []

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
            baseFilename = f"{dataLocation}local_nosw_noev_d={density}_r={radius}_{initialStateString}_nsm={nsm.value}_k={k}"
            sTypes = []
        elif type == "nosw":
            baseFilename = f"{dataLocation}local_nosw_1ev_d={density}_r={radius}_{initialStateString}_nsm={nsm.value}_k={k}_ee={eventEffect.val}"
            sTypes = []
        elif type == "nsmswnoev":
            baseFilename = f"{dataLocation}local_nsmsw_noev_d={density}_r={radius}_{initialStateString}_st={nsm.value}_nsmCombo={combo[0].value}-{combo[1].value}_k={k}"
            sTypes = [SwitchType.NEIGHBOUR_SELECTION_MECHANISM]
        elif type == "nsmsw":
            baseFilename = f"{dataLocation}local_nsmsw_1ev_d={density}_r={radius}_{initialStateString}_st={nsm.value}_nsmCombo={combo[0].value}-{combo[1].value}_k={k}_ee={eventEffect.val}"
            sTypes = [SwitchType.NEIGHBOUR_SELECTION_MECHANISM]
        elif type == "kswnoev":
            baseFilename = f"{dataLocation}local_ksw_noev_d={density}_r={radius}_{initialStateString}_st={k}_nsm={nsm.value}_kCombo={combo[0]}-{combo[1]}"
            sTypes = [SwitchType.K]
        elif type == "ksw":
            baseFilename = f"{dataLocation}local_ksw_1ev_d={density}_r={radius}_{initialStateString}_st={k}_nsm={nsm.value}_kCombo={kCombo[0]}-{kCombo[1]}_ee={eventEffect.val}"
            sTypes = [SwitchType.K]

        
        filenames = ServiceGeneral.createListOfFilenamesForI(baseFilename=baseFilename, minI=iStart, maxI=iStop, fileTypeString="json")
        #filenames = [f"{name}.csv" for name in filenames]
        if type not in ["nosw", "noswnoev"]:
            modelParamsDensity, simulationDataDensity, switchTypeValues = ServiceSavedModel.loadModels(filenames, loadColours=False, loadSwitchValues=True, switchTypes=sTypes, loadFromCsv=False)
            switchTypes.append(switchTypeValues[sTypes[0].switchTypeValueKey])
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
        if type == "noswnoev":
            savePath = f"{saveLocation}{metric.val}_local_nosw_noev_d={density}_r={radius}_nsm={nsm.value}_k={k}.svg"
        elif type == "nosw":
            savePath = f"{saveLocation}{metric.val}_local_nosw_1ev_d={density}_r={radius}_nsm={nsm.value}_k={k}_ee={eventEffect.val}.svg"
        elif type == "nsmswnoev":
            savePath = f"{saveLocation}{metric.val}_local_nsmsw_noev_d={density}_r={radius}_st={nsm.value}_nsmCombo={combo[0].value}-{combo[1].value}_k={k}.svg"
        elif type == "nsmsw":
            savePath = f"{saveLocation}{metric.val}_local_nsmsw_1ev_d={density}_r={radius}_st={nsm.value}_nsmCombo={combo[0].value}-{combo[1].value}_k={k}_ee={eventEffect.val}.svg"
        elif type == "kswnoev":
            savePath = f"{saveLocation}{metric.val}_local_ksw_noev_d={density}_r={radius}_st={k}_nsm={nsm.value}_kCombo={combo[0]}-{combo[1]}.svg"
        elif type == "ksw":
            savePath = f"{saveLocation}{metric.val}_local_ksw_1ev_d={density}_r={radius}_st={k}_nsm={nsm.value}_kCombo={kCombo[0]}-{kCombo[1]}_ee={eventEffect.val}.svg"

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


startTime = time.time()

startNoswnoev = time.time()
ServiceGeneral.logWithTime("Starting eval for nosw noev")
for density in densities:
    n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
    for radius in radii:
        for nsm in nsms:
            for k in ks:
                eval(density=density, n=n, radius=radius, eventEffect=None, metrics=metrics, type="noswnoev", nsm=nsm, k=k, 
                     combo=None, evalInterval=evaluationInterval, tmax=tmax)
endNoswnoev = time.time()
ServiceGeneral.logWithTime(f"Completed eval for nosw noev in {ServiceGeneral.formatTime(endNoswnoev-startNoswnoev)}")

startNosw = time.time()
ServiceGeneral.logWithTime("Starting eval for nosw")
for density in densities:
    n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
    for radius in radii:
        for nsm in nsmsReduced:
            for k in ks:
                for eventEffect in eventEffects:
                    eval(density=density, n=n, radius=radius, eventEffect=eventEffect, metrics=metrics, type="nosw", nsm=nsm, k=k, 
                        combo=None, evalInterval=evaluationInterval, tmax=tmax)
endNosw = time.time()
ServiceGeneral.logWithTime(f"Completed eval for nosw in {ServiceGeneral.formatTime(endNosw-startNosw)}")

startNsmswnoev = time.time()
ServiceGeneral.logWithTime("Starting eval for nsmsw noev")
for density in densities:
    n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
    for radius in radii:
        for nsmCombo in nsmCombos:
            for k in ks:
                eval(density=density, n=n, radius=radius, eventEffect=None, metrics=metrics, type="nsmswnoev", nsm=None, k=k, 
                     combo=nsmCombo, evalInterval=evaluationInterval, tmax=tmax)
endNsmswnoev = time.time()
ServiceGeneral.logWithTime(f"Completed eval for nsmsw noev in {ServiceGeneral.formatTime(endNsmswnoev-startNsmswnoev)}")

startNsmsw = time.time()
ServiceGeneral.logWithTime("Starting eval for nsmsw")
for density in densities:
    n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
    for radius in radii:
        for nsmCombo in nsmCombos:
            for k in ks:
                for eventEffect in eventEffects:
                    eval(density=density, n=n, radius=radius, eventEffect=eventEffect, metrics=metrics, type="nsmsw", nsm=None, k=k, 
                        combo=nsmCombo, evalInterval=evaluationInterval, tmax=tmax)
endNsmsw = time.time()
ServiceGeneral.logWithTime(f"Completed eval for nsmsw in {ServiceGeneral.formatTime(endNsmsw-startNsmsw)}")

startKswnoev = time.time()
ServiceGeneral.logWithTime("Starting eval for ksw noev")
for density in densities:
    n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
    for radius in radii:
        for nsm in nsmsReduced:
            for kCombo in kCombos:
                eval(density=density, n=n, radius=radius, eventEffect=None, metrics=metrics, type="kswnoev", nsm=nsm, k=None, 
                     combo=kCombo, evalInterval=evaluationInterval, tmax=tmax)
endKswnoev = time.time()
ServiceGeneral.logWithTime(f"Completed eval for ksw noev in {ServiceGeneral.formatTime(endKswnoev-startKswnoev)}")

startKsw = time.time()
ServiceGeneral.logWithTime("Starting eval for ksw")
for density in densities:
    n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
    for radius in radii:
        for nsm in nsmsReduced:
            for kCombo in kCombos:
                for eventEffect in eventEffects:
                    eval(density=density, n=n, radius=radius, eventEffect=eventEffect, metrics=metrics, type="ksw", nsm=nsm, k=None, 
                        combo=kCombo, evalInterval=evaluationInterval, tmax=tmax)
endKsw = time.time()
ServiceGeneral.logWithTime(f"Completed eval for ksw in {ServiceGeneral.formatTime(endKsw-startKsw)}")

endTime = time.time()
print(f"Total duration: {ServiceGeneral.formatTime(endTime-startTime)}")
    

