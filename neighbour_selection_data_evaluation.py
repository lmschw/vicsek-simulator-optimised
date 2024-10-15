import time
import numpy as np

from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from enums.EnumMetrics import Metrics
from enums.EnumSwitchType import SwitchType
from enums.EnumDistributionType import DistributionType
from enums.EnumEventEffect import EventEffect

import evaluators.EvaluatorMultiComp as EvaluatorMultiComp
import services.ServiceSavedModel as ServiceSavedModel
import services.ServicePreparation as ServicePreparation
import services.ServiceGeneral as ServiceGeneral
import services.ServiceMetric as ServiceMetric

import DefaultValues as dv
import animator.AnimatorMatplotlib as AnimatorMatplotlib
import animator.Animator2D as Animator2D


def eval(density, n, radius, eventEffect, metric, type, nsm=None, k=None, combo=None, evalInterval=1, tmax=15000):
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
    startEval = time.time()
    if type == "global":
        ServiceGeneral.logWithTime(f"d={density}, r={radius}, nsm={nsm}, k={k}, metric={metric.name}, type={type}") 
    else:
        ServiceGeneral.logWithTime(f"d={density}, r={radius}, nsm={nsm}, k={k}, combo={combo}, eventEffect={eventEffect.val}, metric={metric.name}, type={type}")
    modelParams = []
    simulationData = []
    switchTypes = []

    for initialStateString in ["ordered", "random"]:
        if type in ["nsmsw", "ksw"]:
            orderValue, disorderValue = combo
        if type == "nsmsw": 
            if initialStateString == "ordered":
                nsm = orderValue
            else:
                nsm = disorderValue
        elif type == "ksw": 
            if initialStateString == "ordered":
                k = orderValue
            else:
                k = disorderValue
        
        if type == "nosw":
            baseFilename = f"{baseDataLocation}local_nosw_1ev_{initialStateString}_d={density}_n={n}_r={radius}_nsm={nsm.value}_k={k}_ee={eventEffect.val}"
        elif type == "nsmsw":
            baseFilename = f"{baseDataLocation}local_nsmsw_1ev_{initialStateString}_st={nsm.value}_d={density}_n={n}_r={radius}_nsmCombo={combo[0].value}-{combo[1]}_k={k}"
        elif type == "ksw":
            baseFilename = f"{baseDataLocation}local_ksw_1ev_{initialStateString}_st={k}_d={density}_n={n}_r={radius}_nsm={nsm.value}_kCombo={combo[0]}-{combo[1]}_ee={eventEffect.val}"
        elif type == "global":
            baseFilename = f"{baseDataLocation}global_nosw_noev_{initialStateString}_d={density}_n={n}_r={radius}_nsm={nsm.value}_k={k}"
        
        filenames = ServiceGeneral.createListOfFilenamesForI(baseFilename=baseFilename, minI=iStart, maxI=iStop, fileTypeString="json")
        if type not in ["nosw", "global"]:
            modelParamsDensity, simulationDataDensity, switchTypeValues = ServiceSavedModel.loadModels(filenames, loadSwitchValues=True)
            switchTypes.append(switchTypeValues)
        else:
            modelParamsDensity, simulationDataDensity = ServiceSavedModel.loadModels(filenames, loadSwitchValues=False)
        modelParams.append(modelParamsDensity)
        simulationData.append(simulationDataDensity)

#paths.append(f"density-vs-noise_ORDER_mode-comparision_n={n}_k=1_radius=10_density={density}_noise={noisePercentage}%_hierarchical_clustering_threshold=0.01.png")
#createMultiPlotFromImages(title, numX, numY, rowLabels, colLabels, paths)
    threshold = 0.01
    evaluator = EvaluatorMultiComp.EvaluatorMultiAvgComp(modelParams, metric, simulationData, evaluationTimestepInterval=evalInterval, threshold=threshold, switchTypeValues=switchTypes, switchTypeOptions=combo)
    
    labels = ["ordered", "disordered"]
    if metric == Metrics.DUAL_OVERLAY_ORDER_AND_PERCENTAGE:
        labels = ["ordered - order", "ordered - percentage of order-inducing value", "disordered - order", "disordered - percentage of order-inducing value"]
    if type == "nosw":
        savePath = f"{metric.val}_d={density}_n={n}_r={radius}_nosw_nsm={nsm.value}_k={k}_ee={eventEffect.val}.jpeg"
    elif type == "nsmsw":
        savePath = f"{metric.val}_d={density}_n={n}_r={radius}_swt=MODE_o={orderValue.value}_do={disorderValue.value}_k={k}_ee={eventEffect.val}.jpeg"
    elif type == "ksw":
        savePath = f"{metric.val}_d={density}_n={n}_r={radius}_swt=K_o={orderValue}_do={disorderValue}_nsm={nsm.value}_ee={eventEffect.val}.jpeg"
    elif type == "global":
        savePath = f"{metric.val}_d={density}_n={n}_r={radius}_global_nsm={nsm.value}_k={k}_th={threshold}.jpeg"

    evaluator.evaluateAndVisualize(labels=labels, xLabel=xAxisLabel, yLabel=yAxisLabel, colourBackgroundForTimesteps=[e1Start, e1Start+duration], showVariance=True, xlim=xlim, ylim=ylim, savePath=savePath)    
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

def getOrderDisorderValue(switchType):
    match switchType:
        case SwitchType.K:
            return 5, 1
        case SwitchType.NEIGHBOUR_SELECTION_MODE:
            return NeighbourSelectionMechanism.FARTHEST, NeighbourSelectionMechanism.NEAREST


xLabel = "time steps"

speed = 1

angle = np.pi
blockSteps = -1
threshold = [0.1]


#domainSize = ServicePreparation.getDomainSizeForConstantDensity(0.09, 100)
domainSize = (50, 50)

distTypeString = "lssmid"
distributionType = DistributionType.LOCAL_SINGLE_SITE
percentage = 100
e1Start = 5000
e2Start = 10000
e3Start = 15000

noisePercentages = [1] # to run again with other noise percentages, make sure to comment out anything that has fixed noise (esp. local)
densities = [0.01]
psteps = 100
numbersOfPreviousSteps = [psteps]
durations = [1000]
ks = [1,5]

neighbourSelectionMechanisms = [
                           NeighbourSelectionMechanism.NEAREST,
                           NeighbourSelectionMechanism.FARTHEST,
                           NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE,
                           NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE,
                           NeighbourSelectionMechanism.ALL,
                           NeighbourSelectionMechanism.RANDOM
                           ]

orderNeighbourSelectionMechanisms = [NeighbourSelectionMechanism.ALL,
                                NeighbourSelectionMechanism.RANDOM,
                                NeighbourSelectionMechanism.FARTHEST,
                                NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE]

disorderNeighbourSelectionMechanisms = [NeighbourSelectionMechanism.NEAREST,
                                   NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE]

localNeighbourSelectionMechanisms = [NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE,
                                NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE]

eventEffects = [EventEffect.ALIGN_TO_FIXED_ANGLE,
                EventEffect.AWAY_FROM_ORIGIN,
                EventEffect.RANDOM]

eventEffectsOrder = [
                     EventEffect.ALIGN_TO_FIXED_ANGLE,
                     ]

eventEffectsDisorder = [EventEffect.AWAY_FROM_ORIGIN,
                        EventEffect.RANDOM]

saveLocation = f""
iStart = 1
iStop = 2

baseDataLocation = ""

densities = [0.01, 0.09]
radii = [5, 20]
interval = 1
kMax = 5
noisePercentage = 1

# ------------------------------------------------ LOCAL ---------------------------------------------------------------
levelDataLocation = "local/switchingActive/"

data = {}

ks = [1, 5]

# K VS. START
metrics = [
           Metrics.ORDER
           ]
xAxisLabel = "timesteps"


startTime = time.time()

duration = 1000

for density in densities:
    n = int(ServicePreparation.getNumberOfParticlesForConstantDensity(density, domainSize))
    for radius in radii:
        tmax = 3000
        iStop = 2
        for nsm in neighbourSelectionMechanisms:
            for k in ks:
                    for metric in metrics:
                        eval(density=density, n=n, radius=radius, eventEffect=None, metric=metric, type="global", nsm=nsm, k=k, evalInterval=interval, tmax=tmax)

"""        
        tmax = 15000
        iStop = 2
        for nsm in neighbourSelectionMechanisms:
            for k in ks:
                for eventEffect in eventEffects:
                    for metric in metrics:
                        eval(density=density, n=n, radius=radius, eventEffect=eventEffect, metric=metric, type="nosw", nsm=nsm, k=k, evalInterval=interval, tmax=tmax)

        
        for nsmCombo in [[NeighbourSelectionMechanism.FARTHEST, NeighbourSelectionMechanism.NEAREST],
                         [NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE, NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE]]:
            for k in ks:
                for eventEffect in eventEffects:
                    for metric in metrics:
                        eval(density=density, n=n, radius=radius, eventEffect=eventEffect, metric=metric, type="nsmsw", k=k, combo=nsmCombo, evalInterval=interval, tmax=tmax)
        
        for nsm in NeighbourSelectionMechanisms:
            for kCombo in [[5,1]]:
                for eventEffect in eventEffects:
                    for metric in metrics:
                        eval(density=density, n=n, radius=radius, eventEffect=eventEffect, metric=metric, type="ksw", nsm=nsm, combo=kCombo, evalInterval=interval, tmax=tmax)
 """       
endTime = time.time()
print(f"Total duration: {ServiceGeneral.formatTime(endTime-startTime)}")
    

