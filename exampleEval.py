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


def eval(density, n, radius, eventEffect, metrics, type, nsm=None, k=None, combo=None, evalInterval=1, tmax=15000, duration=1000):

    startEval = time.time()
    if type == "global":
        ServiceGeneral.logWithTime(f"d={density}, r={radius}, nsm={nsm}, k={k}, type={type}") 
    else:
        ServiceGeneral.logWithTime(f"d={density}, r={radius}, nsm={nsm}, k={k}, combo={combo}, eventEffect={eventEffect.val}, type={type}")
    modelParams = []
    simulationData = []
    colours = []
    switchTypes = []

    for initialStateString in ["ordered"]:
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
            baseFilename = f"local_1e_nosw_{initialStateString}_st={nsm.value}__d={density}_n={n}_r={radius}_k={k}_noise=1_drn={duration}_{e1Start}-{eventEffect.val}"
            sTypes = []
        elif type == "nsmsw":
            baseFilename = f"local_1e_switchType=MODE_{initialStateString}_st={nsm.value}_o={orderValue.value}_do={disorderValue.value}_d={density}_n={n}_r={radius}_k={k}_noise={noisePercentage}_drn={duration}_{e1Start}-{eventEffect.val}"
            sTypes = [SwitchType.NEIGHBOUR_SELECTION_MECHANISM]
        elif type == "ksw":
            baseFilename = f"c:/Users/lschw/dev/vicsek-simulator-optimised/vicsek-simulator-optimised/test_info_ordered_predator_tmax={tmax}"
            sTypes = [SwitchType.K]
        elif type == "global":
            baseFilename = f"c:/Users/lschw/Downloads/OneDrive_1_21.3.2025/global_noev_nosw_random_st=HOD_d=0.06_n=150_r=20_tmax=10000000_k=1_noise=1"
            baseFilename = f"C:/Users/lschw/dev/vicsek-simulator-optimised/vicsek-simulator-optimised/test_hod_10000000_random"
            sTypes = []
        
        filenames = ServiceGeneral.createListOfFilenamesForI(baseFilename=baseFilename, minI=iStart, maxI=iStop, fileTypeString="csv")
        #filenames = [f"{name}.csv" for name in filenames]
        if type not in ["nosw", "global"]:
            modelParamsDensity, simulationDataDensity, switchTypeValues, coloursDensity = ServiceSavedModel.loadModels(filenames, loadColours=True, loadSwitchValues=True, switchTypes=sTypes, loadFromCsv=True)
            switchTypes.append([switchTypeValues[0][sTypes[0].switchTypeValueKey]])
        else:
            modelParamsDensity, simulationDataDensity, coloursDensity = ServiceSavedModel.loadModels(filenames, loadColours=True, loadSwitchValues=False, switchTypes=sTypes, loadFromCsv=True)
        modelParams.append(modelParamsDensity)
        simulationData.append(simulationDataDensity)
        colours.append(coloursDensity)

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
        if type == "nosw":
            savePath = f"{metric.val}_d={density}_n={n}_r={radius}_nosw_nsm={nsm.value}_k={k}_ee={eventEffect.val}.svg"
        elif type == "nsmsw":
            savePath = f"{metric.val}_d={density}_n={n}_r={radius}_swt=MODE_o={orderValue.value}_do={disorderValue.value}_k={k}_ee={eventEffect.val}.svg"
        elif type == "ksw":
            savePath = f"results_27032025/{metric.val}_d={density}_n={n}_r={radius}_swt=K_o={orderValue}_do={disorderValue}_nsm={nsm.value}_ee={eventEffect.val}.svg"
        elif type == "global":
            savePath = f"{metric.val}_random_d={density}_n={n}_r={radius}_global_nsm={nsm.value}_k={k}_th={threshold}_{interval}_new.svg"

        evaluator.evaluateAndVisualize(labels=labels, xLabel=xAxisLabel, yLabel=yAxisLabel, colourBackgroundForTimesteps=[e1Start, e1Start+duration], showVariance=True, xlim=xlim, ylim=ylim, savePath=savePath)    
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

def getOrderDisorderValue(switchType):
    match switchType:
        case SwitchType.K:
            return 5, 1
        case SwitchType.NEIGHBOUR_SELECTION_MECHANISM:
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
e1Start = 1000
e2Start = 10000
e3Start = 15000

noisePercentages = [1] # to run again with other noise percentages, make sure to comment out anything that has fixed noise (esp. local)
densities = [0.01]
psteps = 100
numbersOfPreviousSteps = [psteps]
durations = [1000]
ks = [1,5]

neighbourSelectionModes = [
                           NeighbourSelectionMechanism.NEAREST,
                           NeighbourSelectionMechanism.FARTHEST,
                           NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE,
                           NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE,
                           NeighbourSelectionMechanism.ALL,
                           NeighbourSelectionMechanism.RANDOM
                           ]

orderNeighbourSelectionModes = [NeighbourSelectionMechanism.ALL,
                                NeighbourSelectionMechanism.RANDOM,
                                NeighbourSelectionMechanism.FARTHEST,
                                NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE]

disorderNeighbourSelectionModes = [NeighbourSelectionMechanism.NEAREST,
                                   NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE]

localNeighbourSelectionmodes = [NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE,
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


densities = [0.06]
radii = [20]
interval = 1
kMax = 5
noisePercentage = 1

# ------------------------------------------------ LOCAL ---------------------------------------------------------------

data = {}

ks = [1]

# K VS. START
metrics = [
           Metrics.ORDER, 
           Metrics.ORDER_VALUE_PERCENTAGE,
           Metrics.DUAL_OVERLAY_ORDER_AND_PERCENTAGE
           ]
xAxisLabel = "timesteps"


startTime = time.time()

radius = 20
tmax = 5000

for density in densities:
    n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
    for nsm in [NeighbourSelectionMechanism.NEAREST]:
        for kCombo in [[5,1]]:
            for eventEffect in [EventEffect.AWAY_FROM_ORIGIN]:
                
                eval(density=density, n=n, radius=radius, eventEffect=eventEffect, metrics=metrics, type="ksw", nsm=nsm, combo=kCombo, evalInterval=interval, tmax=tmax)

endTime = time.time()
print(f"Total duration: {ServiceGeneral.formatTime(endTime-startTime)}")
    

