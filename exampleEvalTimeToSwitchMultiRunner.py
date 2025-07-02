import time
import numpy as np

from enums.EnumMetrics import TimeDependentMetrics
from enums.EnumSwitchType import SwitchType
from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from enums.EnumEventEffect import EventEffect   
from evaluators.EvaluatorMultiDependentInformation import EvaluatorMultiDependentInformation
import services.ServiceSavedModel as ssm
import services.ServiceGeneral as sg
import services.ServicePreparation as sp



def eval(density, n, radius, eventEffect, type, nsm=None, k=None, combo=None, evalInterval=1, tmax=15000, iMin=1, iMax=101, from_csv=False, use_agglomerative_clustering=True):
    metric = TimeDependentMetrics.TIME_TO_SWITCH
    threshold = 0.01
   
    startEval = time.time()

    sg.logWithTime(f"d={density}, r={radius}, nsm={nsm}, k={k}, combo={combo}, eventEffect={eventEffect.val}, metric={metric.name}, type={type}")
    modelParams = []
    simulationData = []
    colours = []
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
            baseFilename = f"{baseDataLocation}local/switchingInactive/local_1e_nosw_{initialStateString}_st={nsm.value}__d={density}_n={n}_r={radius}_k={k}_noise=1_drn={duration}_{e1Start}-{eventEffect.val}"
        elif type == "nsmsw":
            baseFilename = f"{baseDataLocation}{levelDataLocation}"
        elif type == "ksw":
            baseFilename = f"{baseDataLocation}{levelDataLocation}local_1e_switchType=K_{initialStateString}_st={k}_o={orderValue}_do={disorderValue}_d={density}_n={n}_r={radius}_nsm={nsm.value}_noise={noisePercentage}_drn={duration}_{e1Start}-{eventEffect.val}"

        filenames = sg.createListOfFilenamesForI(baseFilename=baseFilename, minI=iStart, maxI=iStop, fileTypeString="json")
        if type not in ["nosw", "global"]:
            modelParamsDensity, simulationDataDensity, coloursDensity, switchTypeValues = ssm.loadModels(filenames, loadSwitchValues=True)
            switchTypes.append(switchTypeValues)
        else:
            modelParamsDensity, simulationDataDensity, coloursDensity = ssm.loadModels(filenames, loadSwitchValues=False, fromCsv=True)
        modelParams.append(modelParamsDensity)
        simulationData.append(simulationDataDensity)
        colours.append(coloursDensity)

#paths.append(f"density-vs-noise_ORDER_mode-comparision_n={n}_k=1_radius=10_density={density}_noise={noisePercentage}%_hierarchical_clustering_threshold=0.01.png")
#createMultiPlotFromImages(title, numX, numY, rowLabels, colLabels, paths
# 
#      
    if type == "nosw":
        save_path = f"{metric.val}_d={density}_n={n}_r={radius}_nosw_nsm={nsm.value}_k={k}_ee={eventEffect.val}."
    elif type == "nsmsw":
        save_path = f"{metric.val}_d={density}_n={n}_r={radius}_swt=MODE_o={orderValue.value}_do={disorderValue.value}_k={k}_ee={eventEffect.val}"
    elif type == "ksw":
        save_path = f"{metric.val}_d={density}_n={n}_r={radius}_swt=K_o={orderValue}_do={disorderValue}_nsm={nsm.value}_ee={eventEffect.val}"
    
    evaluator = EvaluatorMultiDependentInformation(metric=metric,
                                                    base_paths=[save_path],
                                                    min_i=iMin,
                                                    max_i=iMax,
                                                    from_csv=from_csv,
                                                    domain_size=domainSize,
                                                    radius=radius,
                                                    threshold=threshold,
                                                    use_agglomerative_clustering=use_agglomerative_clustering)

    evaluator.evaluateAndVisualize(xLabel="time steps since last event exposure", yLabel="number occurrences", savePath=save_path, show=False) 
    endEval = time.time()
    print(f"Duration eval {sg.formatTime(endEval-startEval)}") 


def getLabelsFromNoisePercentages(noisePercentages):
    return [f"{noisePercentage}% noise" for noisePercentage in noisePercentages]

def getLabelsFromKValues(ks):
    return [f"k={k}" for k in ks]

def getLabelsFromNeighbourSelectionModes(neighbourSelectionModes):
    return [neighbourSelectionMode.name for neighbourSelectionMode in neighbourSelectionModes]

def getLabelsFromEventEffects(eventEffects):
    return [eventEffect.label for eventEffect in eventEffects]

xLabel = "time steps since last exposure to event"
yLabel = "number of occurrences"

speed = 1

angle = np.pi
blockSteps = -1
threshold = [0.1]


#domainSize = ServicePreparation.getDomainSizeForConstantDensity(0.09, 100)
domainSize = (50, 50)

distTypeString = "lssmid"
percentage = 100
e1Start = 5000

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

eventEffects = [EventEffect.ALIGN_TO_FIXED_ANGLE,
                EventEffect.AWAY_FROM_ORIGIN,
                EventEffect.RANDOM]

saveLocation = f"results/randomly_moving_predator/"
iStart = 1
iStop = 11

baseDataLocation = "J:/randomly_moving_predator/"

densities = [0.09]
radii = [10]
interval = 1
kMax = 5
noisePercentage = 1

# ------------------------------------------------ LOCAL ---------------------------------------------------------------
levelDataLocation = ""

data = {}

ks = [1, 5]

# K VS. START

iMin = 1
iMax = 11
from_csv = False
use_agglo = True

startTime = time.time()

duration = 1000
tmax = 15000

for density in densities:
    n = int(sp.getNumberOfParticlesForConstantDensity(density, domainSize))
    for radius in radii:
        for nsm in [NeighbourSelectionMechanism.NEAREST, NeighbourSelectionMechanism.FARTHEST,
                    NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE]:
            for k in ks:
                for eventEffect in eventEffects:
                    eval(density=density, n=n, radius=radius, eventEffect=eventEffect, type="nosw", k=k, combo=None, evalInterval=interval, tmax=tmax, iMin=iMin, iMax=iMax, from_csv=from_csv, use_agglomerative_clustering=use_agglo)
         
        for nsmCombo in [[NeighbourSelectionMechanism.FARTHEST, NeighbourSelectionMechanism.NEAREST]]:
            for k in ks:
                for eventEffect in eventEffects:
                    eval(density=density, n=n, radius=radius, eventEffect=eventEffect, type="nsmsw", k=k, combo=nsmCombo, evalInterval=interval, tmax=tmax, iMin=iMin, iMax=iMax, from_csv=from_csv, use_agglomerative_clustering=use_agglo)
         
        for nsm in [NeighbourSelectionMechanism.NEAREST, NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE]:
            for kCombo in [[5,1]]:
                for eventEffect in eventEffects:
                    eval(density=density, n=n, radius=radius, eventEffect=eventEffect,  type="ksw", nsm=nsm, combo=kCombo, evalInterval=interval, tmax=tmax, iMin=iMin, iMax=iMax, from_csv=from_csv, use_agglomerative_clustering=use_agglo)
        
endTime = time.time()
print(f"Total duration: {sg.formatTime(endTime-startTime)}")
    