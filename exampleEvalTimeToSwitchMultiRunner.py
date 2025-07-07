import time
import numpy as np

from enums.EnumMetrics import TimeDependentMetrics
from enums.EnumSwitchType import SwitchType
from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from enums.EnumEventEffect import EventEffect   
from enums.EnumEventSelectionType import EventSelectionType
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
    switchTypes = []

    
    if eventEffect == EventEffect.ALIGN_TO_FIXED_ANGLE:
        initialStateString = "random"
    elif eventEffect == EventEffect.AWAY_FROM_ORIGIN or eventEffect == EventEffect.RANDOM:
        initialStateString = "ordered"

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
    
    if type == "nsmsw":
        baseFilename = f"{baseDataLocation}local_nsmsw_1ev_{initialStateString}_st={nsm.value}_d={density}_n={n}_r={radius}_nsmCombo={nsmCombo[1].value}-{nsmCombo[0].value}_k={k}_noise={noisePercentage}_speed={speed}_ee={eventEffect.val}"
    elif type == "ksw":
        baseFilename = f"{baseDataLocation}local_ksw_1ev_{initialStateString}_st={k}_d={density}_n={n}_r={radius}_nsm={nsm.value}_kCombo={kCombo[1]}-{kCombo[0]}_ee={eventEffect.val}_noise={noisePercentage}_speed={speed}"

#paths.append(f"density-vs-noise_ORDER_mode-comparision_n={n}_k=1_radius=10_density={density}_noise={noisePercentage}%_hierarchical_clustering_threshold=0.01.png")
#createMultiPlotFromImages(title, numX, numY, rowLabels, colLabels, paths
# 
#      
    if type == "nsmsw":
        switch_type = SwitchType.NEIGHBOUR_SELECTION_MECHANISM
        save_path = f"{metric.val}_d={density}_n={n}_r={radius}_swt=MODE_o={orderValue.value}_do={disorderValue.value}_k={k}_ee={eventEffect.val}_{initialStateString}"
    elif type == "ksw":
        switch_type = SwitchType.K
        save_path = f"{metric.val}_d={density}_n={n}_r={radius}_swt=K_o={orderValue}_do={disorderValue}_nsm={nsm.value}_ee={eventEffect.val}_{initialStateString}"
    
    match eventEffect:
        case EventEffect.ALIGN_TO_FIXED_ANGLE:
            target_switch_value = orderValue
        case EventEffect.AWAY_FROM_ORIGIN:
            target_switch_value = disorderValue
        case EventEffect.RANDOM:
            target_switch_value = disorderValue

    evaluator = EvaluatorMultiDependentInformation(metric=metric,
                                                    base_paths=[baseFilename],
                                                    min_i=iMin,
                                                    max_i=iMax,
                                                    threshold=threshold,
                                                    use_agglomerative_clustering=use_agglomerative_clustering,
                                                    switch_type=switch_type,
                                                    from_csv=from_csv,
                                                    target_switch_value=target_switch_value,
                                                    event_start=e1Start,
                                                    event_origin_point=(domainSize[0] / 2, domainSize[1] / 2),
                                                    event_selection_type=event_selection_type,
                                                    number_of_affected=number_of_affected,
                                                    include_affected=include_affected,
                                                    evaluationTimestepInterval=evalInterval)

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

saveLocation = f"results_070725/"
iStart = 1
iStop = 11

baseDataLocation = "j:/noise_old_code/"

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

iMin = 11
iMax = 15
from_csv = False
use_agglo = True
event_selection_type = EventSelectionType.RANDOM
number_of_affected = None
include_affected = True


startTime = time.time()

duration = 1000
tmax = 15000

for density in densities:
    n = int(sp.getNumberOfParticlesForConstantDensity(density, domainSize))
    for radius in radii:
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
    