#import csv

import codecs, json
import numpy as np
"""
Service contains static methods to save and load models to/from json files.
"""

def saveModel(simulationData, path="sample.json", modelParams=None, saveInterval=1, switchValues={'nsms': [], 'ks': [], 'speeds': [], 'activationTimeDelays': []}, colours=None):
    """
    Saves a model trained by the Viscek simulator implementation.

    Parameters:
        - simulationData (times, positions, orientations): the data to be saved
        - path (string) [optional]: the location and name of the target file
        - modelParams (dict) [optional]: a summary of the model's params such as n, k, neighbourSelectionMode etc.
        - saveInterval (int) [optional]: specifies the interval at which the saving should occur, i.e. if any time steps should be skipped
        - switchValues (array) [optional]: the switch type value assigned to each particle at every timestep
        
    Returns:
        Nothing. Creates or overwrites a file.
    """
    time, positions, orientations = simulationData
    dict = {"time": __getSpecifiedIntervals(saveInterval, time.tolist()), 
            "positions": __getSpecifiedIntervals(saveInterval, positions.tolist()), 
            "orientations": __getSpecifiedIntervals(saveInterval, orientations.tolist())}
    for key in switchValues.keys():
        vals = {key :__getSpecifiedIntervals(saveInterval, switchValues[key])}
        dict["switchValues"] = {k : np.array(v).tolist() for k,v in vals.items()} # deal with np.array instances in the values
    if colours:
        dict["colours"] = [np.array(cols).tolist() for cols in colours]
    saveDict(path, dict, modelParams)

def loadModel(path, loadSwitchValues=False, loadColours=False):
    """
    Loads a single model from a single file.

    Parameters:
        - path (string): the location and file name of the file containing the model data
        - loadSwitchValues (boolean) [optional]: loads the switch type values from the save file

    Returns:
        The model's params as well as the simulation data containing the time, positions, orientations.
    """
    loadedJson = __loadJson(path)

    modelParams = loadedJson["modelParams"]
    time = np.array(loadedJson["time"])
    positions = np.array(loadedJson["positions"])
    orientations = np.array(loadedJson["orientations"])
    if loadSwitchValues == True and loadColours == True:
        switchValues = loadedJson["switchValues"]
        colours = loadedJson["colours"]
        return modelParams, (time, positions, orientations), switchValues, colours
    elif loadSwitchValues == True:
        switchValues = loadedJson["switchValues"]
        return modelParams, (time, positions, orientations), switchValues
    elif loadColours == True:
        colours = loadedJson["colours"]
        return modelParams, (time, positions, orientations), colours
    return modelParams, (time, positions, orientations)

def loadModels(paths, loadSwitchValues=False, loadColours=False):
    """
    Loads multiple models from multiple files.

    Parameters:
        - paths (array of strings): An array containing the locations and names of the files containing a single model each
        - loadSwitchValues (boolean) [optional]: loads the switch type values from the save files
        
    Returns:
        Returns an array containing the model params for each model and a second array containing the simulation data for each model. Co-indexed.
    """
    data = []
    params = []
    switchValuesArr = []
    coloursArr = []
    for path in paths:
        if loadSwitchValues == True and loadColours == True:
            modelParams, simulationData, switchValues, colours = loadModel(path, loadSwitchValues=loadSwitchValues, loadColours=loadColours)
            switchValuesArr.append(switchValues)
            coloursArr.append(colours)
        elif loadSwitchValues == True:
            modelParams, simulationData, switchValues = loadModel(path, loadSwitchValues=loadSwitchValues)
            switchValuesArr.append(switchValues)
        elif loadColours == True:
            modelParams, simulationData, colours = loadModel(path, loadSwitchValues=loadSwitchValues, loadColours=loadColours)
            coloursArr.append(colours)
        else:
            modelParams, simulationData = loadModel(path, loadSwitchValues=loadSwitchValues)
        params.append(modelParams)
        data.append(simulationData)
    if loadSwitchValues == True and loadColours == True:
        return params, data, switchValuesArr, coloursArr
    elif loadSwitchValues == True:
        return params, data, switchValuesArr
    elif loadColours == True:
        return params, data, coloursArr
    return params, data

def saveTimestepsResults(results, path, modelParams=None, saveInterval=1):
    """
    Saves evaluator results for all timesteps.

    Parameters:
        - results (dictionary): the evaluation results per timestep
        - path (string): the location and name of the target file
        - modelParams (dict) [optional]: a summary of the model's params such as n, k, neighbourSelectionMode etc.
        - saveInterval (int) [optional]: specifies the interval at which the saving should occur, i.e. if any time steps should be skipped
    
    Returns:
        Nothing. Creates or overwrites a file.
    """
    dict = {"time": __getSpecifiedIntervals(saveInterval, list(results.keys())),
            "results": __getSpecifiedIntervals(saveInterval, list(results.values()))}
    saveDict(path, dict, modelParams)

def loadTimestepsResults(path):
    """
    Loads the evaluation results from a single file.

    Parameters:
        - path (string): the location and file name of the file containing the model data

    Returns:
        The model's params as well as the evaluation data as a {time: results} dictionary
    """
    loadedJson = __loadJson(path)
    modelParams = loadedJson["modelParams"]
    time = np.array(loadedJson["time"])
    results = np.array(loadedJson["results"])
    data = {time[i]: results[i] for i in range(len(time))}
    return modelParams, data

def saveConnectionTrackingInformation(data, path="sample.json"):
    """
    Saves a model trained by the Viscek simulator implementation.

    Parameters:
        - data (dict): the data to be saved
        - path (string) [optional]: the location and name of the target file
        - saveInterval (int) [optional]: specifies the interval at which the saving should occur, i.e. if any time steps should be skipped
        
    Returns:
        Nothing. Creates or overwrites a file.
    """
    saveDict(path, data)

def loadConnectionTrackingInformation(path):
    """
    Loads a single model from a single file.

    Parameters:
        - path (string): the location and file name of the file containing the model data
        - loadSwitchValues (boolean) [optional]: loads the switch type values from the save file

    Returns:
        The model's params as well as the simulation data containing the time, positions, orientations.
    """
    loadedJson = __loadJson(path)

    neighbours = loadedJson["neighbours"]
    distances = loadedJson["distances"]
    localOrders = loadedJson["localOrders"]
    orientationDifferences = loadedJson["orientationDifferences"]
    selected = loadedJson["selected"]
    return neighbours, distances, localOrders, orientationDifferences, selected

def loadConnectionTrackingInformations(paths):
    """
    Loads multiple instances of connection tracking information from multiple files.

    Parameters:
        - paths (array of strings): An array containing the locations and names of the files containing a single model each
        
    Returns:
        Returns an array containing the model params for each model and a second array containing the simulation data for each model. Co-indexed.
    """
    neighbours = []
    distances = []
    localOrders = []
    orientationDifferences = []
    selected = []
    for path in paths:
        neighs, dists, los, ods, sels = loadConnectionTrackingInformation(path)
        neighbours.append(neighs)
        distances.append(dists)
        localOrders.append(los)
        orientationDifferences.append(ods)
        selected.append(sels)
    return neighbours, distances, localOrders, orientationDifferences, selected
    
    
def __getSpecifiedIntervals(interval, lst):
    """
    Selects the data within the list which coincides with the specified interval, e.g. every third data point.

    Parameters:
        - interval (int): which data points should be considered, e.g. 3 would indicate indices 0, 3, 6 etc.
        - lst (list): the data to be reduced according to the intervals
    
    Returns:
        A reduced list containing only the data points of the original list at the specified intervals.
    """
    return [lst[idx] for idx in range(0, len(lst)) if idx % interval == 0]

def saveGenInfo(path, dict):
    fitnessDict = {}
    allFit = dict["all_fitnesses"]
    for i in range(len(allFit)):
        for key in allFit[i].keys():
            fitnessDict[str(key)] = allFit[i][key]
    dict["all_fitnesses"] = [fitnessDict]
    saveDict(path, dict)

def saveDict(path, dict, modelParams=None):
    """
    Saves the values of a dictionary to a file at the specified path.

    Parameters:
        - path (string): the location and name of the target file
        - dict (dictionary): the dictionary containing the data to be saved
        - modelParams (dict) [optional]: a summary of the model's params such as n, k, neighbourSelectionMode etc.

    Returns:
        Nothing. Creates or overwrites a file.
    """
    if modelParams != None:
        paramsDict = {"modelParams": modelParams}
        paramsDict.update(dict)
        dict = paramsDict
        
    with open(path, "w") as outfile:
        json.dump(dict, outfile)

def __loadJson(path):
    """
    Loads data as JSON from a single file.

    Parameters:
        - path (string): the location and file name of the file containing the data

    Returns:
        All the data from the file as JSON.
    """
    obj_text = codecs.open(path, 'r', encoding='utf-8').read()
    return json.loads(obj_text)
   

"""
def appendCsvRow(path, fieldnames, rowDict):
    with open(path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(rowDict)
"""
"""
def appendCsvRow(path, row):
    with open(path,'a') as fd:
        fd.write(row)
"""