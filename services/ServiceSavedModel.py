#import csv

import codecs, json, csv
import numpy as np
import pandas as pd
import ast 

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
    vals = {}
    for key in switchValues.keys():
        vals[key] = __getSpecifiedIntervals(saveInterval, switchValues[key])
    if len(vals) > 0:
        dict["switchValues"] = {k : np.array(v).tolist() for k,v in vals.items()} # deal with np.array instances in the values
    if colours:
        dict["colours"] = [np.array(cols).tolist() for cols in colours]
    saveDict(path, dict, modelParams)


def logModelParams(path, modelParamsDict):
    """
    Logs the model params as a single row with headers.
    """
    with open(f"{path}.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(modelParamsDict.keys())
        w.writerow(modelParamsDict.values())

def initialiseCsvFileHeaders(path, headers=['t', 'i', 'x', 'y', 'u', 'v', 'colour'], addSwitchTypeHeader=True):
    """
    Appends the headers to the csv file.

    Params:
        - headers (list of strings): the headers to be inserted into the file
        - save_path (string): the path of the file where the headers should be inserted

    Returns:
        Nothing.
    """
    if addSwitchTypeHeader:
        headers.append('switchValue')
    with open(f"{path}.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)

def createSwitchValueDict(switchTypes, switchValues, i):
    switchValueDict = {}
    for switchType in switchTypes:
        switchValueDict[switchType.switchTypeValueKey] = switchValues[switchType.switchTypeValueKey][i]
    return switchValueDict

def transformSwitchValue(switchValue):
    if isinstance(switchValue, int) or isinstance(switchValue, float):
        return switchValue
    return switchValue.value

def createDictList(timestep, positions, orientations, colours, switchValues, switchTypes=[]):
    if len(switchTypes) > 0:

        return [{'t': timestep, 'i': i, 'x': positions[i][0], 'y': positions[i][1], 'u': orientations[i][0], 'v': orientations[i][1], 'colour': colours[i], 'switchValue': createSwitchValueDict(switchTypes, switchValues, i)} for i in range(len(positions))]
    return [{'t': timestep, 'i': i, 'x': positions[i][0], 'y': positions[i][1], 'u': orientations[i][0], 'v': orientations[i][1], 'colour': colours[i]} for i in range(len(positions))]

def saveModelTimestep(timestep, positions, orientations, colours, path, switchValues, switchTypes=[]):
    dict_list = createDictList(timestep, positions, orientations, colours, switchValues, switchTypes)
    with open(f"{path}.csv", 'a', newline='') as f:
        w = csv.writer(f)
        for dict in dict_list:
            w.writerow(dict.values())

def loadModelFromCsv(filepathData, filePathModelParams, switchTypes=[], loadColours=False):
    dfParams = pd.read_csv(filePathModelParams,index_col=False)
    modelParams = dfParams.to_dict(orient='records')[0]
    domainSize = modelParams['domainSize'].split(',')
    modelParams['domainSize'] = [float(domainSize[0][1:]), float(domainSize[1][:-1])]

    if len(switchTypes) > 0:
        df = pd.read_csv(filepathData, index_col=False, converters = {'switchValue': to_dict})
    else:
        df = pd.read_csv(filepathData,index_col=False)
    times = []
    positions = []
    orientations = []
    colours = []
    switchValues = {k.switchTypeValueKey :[] for k in switchTypes}
    #tmax = df['t'].max()
    for t in df['t']:
        if t not in times:
            times.append(t)
            dfT = df[df['t'] == t]
            positions.append(np.column_stack((dfT['x'], dfT['y'])))
            orientations.append(np.column_stack((dfT['u'], dfT['v'])))
            if loadColours:
                colours.append(dfT['colour'].to_list())
            if len(switchTypes) > 0:
                for switchType in switchTypes:
                    switchValues[switchType.switchTypeValueKey].append(extract_values(dfT, 'switchValue', switchType.switchTypeValueKey))
    if len(switchTypes) > 0:
        if loadColours:
            return modelParams, (np.array(times), np.array(positions), np.array(orientations)), switchValues, np.array(colours)
        else:
            return modelParams, (np.array(times), np.array(positions), np.array(orientations)), switchValues
    if loadColours:
        return modelParams, (np.array(times), np.array(positions), np.array(orientations)), np.array(colours)
    else:
        return modelParams, (np.array(times), np.array(positions), np.array(orientations))

def loadModel(path, switchTypes=[], loadSwitchValues=False, loadColours=False):
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
    switchTypeValues = {}
    if loadSwitchValues == True and loadColours == True:
        switchValues = loadedJson["switchValues"]
        for switchType in switchTypes:
            switchTypeValues[switchType.switchTypeValueKey].append(switchValues[switchType.switchTypeValueKey])
        colours = loadedJson["colours"]
        return modelParams, (time, positions, orientations), switchValues, colours
    elif loadSwitchValues == True:
        switchValues = loadedJson["switchValues"]
        for switchType in switchTypes:
            switchTypeValues[switchType.switchTypeValueKey].append(switchValues[switchType.switchTypeValueKey])
        return modelParams, (time, positions, orientations), switchValues
    elif loadColours == True:
        colours = loadedJson["colours"]
        return modelParams, (time, positions, orientations), colours
    return modelParams, (time, positions, orientations)

def loadModels(paths, switchTypes=[], loadSwitchValues=False, loadColours=False, loadFromCsv=False):
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
        filePathModelParams = path.split(".")[0] + '_modelParams.' + path.split(".")[1]
        if loadSwitchValues == True and loadColours == True:
            if loadFromCsv:
                modelParams, simulationData, switchValues, colours = loadModelFromCsv(filepathData=path, filePathModelParams=filePathModelParams, switchTypes=switchTypes, loadColours=loadColours)
            else:
                modelParams, simulationData, switchValues, colours = loadModel(path, switchType=switchTypes, loadSwitchValues=loadSwitchValues, loadColours=loadColours)
            switchValuesArr.append(switchValues)
            coloursArr.append(colours)
        elif loadSwitchValues == True:
            if loadFromCsv:
                modelParams, simulationData, switchValues = loadModelFromCsv(filepathData=path, filePathModelParams=filePathModelParams, switchTypes=switchTypes, loadColours=loadColours)
            else:
                modelParams, simulationData, switchValues = loadModel(path, switchTypes=switchTypes, loadSwitchValues=loadSwitchValues)
            switchValuesArr.append(switchValues)
        elif loadColours == True:
            if loadFromCsv:
                modelParams, simulationData, colours = loadModelFromCsv(filepathData=path, filePathModelParams=filePathModelParams, loadColours=loadColours)
            else:
                modelParams, simulationData, colours = loadModel(path, switchTypes=switchTypes, loadSwitchValues=loadSwitchValues, loadColours=loadColours)
            coloursArr.append(colours)
        else:
            if loadFromCsv:
                modelParams, simulationData = loadModelFromCsv(filepathData=path, filePathModelParams=filePathModelParams, loadColours=loadColours)
            else:
                modelParams, simulationData = loadModel(path, switchTypes=switchTypes, loadSwitchValues=loadSwitchValues)
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

def to_dict(x):
    try:
        y = ast.literal_eval(x)
        if type(y) == dict:
            return y
    except:
        return None
    
def extract_values(df, column, key):
    return list(df[column].apply(lambda x: pd.Series(extract_value(x, key))).to_dict()[0].values())

def extract_value(d, key):
    return d[key]


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