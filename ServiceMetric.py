
import numpy as np
import math
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering

from EnumMetrics import Metrics
import ServiceOrientations

"""
Service containing static methods to handle metrics.
"""

def evaluateSingleTimestep(positions, orientations, metric, radius=None, threshold=0.99, switchTypeValues=None, switchTypeOptions=None):
     """
        Evaluates the simulation data for a single timestep according to the selected metric.

        Parameters:
            - positions (array): the position of every particle at this timestep
            - orientations (array): the orientation of every particle at this timestep
            - metric (EnumMetrics): the metric for evaluating the data
            - radius (int) [optional]: the perception radius of every particle. Radius is only relevant for certain metrics such as Clustering, therefore it can be None for the others.
            - threshold (float) [optional]: the threshold for the agglomerative clustering
            - switchTypeValues (array) [optional]: the switch type values for individual switching
            - switchTypeOptions (tuple) [optional]: contains the orderValue and the disorderValue respectively
        Returns:
            An array of the results according to the metric.
     """
     n = len(positions)
     match metric:
        case Metrics.ORDER:
            return computeGlobalOrder(orientations)
        case Metrics.CLUSTER_NUMBER:
            nClusters, _ = findClusters(positions, orientations, threshold)
            return nClusters
        case Metrics.CLUSTER_NUMBER_WITH_RADIUS:
            nClusters, _ = findClustersWithRadius(positions, orientations, radius, threshold)
            return nClusters
        case Metrics.CLUSTER_SIZE:
            nClusters, clusters = findClusters(positions, orientations, threshold)
            clusterSizes = computeClusterSizes(nClusters, clusters)
            return clusterSizes
        case Metrics.ORDER_VALUE_PERCENTAGE:
            orderCount, _ = getNumbersPerSwitchTypeValue(switchTypeValues, switchTypeOptions)
            return orderCount
        case Metrics.DUAL_OVERLAY_ORDER_AND_PERCENTAGE: # not a single metric but rather overlaying two metrics in the same graph
            order = computeGlobalOrder(orientations)
            orderCount, _ = getNumbersPerSwitchTypeValue(switchTypeValues, switchTypeOptions)
            return order, orderCount/100 # normalise to fit with order
        case Metrics.AVERAGE_NUMBER_NEIGHBOURS:
            _, avg, _ =  getMinAvgMaxNumberOfNeighbours(positions, radius)
            return avg
        case Metrics.MIN_AVG_MAX_NUMBER_NEIGHBOURS:
            return getMinAvgMaxNumberOfNeighbours(positions, radius)
        case Metrics.AVG_DISTANCE_NEIGHBOURS:
            _, avg, _ = getMinAvgMaxDistanceOfNeighbours(positions, radius)
            return avg
        case Metrics.AVG_CENTROID_DISTANCE:
            _, avg, _ = getMinAvgMaxDistanceFromCentroid(positions)
            return avg
     
def computeGlobalOrder(orientations):
    """
    Computes the order within the provided orientations. 
    Can also be called for a subsection of all particles by only providing their orientations.

    Params:
        - orientations (array of (u,v)-coordinates): the orientation of all particles that should be included
    
    Returns:
        A float representing the order in the given orientations
    """
    """
    sumOrientation = [0,0]
    for j in range(len(orientations)):
        sumOrientation += orientations[j]
    return np.sqrt(sumOrientation[0]**2 + sumOrientation[1]**2) / len(orientations)
    """
    sumOrientation = np.sum(orientations[np.newaxis,:,:],axis=1)
    return np.divide(np.sqrt(np.sum(sumOrientation**2,axis=1)), len(orientations))[0]

def computeLocalOrders(orientations, neighbours):
    """
    Computes the local order for every individual.

    Params: 
        - orientations (array of floats): the orientation of every individual at the current timestep
        - neighbours (array of arrays of booleans): the identity of every neighbour of every individual

    Returns:
        An array of floats representing the local order for every individual at the current time step (values between 0 and 1)
    """
    sumOrientation = np.sum(neighbours[:,:,np.newaxis]*orientations[np.newaxis,:,:],axis=1)
    localOrders = np.divide(np.sqrt(np.sum(sumOrientation**2,axis=1)), np.count_nonzero(neighbours, axis=1))
    return localOrders 

def findClusters(positions, orientations, threshold):
    """
    Find clusters in the data using AgglomerativeClustering.

    Params:
        - positions (array of arrays of float): the position of every particle at every timestep
        - orientations (array of arrays of float): the orientation of every particle at every timestep
        - threshold (float): the threshold used to cut the tree in AgglomerativeClustering

    Returns:
        The number of clusters, the labels of the clusters
    """
    cluster = AgglomerativeClustering(n_clusters=None, metric='euclidean', linkage='single', compute_full_tree=True, distance_threshold=threshold)

    # Cluster the data
    cluster.fit_predict(orientations)

    # number of clusters
    nClusters = 1+np.amax(cluster.labels_)

    return nClusters, cluster.labels_
    

def getOrientationsXy(positions, orientations):
    # TODO move to ServiceOrientations
    """
    Computes the (x,y)-coordinate equivalents of the (u,v)-coordinate orientations with repect to the current positions

    Params:
        - positions (array of array of floats): the positions of every particle
        - orientations (array of array of floats): the orientations of every particle

    Returns:
        An array containing the orientation of every particle in (x,y)-coordinates.
    """
    return [[positions[i][0] + np.cos(np.arcsin(orientations[i][0])), positions[i][1] + np.sin(np.arcsin(orientations[i][0]))] for i in range(len(orientations))]

         
def findClustersWithRadius(positions, orientations, radius, threshold=0.99):
    """
    Finds clusters in the particle distribution. The clustering is performed according to the following constraints:
        - to belong to a cluster, a particle needs to be within the radius of at least one other member of the same cluster
        - to belong to a cluster, the orientation has to be similar or equal (<= 1.29° orientation difference by default)
    
    Parameters:
        - positions (array): the position of every particle at the current timestep
        - orientations (array): the orientations of every particle at the current timestep
        - radius (int): the perception radius of the particles
    
    Returns:
        A tuple containing the number of clusters and the clusters.
    """
    if radius == None:
        print("ERROR: Radius needs to be provided for clustering.")

    n = len(positions)
    clusters = np.zeros(n)
    clusterMembers = np.zeros((n,n))

    for i in range(n):
        neighbourIndices = findNeighbours(i, positions, radius)
        for neighbourIdx in neighbourIndices:
            localOrder = computeGlobalOrder([orientations[i], orientations[neighbourIdx]])
            if localOrder >= threshold:
                clusterMembers[i][neighbourIdx] = 1
    
    clusterCounter = 1
    for i in range(n):
        if markClusters(i, clusterCounter, clusters, clusterMembers, n) == True:
            clusterCounter += 1

    return clusterCounter, clusters
       
            
def findNeighbours(i, positions, radius):
    """
    Determines which particles are neighbours of particle i.

    Parameters:
        - i (int): the index of the target particle for which the neighbours should be found
        - positions (array): the position of every particle at the current timestep
        - radius (int): the perception radius of every particle

    Returns:
        A list of indices of all neighbours within the perception range.
    """
    return [idx for idx in range(len(positions)) if isNeighbour(radius, positions, i, idx) and idx != i]


def isNeighbour(radius, positions, targetIdx, candidateIdx):
    """
    Checks if two particles are neighbours.

    Parameters:
        - radius (int): the perception radius of every particle
        - positions (array): the position of every particle at the current timestep
        - targetIdx (int): the index of the target particle within the positions array
        - candidateIdx (int): the index of the candidate particle within the positions array
    
    Returns:
        A boolean stating whether or not the two particles are neighbours.
    """
    return ((positions[candidateIdx][0] - positions[targetIdx][0])**2 + (positions[candidateIdx][1] - positions[targetIdx][1])**2) <= radius **2 

def angleBetweenTwoVectors(vec1, vec2):
    # TODO: move to ServiceOrientations
    """
    Computes the angle between to vectors.

    Params:
        - vec1 (array of floats): the first vector
        - vec2 (array of floats): the second vector.

    Returns:
        Float representing the angle between the two vectors.
    """
    return np.arctan2(vec1[1]-vec2[1], vec1[0]-vec2[0])

def cosAngle(vec1, vec2):
    # TODO: move to ServiceOrientations
    """
    Checks the relative orientations of two particles. If the cosAngle is close to 1, their directions are identical. 
    If it is close to -1, they look in opposite directions.

    Parameters:
        - vec1 (vector): the orientation of the first particle
        - vec2 (vector): the orientation of the second particle

    Returns:
        The cosAngle as an integer representing the similarity of the orientations.
    """
    return (vec1[0] * vec2[0] + vec1[1] * vec2[1]) / math.sqrt((vec1[0]**2 + vec1[1]**2) * (vec2[0]**2 + vec2[1]**2))

def markClusters(currentIdx, clusterCounter, clusters, clusterMembers, n):
    """
    Recursive function that marks all neighbours with a similar orientation as belonging to the same cluster.

    Parameters:
        - currentIdx (int): index of the current particle within the clusters array
        - clusterCounter (int): the current maximum of found clusters. Will not be updated in this function
        - clusters (array): represents the cluster membership of all particles by the id of the cluster
        - clusterMembers (array of arrays): represents which other particles are neighbours with a similar 
            orientation to the current particle, i.e. the members of the same cluster as seen by the current particle
        - n (int): the total number of particles in the domain

    Returns:
        If any particle's cluster id has been updated. If the particle's own id is not updated, neither are its children.
        Therefore, the return values are not compared.
    """
    if clusters[currentIdx] != 0:
        return False
    clusters[currentIdx] = clusterCounter
    for i in range(n):
        if clusterMembers[currentIdx][i] == 1:
            markClusters(i, clusterCounter, clusters, clusterMembers, n)
    return True

def computeClusterSizes(clusterCounter, clusters):
    """
    Computes the size of every cluster.

    Parameters:
        - clusterCounter (int): the total number of clusters in the current state of the domain
        - clusters (array): array containing the id of the cluster that every particle belongs to

    Returns:
        An array with the length of clusterCounter containing the size of the respective cluster.
    """
    clusterSizes = clusterCounter * [0]
    for cluster in clusters:
        clusterSizes[int(cluster)] += 1
    return clusterSizes

def computeClusterNumberOverParticleLifetime(clusters):
    """
    Computes the number of clusters that every particle has belonged to over the whole course of a simulation.

    Parameters:
        - clusters (array of arrays): Contains the cluster membership for every particle at every timestep

    Returns:
        A dictionary with the index of the particle as its key and the total number of clusters that the particle 
        has belonged to as its value.
    """
    dd = defaultdict(list)
    for i in range(len(clusters[0])):
        for key, value in clusters.items():
            dd[i].append(value[i])
    countsPerParticle = {}
    for key, values in dd.items():
        countsPerParticle[key] = len(np.unique(values))
    return countsPerParticle

def identifyClusters(clusters, orientations):
    # TODO fix
    clusterIds = {}
    referenceOrientations = {}
    counter = 0
    # identify the orientation for the cluster
    # add cluster colour to clusterColours for the corresponding index
    # use orientation as key to combine them

    # could also try identifying the clusters by checking if the majority of the particles is the same
    """
    for i in range(len(clusters)):
        handledClusters = []
        for j in range(len(clusters[0])):
            clusterId = clusters[i][j]
            orientation = orientations[i][j]
            if clusterId not in handledClusters:
                isNewCluster = True
                for clusterIdx in clusterIds.keys():
                    referenceOrientation = referenceOrientations.get(clusterIdx)
                    if cosAngle(referenceOrientation, orientation) == 1:
                        isNewCluster = False
                        clusterIds[clusterIdx] = clusterIds.get(clusterIdx).append(clusterId)  
                if isNewCluster == True:
                    clusterIds[counter] = [clusterId]
                    referenceOrientations[counter] = orientation
                    counter += 1
                handledClusters.append(clusterId)
    return clusterIds
    """
    """
    clusterMembers = {}
    for timestep in range(len(clusters)):
        stepMembers = {}
        for particleIdx in range(len(clusters[0])):
            clusterId = clusters[timestep][particleIdx]
            if clusterId in stepMembers.keys():
                stepMembers[clusterId] = stepMembers[clusterId].append(particleIdx)
            else:
                stepMembers[clusterId] = [particleIdx]
                
    """

def getNumbersPerSwitchTypeValue(switchTypeValues, switchTypeOptions):
    """
    Counts the occurrences for all switch type values.

    Params:
        - switchTypeValues (array): the switch type values for individual switching
        - switchTypeOptions (array): all possible switch type values
    Returns:
        Two integer representing the counts of the orderValue and disorderValue respectively
    """
    unique, counts = np.unique(switchTypeValues, return_counts=True)
    d = dict(zip(unique, counts))
    sortedD = dict(sorted(d.items()))
    n = sum(list(sortedD.values()))
    if sortedD.get(switchTypeOptions[0]) != None:
        percentageOrdered = sortedD.get(switchTypeOptions[0])/n
    else:
        percentageOrdered = 0
    if sortedD.get(switchTypeOptions[1]) != None:
        percentageDisordered = sortedD.get(switchTypeOptions[1])/n
    else:
        percentageDisordered = 0

    return percentageOrdered * 100, percentageDisordered * 100

def getLocalOrderGrid(simulationData, domainSize):
    """
    Computes the local order for the whole domain based on a grid of 10x10 cells for every timestep.

    Params:
        - simulationData (times, positions, orientations): the data
        - domainSize (tuple of floats): the x and y widths of the domain
    """
    # TODO add radius dependency
    timesteps, positions, orientations = simulationData 
    length = domainSize[0]/10
    localOrderGrid = np.ones((len(timesteps), 10, 10))
    for t in timesteps:
        for x in range(0, 10):
            for y in range(0, 10):
                candOrientations = [orientations[t][part] for part in range(len(positions[t])) if positions[t][part][0] <= ((x+1) * length) and positions[t][part][0] >= (x * length) and positions[t][part][1] <= ((y+1) * length) and positions[t][part][1] >= (y * length)]
                if len(candOrientations) > 0:
                    localOrderGrid[t][x][y] = computeGlobalOrder(candOrientations)
    return timesteps, localOrderGrid

def checkTurnSuccess(orientations, fixedAngle, noise, eventStartTimestep, interval=100):
    """
    Checks if the event has managed to make the whole swarm align to the new angle or if the defecting group has been
    reabsorbed.
    Is only really useful for EventEffect.ALIGN_TO_FIXED_ANGLE (DISTANT).

    Params:
        - orientations (array of (u,v)-coordinates): the orientation of every particle at every timestep
        - fixedAngle (angle in radians): the angle to which the affected particles have turned
        - noise (float): the noise level in the domain impacting the actual orientations
        - eventStartstep (int): the timestep at which the event first occurs
        - eventDuration (int) [optional]: the number of timesteps that need to pass before comparing. By default 100.

    Returns:
        Boolean signifying whether the whole swarm has managed to align to the fixed angle.
    """
    #print("starting turn success eval...")
    if eventStartTimestep == 0:
        raise Exception("Cannot be used if there is no previous timestep for comparison")
    if eventStartTimestep + interval > len(orientations[0]):
        interval -= 1

    orientationsBefore = orientations[eventStartTimestep-1]
    orientationsExpected = [ServiceOrientations.computeUvCoordinates(fixedAngle) for orient in orientationsBefore]
    orientationsAfter = orientations[eventStartTimestep+interval]
    if (1-computeGlobalOrder(orientationsAfter)) <= noise: # if the swarm is aligned
        before = [ServiceOrientations.normaliseAngle(ServiceOrientations.computeAngleForOrientation(orientationsBefore[i])) for i in range(0, len(orientationsBefore))]
        after = [ServiceOrientations.normaliseAngle(ServiceOrientations.computeAngleForOrientation(orientationsAfter[i])) for i in range(0, len(orientationsAfter))]
        expected = [ServiceOrientations.normaliseAngle(ServiceOrientations.computeAngleForOrientation(orientationsExpected[i])) for i in range(0, len(orientationsExpected))]

        beforeAvg = np.average(before)
        afterAvg = np.average(after)
        expectedAvg = np.average(expected)

        if np.absolute(beforeAvg-expectedAvg) <= noise:
            #print("No turn necessary")
            return "not_necessary"

        # if the average new angle is closer to the expected angle and the difference between the expected and the new angles can be explained by noise, the turn was successful
        if np.absolute(expectedAvg-afterAvg) < np.absolute(beforeAvg-afterAvg) and np.absolute(expectedAvg-afterAvg) <= noise:
            return "turned"

    return "not_turned"

def getOverallMinAvgMaxNumberOfNeighbours(positions, radius):
    mins = []
    avgs = []
    maxs = []
    for stepPositions in positions:
        stepMin, stepAvg, stepMax = getMinAvgMaxNumberOfNeighbours(stepPositions, radius)
        mins.append(stepMin)
        avgs.append(stepAvg)
        maxs.append(stepMax)
    return np.min(mins), np.average(avgs), np.max(maxs)

def getMinAvgMaxNumberOfNeighbours(positions, radius):
    """
    Determines the minimum, average and maximum of neighbours perceived by the particles

    Params:
        - positions (array of (x,y)-coordinates): the current positions of all particles
        - radius (float): the perception radius of the particles

    Returns:
        3 floats representing the minimum, average and maximum number of neighbours
    """
    neighbourNumbersArray = np.zeros(len(positions))
    for i in range(len(positions)):
        neighbours = findNeighbours(i, positions, radius)
        neighbourNumbersArray[i] = len(neighbours)
    neighbourNumbersArray.sort()
    return np.min(neighbourNumbersArray), np.average(neighbourNumbersArray), np.max(neighbourNumbersArray)

def getMinAvgMaxDistanceOfNeighbours(positions, radius):
    neighbourDistances = []
    for i in range(len(positions)):
        neighbours = findNeighbours(i, positions, radius)
        candidateDistances = np.array([math.dist(positions[i], positions[candidateIdx]) for candidateIdx in neighbours])
        if len(neighbours) > 0:
            for dist in candidateDistances:
                neighbourDistances.append(dist)
    distancesFlattened = np.array(neighbourDistances).flatten()
    if len(distancesFlattened) == 0:
        return 0, 0, 0
    return np.min(distancesFlattened), np.average(distancesFlattened), np.max(distancesFlattened)


def getMinAvgMaxDistanceFromCentroid(positions):
    centroid = np.mean(positions, axis=0)
    distances = [math.dist(pos, centroid) for pos in positions]
    return np.min(distances), np.average(distances), np.max(distances)

"""
json:
    {
     neighbours: {0: [1, 2, 3], 1: [...]},
     distances: {0: [1.1, 2.3, 3.4, 4.2, 5.3, 6.2]},
     localOrders: {0: 0.42, 1: 0.32}
     orientationDifferences: {0: [0.03, 0.05, 0.24, 0.54, 0.12, 0.99]}
     selected: {0: [1], 1: [3]}
     }
"""

            






