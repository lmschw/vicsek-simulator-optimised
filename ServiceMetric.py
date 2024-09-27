
import numpy as np
import math
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering

from EnumMetrics import Metrics
import ServiceOrientations
import ServiceVicsekHelper

"""
Service containing static methods to handle metrics.
"""

def evaluateSingleTimestep(positions, orientations, metric, domainSize=None, radius=None, threshold=0.99, switchTypeValues=None, switchTypeOptions=None):
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
            nClusters, _ = findClusters(orientations, threshold)
            return nClusters
        case Metrics.CLUSTER_NUMBER_WITH_RADIUS:
            nClusters, _ = findClustersWithRadius(positions, orientations, domainSize, radius, threshold)
            return nClusters
        case Metrics.CLUSTER_SIZE:
            nClusters, clusters = findClusters(orientations, threshold)
            # TODO: make sure the change from array to dict is taken care of in the visualisation
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
            _, avg, _ =  getMinAvgMaxNumberOfNeighbours(positions, domainSize, radius)
            return avg
        case Metrics.MIN_AVG_MAX_NUMBER_NEIGHBOURS:
            return getMinAvgMaxNumberOfNeighbours(positions, domainSize, radius)
        case Metrics.AVG_DISTANCE_NEIGHBOURS:
            _, avg, _ = getMinAvgMaxDistanceOfNeighbours(positions, domainSize, radius)
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
    return np.divide(np.sqrt(np.sum(sumOrientation**2,axis=1)), np.count_nonzero(neighbours, axis=1))

def findClusters(orientations, threshold):
    """
    Find clusters in the data using AgglomerativeClustering.

    Params:
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
         
def findClustersWithRadius(positions, orientations, domainSize, radius, threshold=0.99):
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


    # TODO: finish refactoring
    neighbours = ServiceVicsekHelper.getNeighbours(positions=positions, domainSize=domainSize, radius=radius)
    localOrders = computeLocalOrders(orientations=orientations, neighbours=neighbours)



    for i in range(n):
        neighbourIndices = []
        #neighbourIndices = findNeighbours(i, positions, radius)
        for neighbourIdx in neighbourIndices:
            localOrder = computeGlobalOrder([orientations[i], orientations[neighbourIdx]])
            if localOrder >= threshold:
                clusterMembers[i][neighbourIdx] = 1
    
    clusterCounter = 1
    for i in range(n):
        if markClusters(i, clusterCounter, clusters, clusterMembers, n) == True:
            clusterCounter += 1

    return clusterCounter, clusters

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
    # TODO refactor
    if clusters[currentIdx] != 0:
        return False
    clusters[currentIdx] = clusterCounter
    for i in range(n):
        if clusterMembers[currentIdx][i] == 1:
            markClusters(i, clusterCounter, clusters, clusterMembers, n)
    return True

def computeClusterSizes(clusters):
    """
    Computes the size of every cluster.

    Parameters:
        - clusterCounter (int): the total number of clusters in the current state of the domain
        - clusters (array): array containing the id of the cluster that every particle belongs to

    Returns:
        An dictionary with the ids of the cluster as keys and the number of members as values.
    """
    unique, counts = np.unique(clusters, return_counts=True)
    return dict(zip(unique, counts))

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
    n = sum(list(d.values()))
    if d.get(switchTypeOptions[0]) != None:
        percentageOrdered = d.get(switchTypeOptions[0])/n
    else:
        percentageOrdered = 0
    if d.get(switchTypeOptions[1]) != None:
        percentageDisordered = d.get(switchTypeOptions[1])/n
    else:
        percentageDisordered = 0

    return percentageOrdered * 100, percentageDisordered * 100

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
    # TODO refactor
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

def getMinAvgMaxNumberOfNeighbours(positions, domainSize, radius):
    """
    Determines the minimum, average and maximum of neighbours perceived by the particles

    Params:
        - positions (array of (x,y)-coordinates): the current positions of all particles
        - radius (float): the perception radius of the particles

    Returns:
        3 floats representing the minimum, average and maximum number of neighbours
    """
    neighbours = ServiceVicsekHelper.getNeighbours(positions=positions, domainSize=domainSize, radius=radius)
    np.fill_diagonal(neighbours, False)
    neighbourNumbersArray = np.count_nonzero(neighbours, axis=1)
    return np.min(neighbourNumbersArray), np.average(neighbourNumbersArray), np.max(neighbourNumbersArray)

def getMinAvgMaxDistanceOfNeighbours(positions, domainSize, radius):
    neighbours = ServiceVicsekHelper.getNeighbours(positions=positions, domainSize=domainSize, radius=radius)
    np.fill_diagonal(neighbours, False)
    posDiff = ServiceVicsekHelper.getPositionDifferences(positions=positions, domainSize=domainSize)
    maskedArray = np.ma.MaskedArray(posDiff, mask=neighbours==False, fill_value=0)
    return np.min(maskedArray), np.average(maskedArray), np.max(maskedArray)

def getMinAvgMaxDistanceFromCentroid(positions):
    centroid = np.mean(positions, axis=0)
    distances = [math.dist(pos, centroid) for pos in positions]
    return np.min(distances), np.average(distances), np.max(distances)

            






