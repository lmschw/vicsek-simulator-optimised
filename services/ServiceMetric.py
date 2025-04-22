
import numpy as np
import math
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering

from enums.EnumMetrics import Metrics
import services.ServiceClusters as ServiceClusters
import services.ServiceOrientations as ServiceOrientations
import services.ServiceVicsekHelper as ServiceVicsekHelper

"""
Service containing static methods to handle metrics.
"""

def evaluateSingleTimestep(positions, orientations, metric, domainSize=None, radius=None, threshold=0.99, switchTypeValues=None, switchType=None, switchTypeOptions=None):
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
            nClusters, _ = ServiceClusters.find_clusters(orientations, threshold)
            return nClusters
        case Metrics.CLUSTER_NUMBER_WITH_RADIUS:
            nClusters, _ = ServiceClusters.find_clusters_with_radius(positions, orientations, domainSize, radius, threshold)
            return nClusters
        case Metrics.CLUSTER_SIZE:
            nClusters, clusters = ServiceClusters.find_clusters(orientations, threshold)
            # TODO: make sure the change from array to dict is taken care of in the visualisation
            clusterSizes = ServiceClusters.compute_cluster_sizes(clusters)
            return clusterSizes
        case Metrics.ORDER_VALUE_PERCENTAGE:
            orderCount, _ = getNumbersPerSwitchTypeValue(switchTypeValues, switchType, switchTypeOptions)
            return orderCount
        case Metrics.DUAL_OVERLAY_ORDER_AND_PERCENTAGE: # not a single metric but rather overlaying two metrics in the same graph
            order = computeGlobalOrder(orientations)
            orderCount, _ = getNumbersPerSwitchTypeValue(switchTypeValues, switchType, switchTypeOptions)
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

def getNumbersPerSwitchTypeValue(switchTypeValues, switchType, switchTypeOptions):
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

def getDistanceOfNeighbours(positions, neighbours, domainSize):
    np.fill_diagonal(neighbours, False)
    posDiff = np.sqrt(ServiceVicsekHelper.getPositionDifferences(positions=positions, domainSize=domainSize))
    return neighbours*posDiff

def getMinAvgMaxDistanceOfNeighbours(positions, domainSize, radius):
    neighbours = ServiceVicsekHelper.getNeighbours(positions=positions, domainSize=domainSize, radius=radius)
    dists = getDistanceOfNeighbours(positions, neighbours, domainSize)
    return np.min(dists), np.average(dists), np.max(dists)

def getMinAvgMaxDistanceFromCentroid(positions):
    centroid = np.mean(positions, axis=0)
    distances = [math.dist(pos, centroid) for pos in positions]
    return np.min(distances), np.average(distances), np.max(distances)

            






