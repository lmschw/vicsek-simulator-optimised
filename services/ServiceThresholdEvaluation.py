import numpy as np

from enums.EnumThresholdEvaluationMethod import ThresholdEvaluationMethod

import services.ServiceMetric as ServiceMetric
import services.ServiceVicsekHelper as ServiceVicsekHelper

def getThresholdEvaluationValuesForChoice(thresholdEvaluationMethod, positions, orientations, neighbours, domainSize, neighbourSubsetSize=None):
    match thresholdEvaluationMethod:
        case ThresholdEvaluationMethod.LOCAL_ORDER:
            return computeLocalOrders(orientations=orientations, neighbours=neighbours)
        case ThresholdEvaluationMethod.LOCAL_ORDER_RANDOM_SUBSET:
            return computeLocalOrdersFromRandomSubset(orientations=orientations, neighbours=neighbours, subsetSize=neighbourSubsetSize)
        case ThresholdEvaluationMethod.ORIENTATION_DIFFERENCE_AVG:
            return computeNormalisedAverageOrientationDifferences(orientations=orientations, neighbours=neighbours, domainSize=domainSize)
        case ThresholdEvaluationMethod.ORIENTATION_DIFFERENCE_MIN: 
            return computeNormalisedMinOrientationDifferences(orientations=orientations, neighbours=neighbours, domainSize=domainSize)
        case ThresholdEvaluationMethod.ORIENTATION_DIFFERENCE_MAX:
            return computeNormalisedMaxOrientationDifferences(orientations=orientations, neighbours=neighbours, domainSize=domainSize)
        case ThresholdEvaluationMethod.NEIGHBOUR_DISTANCE_AVG:
            return computeNormalisedAverageNeighbourDistances(positions=positions, neighbours=neighbours, domainSize=domainSize)
        case ThresholdEvaluationMethod.NEIGHBOUR_DISTANCE_MIN:
            return computeNormalisedMinimumNeighbourDistances(positions=positions, neighbours=neighbours, domainSize=domainSize)
        case ThresholdEvaluationMethod.NEIGHBOUR_DISTANCE_MAX:
            return computeNormalisedMaximumNeighbourDistances(positions=positions, neighbours=neighbours, domainSize=domainSize)

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

def computeLocalOrdersFromRandomSubset(orientations, neighbours, subsetSize):
    """
    Same as computeLocalOrders(), but every individual's local order is computed from a fresh random
    subset of at most subsetSize of its in-radius neighbours (redrawn every time this is called)
    instead of from all of them. The individual itself is always included, exactly as in
    computeLocalOrders().

    Params:
        - orientations (array of floats): the orientation of every individual at the current timestep
        - neighbours (array of arrays of booleans): the identity of every neighbour of every individual (including itself)
        - subsetSize (int): the maximum number of other neighbours to randomly sample per individual

    Returns:
        An array of floats representing the local order for every individual at the current time step (values between 0 and 1)
    """
    otherNeighbours = neighbours.copy()
    np.fill_diagonal(otherNeighbours, False)

    # rank every candidate neighbour by a fresh random score per individual and keep the top subsetSize.
    # non-neighbours are pinned to a sentinel score below the [0, 1) range so they always sort last and
    # can be filtered back out afterwards - this selects a random subset without a per-particle loop.
    randomScores = np.where(otherNeighbours, np.random.random(otherNeighbours.shape), -1.0)
    rankedIndices = np.argsort(-randomScores, axis=1)[:, :subsetSize]
    rows = np.arange(neighbours.shape[0])[:, None]

    sampledNeighbours = np.zeros_like(neighbours)
    sampledNeighbours[rows, rankedIndices] = randomScores[rows, rankedIndices] > -1.0
    np.fill_diagonal(sampledNeighbours, True)
    return computeLocalOrders(orientations=orientations, neighbours=sampledNeighbours)

def computeNormalisedOrientationDifferences(orientations, neighbours, domainSize):
    orientationDifferences = ServiceVicsekHelper.getOrientationDifferences(orientations, domainSize)
    neighbourDifferences = orientationDifferences * neighbours
    return neighbourDifferences / np.max(neighbourDifferences, axis=1)[:,None]

def computeNormalisedAverageOrientationDifferences(orientations, neighbours, domainSize):
    sumDifferences = computeNormalisedOrientationDifferences(orientations, neighbours, domainSize)
    return np.average(sumDifferences, axis=1)

def computeNormalisedMinOrientationDifferences(orientations, neighbours, domainSize):
    np.fill_diagonal(neighbours, False) # otherwise, the min will always be zero
    sumDifferences = computeNormalisedOrientationDifferences(orientations, neighbours, domainSize)
    return np.min(sumDifferences, axis=1)

def computeNormalisedMaxOrientationDifferences(orientations, neighbours, domainSize):
    sumDifferences = computeNormalisedOrientationDifferences(orientations, neighbours, domainSize)
    return np.max(sumDifferences, axis=1)

def computeNormalisedNeighbourDistances(positions, neighbours, domainSize):
    dists = ServiceMetric.getDistanceOfNeighbours(positions, neighbours, domainSize)
    return dists / np.max(dists, axis=1)[:,None]

def computeNormalisedAverageNeighbourDistances(positions, neighbours, domainSize):
    return np.average(computeNormalisedNeighbourDistances(positions, neighbours, domainSize), axis=1)

def computeNormalisedMinimumNeighbourDistances(positions, neighbours, domainSize):
    return np.min(computeNormalisedNeighbourDistances(positions, neighbours, domainSize), axis=1)

def computeNormalisedMaximumNeighbourDistances(positions, neighbours, domainSize):
    return np.max(computeNormalisedNeighbourDistances(positions, neighbours, domainSize), axis=1)