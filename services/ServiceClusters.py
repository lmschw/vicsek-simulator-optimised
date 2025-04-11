import numpy as np
from sklearn.cluster import AgglomerativeClustering

import services.ServiceVicsekHelper as ServiceVicsekHelper

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

def findClustersWithRadius(positions, orientations, domainSize, radius, threshold=0.01):
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
    # TODO refactor
    if radius == None:
        print("ERROR: Radius needs to be provided for clustering.")

    n = len(positions)
    clusters = np.full(n, -1)

    neighbours = ServiceVicsekHelper.getNeighbours(positions=positions, domainSize=domainSize, radius=radius)

    clusterNumber, baseClusters = findClusters(orientations, threshold)
    neighbourIndices = np.argwhere(neighbours)
    neighbourIndicesRegrouped = {}
    maxClusterId = clusterNumber -1
    for indexPair in neighbourIndices:
        if indexPair[0] in neighbourIndicesRegrouped.keys():
            neighbourIndicesRegrouped[indexPair[0]].append(indexPair[1])
        else:
            neighbourIndicesRegrouped[indexPair[0]] = [indexPair[1]]

    for c in range(clusterNumber):
        members = np.where(baseClusters==c)
        for member in members[0]:
            maxClusterId = updateClusters(member, c, maxClusterId, clusters, members[0], neighbourIndicesRegrouped)
        

    return len(np.unique(clusters)), clusters


def updateClusters(currentIdx, clusterId, maxClusterId, clusters, candidates, neighbourIndices):
    # if the currentIdx is already assigned a value, we return
    if clusters[currentIdx] != -1:
        return maxClusterId
    # the first member is always automatically ok
    if currentIdx == candidates[0]:
        clusters[currentIdx] = clusterId
    else:
        # if any of the neighbours are also members of the same cluster, we set the clusterId
        for neighbour in neighbourIndices[currentIdx]:
            if clusters[neighbour] != -1 and neighbour in candidates:
                clusters[currentIdx] = clusters[neighbour]
        # if not, we set a new clusterId
        if clusters[currentIdx] == -1:
            maxClusterId += 1
            clusters[currentIdx] = maxClusterId
    return maxClusterId

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
