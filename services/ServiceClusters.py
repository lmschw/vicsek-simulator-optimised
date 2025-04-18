import numpy as np
import operator, random
from sklearn.cluster import AgglomerativeClustering

import services.ServiceVicsekHelper as ServiceVicsekHelper
import services.ServiceOrientations as ServiceOrientations

def getClusterHistory(positions, orientations, domainSize, radius,  threshold=0.01, use_agglomerative_clustering=True):
    cluster_number_history = []
    cluster_history = []
    for t in range(len(orientations)):
        headings = ServiceOrientations.computeAnglesForOrientations(orientations[t])
        if use_agglomerative_clustering:
            n_clusters, clusters = findClusters(orientations[t], threshold)
        else:
            n_clusters, clusters = findClustersWithRadius(positions[t], orientations[t], domainSize, radius, threshold)
        cluster_number_history.append(n_clusters)
        if t == 0:
            cluster_history.append(clusters)
            headings_clusters_old = compute_average_orientations_for_all_clusters(headings, clusters, n_clusters=n_clusters)
            continue
        headings_clusters = compute_average_orientations_for_all_clusters(headings, clusters, n_clusters=n_clusters)
        new_clusters = match_clusters(clusters, headings_clusters, cluster_history[-1], headings_clusters_old)
        cluster_history.append(new_clusters)
        headings_clusters_old = headings_clusters
    return cluster_history, cluster_number_history

def match_clusters(clusters, cluster_headings, clusters_old, cluster_headings_old, cluster_identity_percentage_cutoff=0.51):
    distances = compute_angle_distances(cluster_headings=cluster_headings, cluster_headings_old=cluster_headings_old)
    clusters_new = np.full(clusters.shape, -1)
    unique_clusters, counts_clusters = np.unique(clusters, return_counts=True)
    unique_clusters_old, counts_clusters_old = np.unique(clusters_old, return_counts=True)
    
    cluster_counter = -2

    cluster_memberships_for_old, cluster_memberships_for_new = compute_common_cluster_membership(clusters=clusters, old_clusters=clusters_old)

    for agt_id in range(len(clusters)):
        new_cluster_id = np.inf

        cluster_id = clusters[agt_id]
        cluster_old_id = clusters_old[agt_id]

        # if the agent has already been assigned to a cluster, we skip it
        if clusters_new[agt_id] != -1:
            continue
            
        unique_clusters_idx = np.argwhere(unique_clusters == cluster_id)[0][0]
        unique_clusters_old_idx = np.argwhere(unique_clusters_old == cluster_old_id)[0][0]
        max_kv = max(cluster_memberships_for_new[cluster_id].items(), key=operator.itemgetter(1))

        # if the old cluster only had a single member and the new cluster only has a single member, we use the old ID for continuance
        if counts_clusters[unique_clusters_idx] == 1 and counts_clusters_old[unique_clusters_old_idx] == 1:
            new_cluster_id = cluster_old_id 

        # if the new cluster only contains a single element, we mark it tentatively as a new cluster
        elif counts_clusters[unique_clusters_idx] == 1:
            new_cluster_id = cluster_counter
            cluster_counter -= 1
        
        # if the new cluster consists primarily of the members of a previous cluster, we keep the cluster ID of that cluster
        elif max_kv[1]/counts_clusters[unique_clusters_idx] > cluster_identity_percentage_cutoff:
            new_cluster_id = max_kv[0]
        
        else:
            # if there is no clear winner, we look for the cluster with the lowest orientation difference with a common member that has not yet been assigned
            dists_sorted_indices = np.argsort(distances.T[cluster_id])
            for i in dists_sorted_indices:
                if clusters_old[i] not in clusters_new:
                    new_cluster_id = i
                    break
            # else we assign a new cluster
            if new_cluster_id == np.inf:
                new_cluster_id = cluster_counter
                cluster_counter -= 1
        clusters_new = np.where(clusters == cluster_id, new_cluster_id, clusters_new)

    for ci in range(len(clusters_new)):
        if clusters_new[ci] < 0:
            if clusters_old[ci] not in clusters_new:
                new_cluster_id = clusters_old[ci]
            else:
                new_cluster_id = max(max(clusters), max(clusters_old)) + 1
            clusters_new = np.where(clusters == cluster_id, new_cluster_id, clusters_new)
    return clusters_new

def compute_common_cluster_membership(clusters, old_clusters):
    """
    compute how many members of a cluster have been in the same cluster previously
    """
    cluster_memberships_for_new = {cl_id: {cl_old_id: 0 for cl_old_id in old_clusters} for cl_id in clusters}
    cluster_memberships_for_old = {cl_old_id: {cl_id: 0 for cl_id in clusters} for cl_old_id in old_clusters}

    for i in np.unique(clusters):
        for j in np.unique(old_clusters):
            old_members = np.argwhere(old_clusters == j)
            new_members = np.argwhere(clusters == i)
            intersect = np.intersect1d(old_members, new_members)
            cluster_memberships_for_new[i][j] = len(intersect)
            cluster_memberships_for_old[j][i] = len(intersect)
    return cluster_memberships_for_old, cluster_memberships_for_new

def compute_angle_distances(cluster_headings, cluster_headings_old):
    xx1, xx2 = np.meshgrid(cluster_headings, cluster_headings_old)
    x_diffs = xx1 - xx2
    distances = np.sqrt(np.multiply(x_diffs, x_diffs))  
    return distances

def compute_average_orientations_for_all_clusters(orientations, clusters, n_clusters):
    orients = []
    for c in range(n_clusters):
        oris = orientations[np.argwhere(clusters == c)]
        orients.append(np.average(oris))
    return np.array(orients)

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


def transform_cluster_history_into_colour_history(cluster_history):
    colours = {}
    colour_history = []
    for t in range(len(cluster_history)):
        colours_t = []
        for c in cluster_history[t]:
            if c in colours:
                colour = colours[c]
            else:
                colour = (random.randrange(255), random.randrange(255), random.randrange(255))
                colours[c] = colour
            colours_t.append(colour)
        colour_history.append(colours_t)
    return colour_history
