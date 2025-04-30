import numpy as np
import itertools
import math

class Solution:
    def __init__(self, distances, clusters, selection=None, selection_cost=0.1):
        # Assert that distances and clusters have the same number of rows
        if distances.shape[0] != clusters.shape[0]:
            raise ValueError("Number of points is different between distances and clusters.")
        # Assert that distances are in [0,1] (i.e. distances are similarities or dissimilarities)
        if not np.all((distances >= 0) & (distances <= 1)):
            raise ValueError("Distances must be in the range [0, 1].")
        # If selection is provided, check if it meets criteria
        if selection is not None:
            # Assert that selection has the same number of points as clusters
            if selection.shape != clusters.shape:
                raise ValueError("Selection must have the same number of points as clusters.")
            # Assert that selection is a numpy array of booleans
            if not isinstance(selection, np.ndarray) or selection.dtype != bool:
                raise TypeError("Selection must be a numpy array of booleans.")
        else:
            selection = np.zeros(clusters.shape[0], dtype=bool)

        self.selection = selection.copy()
        self.distances = distances.copy()
        self.clusters = clusters.copy()
        self.unique_clusters = np.unique(self.clusters)
        self.selection_cost = selection_cost

        # Determine current objective value
        self.objective_value = np.sum(selection) * selection_cost
        # Determine intra cluster costs
        for idx in np.where(~selection)[0]:
            cur_min = np.inf
            for other_idx in np.where((self.clusters == self.clusters[idx]) & selection)[0]:
                cur_min = min(cur_min, self.distances[idx, other_idx])
            self.objective_value += cur_min
        # Determine inter cluster costs
        for cluster_pair in itertools.combinations(self.unique_clusters, 2):
            cluster_1 = np.where((self.clusters == cluster_pair[0]) & selection)[0]
            cluster_2 = np.where((self.clusters == cluster_pair[1]) & selection)[0]
            cur_max = -np.inf
            for point_pair in itertools.product(cluster_1, cluster_2):
                cur_max = max(cur_max, self.distances[point_pair[0], point_pair[1]])
            self.objective_value += cur_max