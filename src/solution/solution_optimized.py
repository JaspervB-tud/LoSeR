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

        # Initialize object attributes
        self.selection = selection.copy()
        self.distances = distances.copy()
        self.clusters = clusters.copy()
        self.unique_clusters = np.unique(self.clusters)
        self.selection_cost = selection_cost

        # Process initial representation to optimize for comparisons speed
        self.points_per_cluster = {cluster: set(np.where(self.clusters == cluster)[0]) for cluster in self.unique_clusters} #points in every cluster
        self.selection_per_cluster = {cluster: set(np.where((self.clusters == cluster) & selection)[0]) for cluster in self.unique_clusters} #selected points in every cluster
        self.nonselection_per_cluster = {cluster: set(np.where((self.clusters == cluster) & ~selection)[0]) for cluster in self.unique_clusters} #unselected points in every cluster
        # INTRA cluster information
        self.closest_distances_intra = np.zeros(self.selection.shape[0], dtype=np.float32) #distances to closest selected point
        self.closest_points_intra = np.arange(0, self.selection.shape[0], dtype=np.int32) #indices of closest selected point
        # INTER cluster information
        self.closest_distances_inter = np.zeros((self.unique_clusters.shape[0], self.unique_clusters.shape[0]), dtype=np.float32) #distances to closest selected point
        self.closest_points_inter = {pair: (None, None) for pair in itertools.combinations(self.unique_clusters, 2)} #indices of closest selected point

        self.feasible = self.determine_feasibility()

        if self.feasible:
            # Set objective value
            self.objective = np.sum(self.selection) * self.selection_cost
            # INTRA cluster distances
            for idx in np.where(~self.selection)[0]:
                cur_min = np.inf
                cur_idx = idx
                for other_idx in np.where((self.clusters == self.clusters[idx]) & self.selection)[0]:
                    cur_dist = self.distances[idx, other_idx]
                    if cur_dist < cur_min:
                        cur_min = cur_dist
                        cur_idx = other_idx
                self.closest_distances_intra[idx] = cur_min
                self.closest_points_intra[idx] = cur_idx
                self.objective += cur_min
            # INTER cluster distances
            for cluster_pair in itertools.combinations(self.unique_clusters, 2):
                cluster_1 = np.where((self.clusters == cluster_pair[0]) & self.selection)[0]
                cluster_2 = np.where((self.clusters == cluster_pair[1]) & self.selection)[0]
                cur_max = -np.inf
                cur_pair = (None, None)
                for point_pair in itertools.product(cluster_1, cluster_2):
                    cur_dist = 1 - self.distances[point_pair[0], point_pair[1]]
                    if cur_dist > cur_max:
                        cur_max = cur_dist
                        cur_pair = point_pair
                self.closest_distances_inter[cluster_pair[0], cluster_pair[1]] = cur_max
                self.closest_distances_inter[cluster_pair[1], cluster_pair[0]] = cur_max
                self.closest_points_inter[(cluster_pair[0], cluster_pair[1])] = cur_pair  # Store the first point index
                self.objective += cur_max
        else:
            # If selection provided is not feasible, don't do anything (YET)
            self.objective = np.inf
            print("The solution is infeasible, objective value is set to INF and closest distances & points are not set.")

    def determine_feasibility(self):
        uncovered_clusters = set(self.unique_clusters)
        for point in np.where(self.selection)[0]:
            uncovered_clusters.discard(self.clusters[point])
        return len(uncovered_clusters) == 0