import solution.solution_optimized as solution
# Global imports
import numpy as np
import itertools

def test_infeasible_solution_1():
    distances = np.array([[0, 0.1, 0.3], [0.1, 0, 0.5], [0.3, 0.5, 0]], dtype=np.float32)
    clusters = np.array([0, 1, 2], dtype=np.int32)

    selection_1 = np.array([False, True, True], dtype=bool) # not feasible
    solution_object_1 = solution.Solution(distances, clusters, selection=selection_1)

    selection_2 = np.array([True, True, True], dtype=bool) # feasible
    solution_object_2 = solution.Solution(distances, clusters, selection=selection_2)

    # Check if infeasible solution is correctly identified
    assert not solution_object_1.determine_feasibility()
    # Check if feasible solution is correctly identified
    assert solution_object_2.determine_feasibility()

def test_infeasible_solution_2():
    distances = np.array([
        [0.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        [0.9, 0.0, 0.4, 0.3, 0.2, 0.1],
        [0.8, 0.4, 0.0, 0.5, 0.6, 0.7],
        [0.7, 0.3, 0.5, 0.0, 0.1, 0.2],
        [0.6, 0.2, 0.6, 0.1, 0.0, 0.3],
        [0.5, 0.1, 0.7, 0.2, 0.3, 0.0]
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 1, 1, 2, 2], dtype=np.int32
    )
    
    selection_1 = np.array([False, True, True, False, False, False], dtype=bool) # not feasible
    solution_object_1 = solution.Solution(distances, clusters, selection=selection_1)

    selection_2 = np.array([True, False, False, True, False, True], dtype=bool) # feasible
    solution_object_2 = solution.Solution(distances, clusters, selection=selection_2)

    # Check if infeasible solution is correctly identified
    assert not solution_object_1.determine_feasibility()
    # Check if feasible solution is correctly identified
    assert solution_object_2.determine_feasibility()

def test_solution_correctness_1():
    distances = np.array([
        [0.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        [0.9, 0.0, 0.4, 0.3, 0.2, 0.1],
        [0.8, 0.4, 0.0, 0.5, 0.6, 0.7],
        [0.7, 0.3, 0.5, 0.0, 0.1, 0.2],
        [0.6, 0.2, 0.6, 0.1, 0.0, 0.3],
        [0.5, 0.1, 0.7, 0.2, 0.3, 0.0]
    ], dtype=np.float32)
    clusters = np.array(
        [0, 0, 0, 1, 1, 1], dtype=np.int32
    )

    selection = np.array([True, False, False, True, False, False], dtype=bool)
    selection_cost = 0.1
    
    expected_closest_distances_intra = np.array([0.0, 0.9, 0.8, 0.0, 0.1, 0.2], dtype=np.float32)
    expected_closest_points_intra = np.array([0, 0, 0, 3, 3, 3], dtype=np.int32)
    expected_closest_distances_inter = np.array([
        [0.0, 0.3],
        [0.3, 0.0]
        ], dtype=np.float32)
    expected_closest_points_inter = {
        (0, 1): (0, 3)
    }

    solution_object = solution.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)
    # Check if the distances are correctly copied
    np.testing.assert_array_equal(solution_object.distances, distances)
    # Check if the clusters are correctly copied
    np.testing.assert_array_equal(solution_object.clusters, clusters)
    # Check if the selection is correctly initiated
    np.testing.assert_array_equal(solution_object.selection, selection)
    # Check if the closest distances intra are correctly calculated
    np.testing.assert_array_equal(solution_object.closest_distances_intra, expected_closest_distances_intra)
    # Check if the closest points intra are correctly calculated
    np.testing.assert_array_equal(solution_object.closest_points_intra, expected_closest_points_intra)
    # Check if the closest distances inter are correctly calculated
    np.testing.assert_array_equal(solution_object.closest_distances_inter, expected_closest_distances_inter)
    # Check if the closest points inter are correctly calculated
    for key in expected_closest_points_inter:
        assert solution_object.closest_points_inter[key] == expected_closest_points_inter[key]
    # Check if the objective value is correctly calculated
    expected_objective_value = groundtruth_objective_value(selection, clusters, distances, selection_cost)
    assert solution_object.objective == expected_objective_value

# Functions for calculating the groundtruth
def groundtruth_objective_value(selection, clusters, distances, selection_cost):
    # Cost for selecting items
    objective_value = np.sum(selection) * selection_cost
    # Intra cluster costs
    for idx in np.where(~selection)[0]:
        cur_min = np.inf
        for other_idx in np.where((clusters == clusters[idx]) & selection)[0]:
            cur_min = min(cur_min, distances[idx, other_idx])
        objective_value += cur_min
    # Inter cluster costs
    unique_clusters = np.unique(clusters)
    for cluster_pair in itertools.combinations(unique_clusters, 2):
        cluster_1 = np.where((clusters == cluster_pair[0]) & selection)[0]
        cluster_2 = np.where((clusters == cluster_pair[1]) & selection)[0]
        cur_max = -np.inf
        for point_pair in itertools.product(cluster_1, cluster_2):
            cur_max = max(cur_max, 1 - distances[point_pair[0], point_pair[1]])
        objective_value += cur_max
    return objective_value