import solution.solution_naive as solution_naive
# Global imports
import numpy as np
import itertools

def test_initialize_solution_empty():
    distances = np.array([[0, 0.1, 0.3], [0.1, 0, 0.5], [0.3, 0.5, 0]], dtype=np.float32)
    clusters = np.array([0, 1, 2], dtype=np.int32)

    selection = np.array([False, False, False], dtype=bool)
    solution_object = solution_naive.Solution(distances, clusters)

    # Check if distances are correctly copied
    np.testing.assert_array_equal(solution_object.distances, distances)
    # Check if clusters are correctly copied
    np.testing.assert_array_equal(solution_object.clusters, clusters)
    # Check if selection is correctly initiated (no solution -> everything should be False)
    np.testing.assert_array_equal(solution_object.selection, selection)

def test_initalize_solution_nonempty():
    distances = np.array([[0, 0.1, 0.3], [0.1, 0, 0.5], [0.3, 0.5, 0]], dtype=np.float32)
    clusters = np.array([0, 1, 2], dtype=np.int32)

    selection = np.array([False, True, True], dtype=bool)
    solution_object = solution_naive.Solution(distances, clusters, selection=selection)

    # Check if distances are correctly copied
    np.testing.assert_array_equal(solution_object.distances, distances)
    # Check if clusters are correctly copied
    np.testing.assert_array_equal(solution_object.clusters, clusters)
    # Check if selection is correctly initiated (no solution -> everything should be False)
    np.testing.assert_array_equal(solution_object.selection, selection)

def test_calculate_objective_value():
    distances = np.array([[0, 0.1, 0.3], [0.1, 0, 0.5], [0.3, 0.5, 0]], dtype=np.float32)
    clusters = np.array([0, 1, 2], dtype=np.int32)

    selection = np.array([True, True, True], dtype=bool)
    selection_cost = 0.1

    # Calculate the expected objective value
    expected_objective_value = groundtruth_objective_value(selection, clusters, distances, selection_cost)

    # Create a solution object
    solution_object = solution_naive.Solution(distances, clusters, selection=selection, selection_cost=selection_cost)

    # Check if the calculated objective value matches the expected value
    assert solution_object.objective_value == expected_objective_value



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
            cur_max = max(cur_max, distances[point_pair[0], point_pair[1]])
        objective_value += cur_max
    return objective_value