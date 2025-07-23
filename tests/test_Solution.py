import solution.solution_optimized as solution
from scipy.spatial.distance import squareform
# Global imports
import numpy as np
import itertools

DISTANCES_SMALL = np.array([
    [0.0, 0.1, 0.3],
    [0.1, 0.0, 0.5],
    [0.3, 0.5, 0.0]
], dtype=np.float64)
CLUSTERS_SMALL = np.array([
    0, 1, 2
], dtype=np.int64)

DISTANCE_MEDIUM = np.array([
    [0.0, 0.9, 1.0, 0.7, 0.6, 0.5],
    [0.9, 0.0, 0.4, 0.3, 0.2, 0.1],
    [1.0, 0.4, 0.0, 0.5, 0.6, 0.7],
    [0.7, 0.3, 0.5, 0.0, 0.1, 0.2],
    [0.6, 0.2, 0.6, 0.1, 0.0, 0.3],
    [0.5, 0.1, 0.7, 0.2, 0.3, 0.0]
], dtype=np.float64)
CLUSTERS_MEDIUM_1 = np.array([
    0, 0, 0, 1, 1, 1
], dtype=np.int64)
CLUSTERS_MEDIUM_2 = np.array([
    0, 0, 1, 1, 2, 2
], dtype=np.int64)

def test_selection_per_cluster_small():
    selection = np.array([True, True, True], dtype=bool)
    solution_object = solution.Solution(DISTANCES_SMALL, CLUSTERS_SMALL, selection=selection)

    expected_selection = {
        0: {0},
        1: {1},
        2: {2}
    }
    expected_nonselection = {
        0: set(),
        1: set(),
        2: set()
    }

    actual_selection = solution_object.selection_per_cluster
    actual_nonselection = solution_object.nonselection_per_cluster

    assert actual_selection == expected_selection
    assert actual_nonselection == expected_nonselection

def test_selection_per_cluster_medium_1():
    selection = np.array([True, False, True, False, True, False], dtype=bool)
    solution_object = solution.Solution(DISTANCE_MEDIUM, CLUSTERS_MEDIUM_1, selection=selection)

    expected_selection = {
        0: {0, 2},
        1: {4}
    }
    expected_nonselection = {
        0: {1},
        1: {3, 5}
    }

    actual_selection = solution_object.selection_per_cluster
    actual_nonselection = solution_object.nonselection_per_cluster

    assert actual_selection == expected_selection
    assert actual_nonselection == expected_nonselection

def test_infeasible_solution_small():
    selection = np.array([False, True, True], dtype=bool) #infeasible solution since not every cluster is selected
    solution_object = solution.Solution(DISTANCES_SMALL, CLUSTERS_SMALL, selection=selection)

    assert not solution_object.determine_feasibility()


# This function calculates the total objective function from scratch, as well as its components for a given solution.
def calculate_objective(selection, distances, clusters, cost_per_cluster):
    try:
        len(cost_per_cluster)
    except TypeError:
        cost_per_cluster = np.array([cost_per_cluster] * len(np.unique(clusters)), dtype=np.float64)

    objective = 0.0
    components = {
        "selection": 0.0,
        "intra": 0.0,
        "inter": 0.0,
    }

    # Assign cost for selecting
    for idx in np.where(selection)[0]:
        components["selection"] += cost_per_cluster[clusters[idx]]
        objective += cost_per_cluster[clusters[idx]]

    # Intra cluster costs
    for cluster in np.unique(clusters):
        indices_selected = np.where(selection & (clusters == cluster))[0]
        indices_nonselected = np.where(~selection & (clusters == cluster))[0]

        for idx in indices_nonselected:
            intra_cost = np.inf
            for selected_idx in indices_selected:
                intra_cost = min(intra_cost, distances[idx, selected_idx])
            components["intra"] += intra_cost
            objective += intra_cost

    # Inter cluster costs
    for cluster1, cluster2 in itertools.combinations(np.unique(clusters), 2):
        indices1 = np.where(selection & (clusters == cluster1))[0]
        indices2 = np.where(selection & (clusters == cluster2))[0]

        inter_cost = -np.inf
        for idx1 in indices1:
            for idx2 in indices2:
                inter_cost = max(inter_cost, 1.0 - distances[idx1, idx2])
        components["inter"] += inter_cost
        objective += inter_cost

    return objective, components