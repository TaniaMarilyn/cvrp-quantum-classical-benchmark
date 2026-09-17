"""
QUANTUM STEP 2: distance_matrix.py
"""
import numpy as np

def build_distance_matrix(coords):
    node_ids = sorted(coords.keys())
    n = len(node_ids)
    matrix = np.zeros((n, n))
    for i, id_i in enumerate(node_ids):
        for j, id_j in enumerate(node_ids):
            xi, yi = coords[id_i]
            xj, yj = coords[id_j]
            matrix[i][j] = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
    return matrix, node_ids

if __name__ == "__main__":
    from small_instance import generate_small_instance
    coords, demands, capacity = generate_small_instance()
    dist_matrix, node_ids = build_distance_matrix(coords)
    print("Distance matrix:\n", dist_matrix)