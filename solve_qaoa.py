"""
QUANTUM STEP 3: solve_qaoa.py
Solves the CVRP QUBO (built from real customers extracted from your
Excel file) using QAOA on a noiseless quantum simulator.
"""
from qiskit.primitives import Sampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from cvrp_to_qubo import build_cvrp_qubo


def solve_with_qaoa(filepath="cvrp_500_customers_single_sheet.xlsx",
                     n_customers=3, capacity=50, K=2, reps=1, maxiter=50):
    qp, qubo, dist_matrix, demands, node_ids = build_cvrp_qubo(filepath, n_customers, capacity, K)

    qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA(maxiter=maxiter), reps=reps)
    solver = MinimumEigenOptimizer(qaoa)
    result = solver.solve(qubo)

    return result, qp, qubo, dist_matrix, demands, node_ids


def decode_solution(result, qp):
    var_names = [v.name for v in qp.variables]
    used_edges = []
    for name, val in zip(var_names, result.x):
        if val > 0.5 and name.startswith("x_"):
            _, i, j = name.split("_")
            used_edges.append((int(i), int(j)))
    return used_edges


if __name__ == "__main__":
    result, qp, qubo, dist_matrix, demands, node_ids = solve_with_qaoa(
        "cvrp_500_customers_single_sheet.xlsx",
        n_customers=3, capacity=50, K=2, reps=1, maxiter=50
    )

    print("=== QAOA Result (real data, noiseless simulator) ===")
    print("Qubits used:", qubo.get_num_binary_vars())
    print("Objective value (total distance):", round(result.fval, 2))
    print("Status:", result.status.name)

    used_edges = decode_solution(result, qp)
    print("\nRoute edges used:")
    for (i, j) in used_edges:
        print(f"  {i} -> {j}   (distance: {dist_matrix[i][j]:.2f})")