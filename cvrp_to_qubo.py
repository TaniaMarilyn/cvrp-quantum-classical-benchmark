"""
QUANTUM STEP 2: cvrp_to_qubo.py (uses REAL data extracted from your Excel file)
"""
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

from extract_subset_from_excel import extract_small_subset
from distance_matrix import build_distance_matrix


def build_cvrp_qubo(filepath="cvrp_500_customers_single_sheet.xlsx",
                     n_customers=3, capacity=50, K=2):
    coords, demands, capacity = extract_small_subset(filepath, n_customers, capacity)
    dist_matrix, node_ids = build_distance_matrix(coords)
    n = len(node_ids)

    qp = QuadraticProgram("CVRP")

    for i in range(n):
        for j in range(n):
            if i != j:
                qp.binary_var(name=f"x_{i}_{j}")

    linear_terms = {f"x_{i}_{j}": dist_matrix[i][j]
                     for i in range(n) for j in range(n) if i != j}
    qp.minimize(linear=linear_terms)

    for i in range(1, n):
        coeffs = {f"x_{i}_{j}": 1 for j in range(n) if j != i}
        qp.linear_constraint(linear=coeffs, sense="==", rhs=1, name=f"out_{i}")

    for j in range(1, n):
        coeffs = {f"x_{i}_{j}": 1 for i in range(n) if i != j}
        qp.linear_constraint(linear=coeffs, sense="==", rhs=1, name=f"in_{j}")

    depot_out = {f"x_{0}_{j}": 1 for j in range(1, n)}
    qp.linear_constraint(linear=depot_out, sense="==", rhs=K, name="depot_out")
    depot_in = {f"x_{i}_{0}": 1 for i in range(1, n)}
    qp.linear_constraint(linear=depot_in, sense="==", rhs=K, name="depot_in")

    converter = QuadraticProgramToQubo()
    qubo = converter.convert(qp)

    return qp, qubo, dist_matrix, demands, node_ids


if __name__ == "__main__":
    qp, qubo, dist_matrix, demands, node_ids = build_cvrp_qubo(
        "cvrp_500_customers_single_sheet.xlsx", n_customers=3, capacity=50, K=2
    )
    print("Decision variables:", qp.get_num_vars())
    print("Constraints:", len(qp.linear_constraints))
    print("Qubits needed:", qubo.get_num_binary_vars())