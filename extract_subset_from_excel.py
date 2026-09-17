"""
QUANTUM STEP 1: extract_subset_from_excel.py

Pulls a SMALL subset of REAL customers (from your actual 500-customer
Excel file) for quantum processing - quantum simulators can only handle
a handful of customers (proven working: 3 customers = 12 qubits).

This keeps quantum and classical grounded in the SAME source dataset,
rather than using separate synthetic data for each.
"""
import pandas as pd


def extract_small_subset(filepath="cvrp_500_customers_single_sheet.xlsx",
                           n_customers=3, capacity_override=50):
    xl = pd.ExcelFile(filepath)

    if "VehicleTypes" in xl.sheet_names:
        customers_df = pd.read_excel(xl, sheet_name="Customers")
    else:
        raw = pd.read_excel(xl, sheet_name="Customers", header=None)
        header_row = raw.iloc[0]
        col_count = 0
        for c in range(raw.shape[1]):
            if pd.isna(header_row[c]):
                break
            col_count += 1
        customers_df = raw.iloc[1:502, 0:col_count].copy()
        customers_df.columns = header_row[0:col_count].tolist()
        customers_df = customers_df.reset_index(drop=True)

    # Row 0 = depot, then take the first n_customers real customers
    depot_row = customers_df.iloc[0]
    subset = customers_df.iloc[1:n_customers + 1]

    coords = {0: (float(depot_row["x_coord"]), float(depot_row["y_coord"]))}
    demands = {0: 0}
    for new_id, (_, row) in enumerate(subset.iterrows(), start=1):
        coords[new_id] = (float(row["x_coord"]), float(row["y_coord"]))
        # scale demand down if needed so it fits a small test capacity
        demands[new_id] = min(int(row["demand"]), capacity_override // 2)

    return coords, demands, capacity_override


if __name__ == "__main__":
    coords, demands, capacity = extract_small_subset(
        "cvrp_500_customers_single_sheet.xlsx", n_customers=3, capacity_override=50
    )
    print("Coordinates (real data from your Excel file):")
    for k, v in coords.items():
        print(f"  Node {k}: {v}")
    print("\nDemands:", demands)
    print("Capacity:", capacity)