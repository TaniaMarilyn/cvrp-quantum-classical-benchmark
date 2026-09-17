"""
FILE 1 (ROBUST VERSION): load_from_single_sheet.py

Automatically detects which layout your Excel file actually uses:
  - Layout A: separate sheets named "Customers", "VehicleTypes", "Parameters"
  - Layout B: one "Customers" sheet with Vehicle Types as a side-table
Works with either, so it doesn't matter which version of the file you have.
"""

import pandas as pd
from scipy.spatial import distance_matrix as scipy_distance_matrix


def load_single_sheet_dataset(filepath="cvrp_500_customers_single_sheet.xlsx"):
    xl = pd.ExcelFile(filepath)
    print("Sheets found in this file:", xl.sheet_names)

    # ---------- LAYOUT A: separate VehicleTypes sheet exists ----------
    if "VehicleTypes" in xl.sheet_names:
        print("Detected layout: separate sheets (Customers + VehicleTypes)")
        customers_df = pd.read_excel(xl, sheet_name="Customers")
        vehicle_types_df = pd.read_excel(xl, sheet_name="VehicleTypes")

    # ---------- LAYOUT B: single sheet with side-tables ----------
    else:
        print("Detected layout: single sheet with side-tables")
        raw = pd.read_excel(xl, sheet_name="Customers", header=None)

        header_row = raw.iloc[0]
        customer_col_count = 0
        for col_idx in range(raw.shape[1]):
            if pd.isna(header_row[col_idx]):
                break
            customer_col_count += 1

        customers_df = raw.iloc[1:502, 0:customer_col_count].copy()
        customers_df.columns = header_row[0:customer_col_count].tolist()
        customers_df = customers_df.reset_index(drop=True)

        # find "VEHICLE TYPES" label anywhere in the sheet (case-insensitive)
        vt_label_row, vt_label_col = None, None
        for r in range(raw.shape[0]):
            for c in range(raw.shape[1]):
                cell_val = str(raw.iat[r, c]).strip().upper()
                if "VEHICLE TYPE" in cell_val:
                    vt_label_row, vt_label_col = r, c
                    break
            if vt_label_row is not None:
                break

        if vt_label_row is None:
            raise ValueError(
                "Could not find a 'VehicleTypes' sheet OR a 'VEHICLE TYPES' label.\n"
                "Sheets in your file: " + str(xl.sheet_names) + "\n"
                "Please check your file structure, or tell me exactly what sheets/labels it has."
            )

        vt_header_row = vt_label_row + 1
        vehicle_types_df = raw.iloc[vt_header_row + 1: vt_header_row + 4,
                                      vt_label_col: vt_label_col + 5].copy()
        vehicle_types_df.columns = raw.iloc[vt_header_row, vt_label_col: vt_label_col + 5].tolist()
        vehicle_types_df = vehicle_types_df.reset_index(drop=True)

    # ---------- Normalize column names and types (works for either layout) ----------
    vehicle_types_df.columns = ["vehicle_type", "capacity", "cost_per_km", "count_available", "avg_speed_kmph"]
    for col in ["capacity", "cost_per_km", "count_available", "avg_speed_kmph"]:
        vehicle_types_df[col] = pd.to_numeric(vehicle_types_df[col], errors="coerce")

    for col in ["customer_id", "x_coord", "y_coord", "demand",
                "time_window_start", "time_window_end", "service_time"]:
        if col in customers_df.columns:
            customers_df[col] = pd.to_numeric(customers_df[col], errors="coerce")

    coords_array = customers_df[["x_coord", "y_coord"]].to_numpy()
    dist_matrix = scipy_distance_matrix(coords_array, coords_array)

    demands = dict(zip(customers_df["customer_id"], customers_df["demand"]))
    window_start = dict(zip(customers_df["customer_id"], customers_df["time_window_start"]))
    window_end = dict(zip(customers_df["customer_id"], customers_df["time_window_end"]))
    service_time = dict(zip(customers_df["customer_id"], customers_df["service_time"]))
    vehicle_types = vehicle_types_df.to_dict("records")

    return dist_matrix, demands, vehicle_types, customers_df, window_start, window_end, service_time


if __name__ == "__main__":
    dist_matrix, demands, vehicle_types, customers_df, window_start, window_end, service_time = load_single_sheet_dataset()
    print("\nCustomer table columns found:", customers_df.columns.tolist())
    print("Customers loaded (excluding depot):", len(demands) - 1)
    print("Distance matrix shape:", dist_matrix.shape)
    print("\nVehicle types:")
    for vt in vehicle_types:
        print(" ", vt)