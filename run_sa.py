"""
FILE 4: run_sa.py

Simulated Annealing - a completely different search strategy from GA,
but scored with the EXACT SAME evaluate_route_cost_vrptw() function,
so its results are directly comparable to the GA's.

How it works: start with one random route. Repeatedly try small tweaks
(swap two customers). Always accept improvements. Sometimes accept a
WORSE move too (probability controlled by 'temperature') so it can escape
bad local solutions early on. As temperature cools, it becomes stricter,
settling into a good final answer.
"""

import random
import numpy as np
import pandas as pd

from load_from_single_sheet import load_single_sheet_dataset
from vrptw_eval import (
    evaluate_route_cost_vrptw,
    split_into_vehicles_with_types,
    evaluate_trip_with_time_windows,
)


def simulated_annealing(dist_matrix, demands, vehicle_types, window_start, window_end, service_time,
                          n_customers, initial_temp=2000, cooling_rate=0.997, iterations=8000, verbose=True):

    current = list(np.random.permutation(range(1, n_customers + 1)))  # real customer IDs, 1..n
    current_cost = evaluate_route_cost_vrptw(current, dist_matrix, demands, vehicle_types,
                                               window_start, window_end, service_time)
    best, best_cost = current[:], current_cost
    temp = initial_temp

    for step in range(iterations):
        neighbor = current[:]
        i, j = random.sample(range(len(neighbor)), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]  # swap two customers

        neighbor_cost = evaluate_route_cost_vrptw(neighbor, dist_matrix, demands, vehicle_types,
                                                     window_start, window_end, service_time)

        delta = neighbor_cost - current_cost
        # always accept improvements; sometimes accept worse moves (escapes local minima)
        if delta < 0 or random.random() < np.exp(-delta / max(temp, 1e-9)):
            current, current_cost = neighbor, neighbor_cost
            if current_cost < best_cost:
                best, best_cost = current[:], current_cost

        temp *= cooling_rate

        if verbose and step % 1000 == 0:
            print(f"  step {step:5d}  temp={temp:8.2f}  current_cost={current_cost:12.2f}  best_cost={best_cost:12.2f}")

    return best, best_cost


def run_sa(filepath, initial_temp=2000, cooling_rate=0.997, iterations=8000, verbose=True):
    dist_matrix, demands, vehicle_types, customers_df, window_start, window_end, service_time = \
        load_single_sheet_dataset(filepath)
    n_customers = len(demands) - 1

    best_order, best_cost = simulated_annealing(
        dist_matrix, demands, vehicle_types, window_start, window_end, service_time,
        n_customers, initial_temp, cooling_rate, iterations, verbose
    )

    best_segments = split_into_vehicles_with_types(best_order, demands, vehicle_types)
    return best_order, best_cost, best_segments, dist_matrix, window_start, window_end, service_time


def save_results(segments, dist_matrix, window_start, window_end, service_time, out_path="sa_results.csv"):
    rows = []
    for i, (seg, vtype) in enumerate(segments, start=1):
        dist_cost, lateness = evaluate_trip_with_time_windows(seg, vtype, dist_matrix, window_start, window_end, service_time)
        rows.append({
            "trip_id": i,
            "vehicle_type": vtype["vehicle_type"],
            "num_stops": len(seg),
            "distance_cost": round(dist_cost, 2),
            "lateness_minutes": round(lateness, 2),
            "customers": ",".join(map(str, seg)),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return rows


if __name__ == "__main__":
    order, cost, segments, dist_matrix, ws, we, st = run_sa(
        "cvrp_500_customers_single_sheet.xlsx",
        initial_temp=2000, cooling_rate=0.997, iterations=8000, verbose=True
    )

    print("\n=== SA FINAL RESULT ===")
    print("Total cost (distance + lateness penalty):", round(cost, 2))
    print("Number of vehicle trips:", len(segments))

    rows = save_results(segments, dist_matrix, ws, we, st, "sa_results.csv")
    print("Saved trip-level results to sa_results.csv")
