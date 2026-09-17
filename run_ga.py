"""
FILE 3: run_ga.py

Genetic Algorithm using DEAP. Keeps a POPULATION of candidate customer
orderings, breeds the best ones together, mutates some randomly, and
repeats over many generations - gradually converging on a low-cost route.
"""

import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms

from load_from_single_sheet import load_single_sheet_dataset
from vrptw_eval import (
    evaluate_route_cost_vrptw_deap,
    split_into_vehicles_with_types,
    evaluate_trip_with_time_windows,
)


def build_ga_toolbox(dist_matrix, demands, vehicle_types, window_start, window_end, service_time, n_customers):
    # Avoid DEAP's "already exists" error if this runs more than once in one session
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # -1.0 = MINIMIZE cost
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    # individuals are 0-indexed permutations (0..n_customers-1) - required by cxOrdered
    toolbox.register("indices", np.random.permutation, n_customers)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxOrdered)                       # ordered crossover - keeps valid permutations
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.02) # small chance each position gets shuffled
    toolbox.register("select", tools.selTournament, tournsize=3)    # pick parents via mini "tournaments"
    toolbox.register(
        "evaluate", evaluate_route_cost_vrptw_deap,
        dist_matrix=dist_matrix, demands=demands, vehicle_types=vehicle_types,
        window_start=window_start, window_end=window_end, service_time=service_time,
    )
    return toolbox


def run_ga(filepath, pop_size=80, generations=100, cxpb=0.7, mutpb=0.2, verbose=True):
    dist_matrix, demands, vehicle_types, customers_df, window_start, window_end, service_time = \
        load_single_sheet_dataset(filepath)
    n_customers = len(demands) - 1  # exclude depot

    toolbox = build_ga_toolbox(dist_matrix, demands, vehicle_types, window_start, window_end, service_time, n_customers)
    population = toolbox.population(n=pop_size)

    hall_of_fame = tools.HallOfFame(1)   # remembers the single best solution ever seen
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    population, logbook = algorithms.eaSimple(
        population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=generations,
        stats=stats, halloffame=hall_of_fame, verbose=verbose,
    )

    best_individual = hall_of_fame[0]
    best_cost = best_individual.fitness.values[0]
    actual_order = [c + 1 for c in best_individual]  # convert back to real customer IDs
    best_segments = split_into_vehicles_with_types(actual_order, demands, vehicle_types)

    return actual_order, best_cost, best_segments, dist_matrix, window_start, window_end, service_time


def save_results(segments, dist_matrix, window_start, window_end, service_time, out_path="ga_results.csv"):
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
    order, cost, segments, dist_matrix, ws, we, st = run_ga(
        "cvrp_500_customers_single_sheet.xlsx", pop_size=80, generations=100, verbose=True
    )

    print("\n=== GA FINAL RESULT ===")
    print("Total cost (distance + lateness penalty):", round(cost, 2))
    print("Number of vehicle trips:", len(segments))

    rows = save_results(segments, dist_matrix, ws, we, st, "ga_results.csv")
    print("Saved trip-level results to ga_results.csv")
