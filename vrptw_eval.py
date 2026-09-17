"""
FILE 2: vrptw_eval.py

Turns a customer VISIT ORDER into actual vehicle trips, then calculates
a total cost that accounts for:
  1. Distance travelled x that trip's vehicle cost-per-km
  2. A penalty for arriving at a customer AFTER their time window closes

This file has no dependency on GA or SA specifically - both solvers call
the same evaluate_route_cost_vrptw() function, which is what makes the
comparison between them fair (identical scoring rules for both).
"""

LATE_PENALTY_PER_MINUTE = 5.0  # increase this to punish lateness more heavily


def split_into_vehicles_with_types(order, demands, vehicle_types):
    """
    Step 1 of evaluation: turn a flat customer order into separate trips.
    Greedy rule: keep adding customers to the current trip until adding the
    next one would exceed the LARGEST available vehicle's capacity, then
    start a new trip.
    """
    max_capacity = max(vt["capacity"] for vt in vehicle_types)
    segments, current, load = [], [], 0

    for c in order:
        if load + demands[c] > max_capacity:
            segments.append(current)
            current, load = [], 0
        current.append(c)
        load += demands[c]
    if current:
        segments.append(current)

    # Step 2: assign the CHEAPEST vehicle type that can actually carry each trip
    assigned = []
    for seg in segments:
        seg_demand = sum(demands[c] for c in seg)
        eligible = [vt for vt in vehicle_types if vt["capacity"] >= seg_demand]
        if not eligible:
            return None  # no vehicle type is big enough -> infeasible
        best_type = min(eligible, key=lambda vt: vt["cost_per_km"])
        assigned.append((seg, best_type))

    return assigned


def evaluate_trip_with_time_windows(seg, vtype, dist_matrix, window_start, window_end, service_time):
    """
    Simulates ONE vehicle trip minute-by-minute:
    depot -> customer -> customer -> ... -> depot
    Returns (distance_cost, total_lateness_minutes).
    """
    path = [0] + seg + [0]                      # 0 = depot
    speed_km_per_min = vtype["avg_speed_kmph"] / 60.0

    distance_cost = 0.0
    total_lateness = 0.0
    current_time = 0.0                            # depot departs at t=0

    for i in range(len(path) - 1):
        dist = dist_matrix[path[i]][path[i + 1]]
        distance_cost += dist * vtype["cost_per_km"]

        travel_time = dist / speed_km_per_min if speed_km_per_min > 0 else 0
        current_time += travel_time

        next_node = path[i + 1]
        if next_node == 0:
            continue  # arriving back at depot - no time window to check

        w_start, w_end = window_start[next_node], window_end[next_node]

        if current_time < w_start:
            current_time = w_start                 # wait for the window to open
        elif current_time > w_end:
            total_lateness += (current_time - w_end)  # arrived late

        current_time += service_time[next_node]      # time spent servicing this stop

    return distance_cost, total_lateness


def evaluate_route_cost_vrptw(order, dist_matrix, demands, vehicle_types,
                                window_start, window_end, service_time,
                                zero_indexed=False):
    """
    THE MAIN SCORING FUNCTION used by both GA and SA.

    order: a list of customer IDs in visiting order.
    zero_indexed=True means 'order' contains 0..n-1 (as DEAP's permutation
    generator produces) and needs +1 added to become real customer IDs.
    """
    real_order = [c + 1 for c in order] if zero_indexed else order

    assigned = split_into_vehicles_with_types(real_order, demands, vehicle_types)
    if assigned is None:
        return 1e9  # heavily penalize infeasible solutions

    total_cost = 0.0
    for seg, vtype in assigned:
        dist_cost, lateness = evaluate_trip_with_time_windows(
            seg, vtype, dist_matrix, window_start, window_end, service_time
        )
        total_cost += dist_cost + (lateness * LATE_PENALTY_PER_MINUTE)

    return total_cost


def evaluate_route_cost_vrptw_deap(individual, dist_matrix, demands, vehicle_types,
                                     window_start, window_end, service_time):
    """DEAP-compatible wrapper - DEAP requires a tuple return, e.g. (cost,)."""
    cost = evaluate_route_cost_vrptw(
        individual, dist_matrix, demands, vehicle_types,
        window_start, window_end, service_time, zero_indexed=True
    )
    return (cost,)
