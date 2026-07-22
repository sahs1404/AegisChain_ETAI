"""
AegisChain — Optimization Engine
Allocates crude procurement from suppliers to Indian refineries,
minimizing cost + delay + risk, subject to capacity/demand constraints.

Supports:
  - baseline scenario
  - disruption scenarios (e.g. Hormuz closure -> corridor capacity collapse)
  - live congestion index injection (from CV pipeline) -> raises effective
    cost/delay on routes through a congested port
"""

import pulp

# ---------------------------------------------------------------------------
# STATIC NETWORK DATA
# Suppliers -> Indian refineries. Each route has a corridor tag so a
# disruption (e.g. "Hormuz closes") can be applied to every route that
# passes through that corridor, not just one edge.
# ---------------------------------------------------------------------------

SUPPLIERS = ["Saudi Arabia", "Iraq", "UAE", "USA_Gulf", "Nigeria", "Russia_ESPO", "Strategic_Reserve"]
REFINERIES = ["Jamnagar", "Paradip", "Mumbai"]

# corridor each supplier's route passes through
CORRIDOR = {
    "Saudi Arabia": "Hormuz",
    "Iraq": "Hormuz",
    "UAE": "Hormuz",
    "USA_Gulf": "Cape",
    "Nigeria": "Atlantic",
    "Russia_ESPO": "EasternRoute",
    "Strategic_Reserve": "Domestic",
}

# cost in ₹ Cr per million-barrel unit shipped (illustrative, ordered
# roughly by real-world landed cost differentials). Reserve release is
# deliberately expensive — it's a last resort, not a cheap option.
COST = {
    "Saudi Arabia": 84, "Iraq": 82, "UAE": 85,
    "USA_Gulf": 96, "Nigeria": 90, "Russia_ESPO": 78,
    "Strategic_Reserve": 130,
}

# baseline transit time in days
TRANSIT_DAYS = {
    "Saudi Arabia": 9, "Iraq": 10, "UAE": 9,
    "USA_Gulf": 21, "Nigeria": 16, "Russia_ESPO": 25,
    "Strategic_Reserve": 1,
}

# baseline geopolitical/route risk score, 0 (safe) - 1 (high risk)
BASE_RISK = {
    "Saudi Arabia": 0.15, "Iraq": 0.20, "UAE": 0.15,
    "USA_Gulf": 0.05, "Nigeria": 0.20, "Russia_ESPO": 0.10,
    "Strategic_Reserve": 0.0,
}

# max supply capacity per supplier, million barrels / week
# Strategic_Reserve capacity models a limited emergency drawdown rate
CAPACITY = {
    "Saudi Arabia": 6.0, "Iraq": 4.0, "UAE": 3.0,
    "USA_Gulf": 3.5, "Nigeria": 2.5, "Russia_ESPO": 3.0,
    "Strategic_Reserve": 2.5,
}

# refinery weekly demand, million barrels
DEMAND = {"Jamnagar": 7.0, "Paradip": 4.0, "Mumbai": 3.5}

# which port each refinery is fed through (used for CV congestion injection)
REFINERY_PORT = {"Jamnagar": "Kandla", "Paradip": "Paradip Port", "Mumbai": "JNPT"}

# penalty weights (tunable — this is where "why B not A" comes from)
DELAY_PENALTY_PER_DAY = 0.6      # ₹Cr per unit per day of transit
RISK_PENALTY = 40                # ₹Cr per unit at risk=1.0


def solve_scenario(
    disrupted_corridor: str = None,
    corridor_capacity_multiplier: float = 0.0,
    port_congestion: dict = None,       # e.g. {"JNPT": 1.4}  -> 40% cost/delay markup
    label: str = "baseline",
):
    """
    Solve one allocation scenario.

    disrupted_corridor: corridor name to disrupt (e.g. "Hormuz")
    corridor_capacity_multiplier: fraction of normal capacity still available
        on that corridor (0.0 = fully closed, 0.5 = half capacity)
    port_congestion: dict of port_name -> multiplier applied to cost & delay
        for refineries fed through that port (this is the CV injection point)
    """
    port_congestion = port_congestion or {}

    prob = pulp.LpProblem(f"AegisChain_{label}", pulp.LpMinimize)

    x = {
        (i, j): pulp.LpVariable(f"x_{i}_{j}", lowBound=0)
        for i in SUPPLIERS for j in REFINERIES
    }
    # shortfall = unmet demand at each refinery. Heavily penalized (fuel
    # shortage has real economic/social cost) but keeps the model always
    # feasible so the optimizer can express "how bad" rather than just
    # failing outright.
    shortfall = {j: pulp.LpVariable(f"short_{j}", lowBound=0) for j in REFINERIES}
    SHORTFALL_PENALTY = 300  # ₹Cr per unit unmet — must dominate all other costs

    def effective_cost(i, j):
        c = COST[i]
        port = REFINERY_PORT[j]
        if port in port_congestion:
            c *= port_congestion[port]
        return c

    def effective_delay(i, j):
        d = TRANSIT_DAYS[i]
        port = REFINERY_PORT[j]
        if port in port_congestion:
            d *= port_congestion[port]
        return d

    # objective: landed cost + delay penalty + risk penalty
    prob += pulp.lpSum(
        x[i, j] * (
            effective_cost(i, j)
            + DELAY_PENALTY_PER_DAY * effective_delay(i, j)
            + RISK_PENALTY * BASE_RISK[i]
        )
        for i in SUPPLIERS for j in REFINERIES
    ) + pulp.lpSum(shortfall[j] * SHORTFALL_PENALTY for j in REFINERIES)

    # supplier capacity constraints (with corridor disruption applied)
    for i in SUPPLIERS:
        cap = CAPACITY[i]
        if disrupted_corridor and CORRIDOR[i] == disrupted_corridor:
            cap *= corridor_capacity_multiplier
        prob += pulp.lpSum(x[i, j] for j in REFINERIES) <= cap, f"cap_{i}"

    # refinery demand constraints (must be met)
    for j in REFINERIES:
        prob += pulp.lpSum(x[i, j] for i in SUPPLIERS) + shortfall[j] >= DEMAND[j], f"dem_{j}"

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))

    allocation = {
        (i, j): x[i, j].value()
        for i in SUPPLIERS for j in REFINERIES
        if x[i, j].value() and x[i, j].value() > 1e-4
    }
    shortfalls = {j: shortfall[j].value() for j in REFINERIES if shortfall[j].value() and shortfall[j].value() > 1e-4}

    total_cost = pulp.value(prob.objective)

    return {
        "label": label,
        "status": pulp.LpStatus[status],
        "allocation": allocation,
        "shortfalls": shortfalls,
        "total_cost": total_cost,
        "feasible": pulp.LpStatus[status] == "Optimal",
    }


def summarize(result):
    lines = [f"--- Scenario: {result['label']} ({result['status']}) ---"]
    if not result["feasible"]:
        lines.append("INFEASIBLE — demand cannot be met under these constraints.")
        return "\n".join(lines)
    for (i, j), qty in sorted(result["allocation"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {i:18s} -> {j:10s} : {qty:6.2f} MMbbl/week  "
                      f"(corridor={CORRIDOR[i]}, {TRANSIT_DAYS[i]}d, risk={BASE_RISK[i]})")
    if result.get("shortfalls"):
        for j, qty in result["shortfalls"].items():
            lines.append(f"  ⚠ SHORTFALL at {j}: {qty:.2f} MMbbl/week unmet")
    lines.append(f"  TOTAL WEIGHTED COST: ₹{result['total_cost']:.1f} Cr/week")
    return "\n".join(lines)


if __name__ == "__main__":
    baseline = solve_scenario(label="baseline")
    print(summarize(baseline))
    print()

    # Disruption: Hormuz closes to 10% capacity, AND JNPT port congested
    # (this congestion multiplier is where the CV pipeline plugs in later)
    disrupted = solve_scenario(
        disrupted_corridor="Hormuz",
        corridor_capacity_multiplier=0.10,
        port_congestion={"JNPT": 1.35},
        label="hormuz_closure",
    )
    print(summarize(disrupted))
    print()

    delta = disrupted["total_cost"] - baseline["total_cost"]
    print(f"Impact of disruption: +₹{delta:.1f} Cr/week vs baseline "
          f"({delta / baseline['total_cost'] * 100:.1f}% increase)")
    reserve_used = sum(v for (i, j), v in disrupted["allocation"].items() if i == "Strategic_Reserve")
    if reserve_used:
        days = reserve_used / CAPACITY["Strategic_Reserve"] * 7
        print(f"Recommendation: release strategic reserves "
              f"(~{reserve_used:.2f} MMbbl, ≈{days:.1f} days of reserve drawdown)")
