"""
AegisChain — Recommendation Confidence

Real technique, not a decorative percentage: perturb the uncertain
inputs (cost, risk, capacity — the numbers you're least sure about in
a live system) by random noise, re-solve the optimization many times,
and measure how often the SAME top recommendation comes out.

If the top supplier switch survives noisy inputs 90% of the time,
that's a defensible 90% confidence claim. If it flips constantly, the
system is telling you the decision is on a knife's edge — which is
itself a useful thing to show a judge (or an operations director).
"""

import random
import copy
from model import (
    SUPPLIERS, REFINERIES, COST, TRANSIT_DAYS, BASE_RISK, CAPACITY,
    solve_scenario,
)
import model as model_module


def _top_edge(result):
    """The single largest allocation flow — used as 'the recommendation'
    to check for agreement across trials."""
    if not result["allocation"]:
        return None
    return max(result["allocation"].items(), key=lambda kv: kv[1])[0]  # (supplier, refinery)


def confidence_score(scenario_kwargs: dict, n_trials: int = 40, noise: float = 0.12, seed: int = 42):
    """
    scenario_kwargs: passed straight through to solve_scenario(), e.g.
        {"disrupted_corridor": "Hormuz", "corridor_capacity_multiplier": 0.10,
         "port_congestion": {"JNPT": 1.35}}

    Returns dict with the reference top recommendation, agreement rate,
    and the resulting confidence percentage.
    """
    rng = random.Random(seed)

    reference = solve_scenario(**scenario_kwargs, label="reference")
    ref_edge = _top_edge(reference)

    # snapshot originals so we can restore after perturbing module globals
    orig_cost = copy.deepcopy(COST)
    orig_risk = copy.deepcopy(BASE_RISK)
    orig_cap = copy.deepcopy(CAPACITY)

    agree = 0
    for _ in range(n_trials):
        for i in SUPPLIERS:
            model_module.COST[i] = orig_cost[i] * (1 + rng.uniform(-noise, noise))
            model_module.BASE_RISK[i] = max(0.0, orig_risk[i] * (1 + rng.uniform(-noise, noise)))
            model_module.CAPACITY[i] = max(0.1, orig_cap[i] * (1 + rng.uniform(-noise, noise)))

        trial = solve_scenario(**scenario_kwargs, label="trial")
        if _top_edge(trial) == ref_edge:
            agree += 1

    # restore
    model_module.COST.update(orig_cost)
    model_module.BASE_RISK.update(orig_risk)
    model_module.CAPACITY.update(orig_cap)

    confidence_pct = round(100 * agree / n_trials, 1)
    return {
        "reference_recommendation": f"{ref_edge[0]} -> {ref_edge[1]}" if ref_edge else None,
        "agreement_trials": agree,
        "total_trials": n_trials,
        "confidence_pct": confidence_pct,
        "noise_level": noise,
    }


if __name__ == "__main__":
    disrupted_kwargs = {
        "disrupted_corridor": "Hormuz",
        "corridor_capacity_multiplier": 0.10,
        "port_congestion": {"JNPT": 1.35},
    }
    result = confidence_score(disrupted_kwargs)
    print(f"Top recommendation: {result['reference_recommendation']}")
    print(f"Confidence: {result['confidence_pct']}% "
          f"({result['agreement_trials']}/{result['total_trials']} trials agree, "
          f"±{result['noise_level']*100:.0f}% input noise)")
