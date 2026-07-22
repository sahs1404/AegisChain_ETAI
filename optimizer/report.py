"""
AegisChain — Decision Summary Card
Consolidates baseline vs scenario into the before/after metrics block
for the dashboard. Every number here is derived from the actual
optimizer allocation — nothing is a placeholder or restated estimate.
"""

from model import SUPPLIERS, TRANSIT_DAYS, BASE_RISK, solve_scenario
from confidence import confidence_score


def _weighted_avg_delay_and_risk(result):
    """Delay and risk, weighted by how much volume each supplier carries
    in this allocation. A recommendation that shifts volume to slower
    or riskier suppliers should show that honestly in these numbers."""
    total_qty = sum(result["allocation"].values())
    if total_qty == 0:
        return 0.0, 0.0
    delay = sum(qty * TRANSIT_DAYS[i] for (i, j), qty in result["allocation"].items()) / total_qty
    risk = sum(qty * BASE_RISK[i] for (i, j), qty in result["allocation"].items()) / total_qty
    return round(delay, 1), round(risk * 100, 1)  # risk as %


def decision_card(scenario_kwargs: dict, scenario_label: str = "scenario") -> dict:
    baseline = solve_scenario(label="baseline")
    scenario = solve_scenario(**scenario_kwargs, label=scenario_label)

    b_delay, b_risk = _weighted_avg_delay_and_risk(baseline)
    s_delay, s_risk = _weighted_avg_delay_and_risk(scenario)

    conf = confidence_score(scenario_kwargs, n_trials=40)

    total_shortfall = sum(scenario.get("shortfalls", {}).values())

    return {
        "before": {
            "expected_delay_days": b_delay,
            "risk_pct": b_risk,
            "cost_cr_per_week": round(baseline["total_cost"], 1),
        },
        "after": {
            "expected_delay_days": s_delay,
            "risk_pct": s_risk,
            "cost_cr_per_week": round(scenario["total_cost"], 1),
            "shortfall_MMbbl": round(total_shortfall, 2) if total_shortfall else 0,
        },
        "cost_change_pct": round(
            (scenario["total_cost"] - baseline["total_cost"]) / baseline["total_cost"] * 100, 1
        ),
        "confidence_pct": conf["confidence_pct"],
        "top_recommendation": conf["reference_recommendation"],
        "baseline_raw": baseline,
        "scenario_raw": scenario,
    }


if __name__ == "__main__":
    card = decision_card({
        "disrupted_corridor": "Hormuz",
        "corridor_capacity_multiplier": 0.10,
        "port_congestion": {"JNPT": 1.35},
    }, scenario_label="hormuz_closure")

    print("BEFORE:", card["before"])
    print("AFTER: ", card["after"])
    print(f"Cost change: {card['cost_change_pct']:+.1f}%")
    print(f"Confidence: {card['confidence_pct']}%")
    print(f"Top move: {card['top_recommendation']}")
