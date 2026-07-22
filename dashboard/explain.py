"""
AegisChain — Policy Copilot
Takes baseline vs disrupted optimizer output and produces a short,
decision-maker-facing explanation. This is the ONE LLM call in the
system — resist the urge to add more agents here, one well-scoped
call is easier to get right in 3 days and easier to defend live.

Requires ANTHROPIC_API_KEY set in the environment.
"""

import os
import json
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()  # reads .env in the current or parent directory if present

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def build_prompt(baseline: dict, disrupted: dict, scenario_desc: str) -> str:
    def fmt(result):
        alloc = {f"{i} -> {j}": round(v, 2) for (i, j), v in result["allocation"].items()}
        return {
            "status": result["status"],
            "allocation_MMbbl_per_week": alloc,
            "shortfalls": result.get("shortfalls", {}),
            "total_weighted_cost_cr_per_week": round(result["total_cost"], 1),
        }

    payload = {
        "scenario": scenario_desc,
        "baseline": fmt(baseline),
        "after_disruption": fmt(disrupted),
    }

    return f"""You are the Policy Copilot in AegisChain, a supply-chain decision
system for India's crude oil procurement. An optimization engine has just
solved a baseline scenario and a disruption scenario. Explain the result to
a refinery operations director in 4-6 sentences.

Rules:
- Lead with the single most important number (cost impact %, or the shortfall
  if any).
- Name which suppliers gained/lost allocation and WHY (cost, transit time,
  risk, corridor disruption) — pull the actual numbers from the data given.
- If strategic reserves were used, say how much and frame it as a deliberate
  trade-off, not a failure.
- If there is a shortfall, state it plainly and what it implies operationally.
- Do not invent numbers not present in the data below.
- Plain, direct, executive tone. No headers, no bullet points, no preamble.

DATA:
{json.dumps(payload, indent=2)}"""


def explain(baseline: dict, disrupted: dict, scenario_desc: str) -> str:
    prompt = build_prompt(baseline, disrupted, scenario_desc)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "../optimizer")
    from model import solve_scenario

    baseline = solve_scenario(label="baseline")
    disrupted = solve_scenario(
        disrupted_corridor="Hormuz",
        corridor_capacity_multiplier=0.10,
        port_congestion={"JNPT": 1.35},
        label="hormuz_closure",
    )

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to test this live. Prompt preview:\n")
        print(build_prompt(baseline, disrupted, "Strait of Hormuz closes to 10% capacity"))
    else:
        print(explain(baseline, disrupted, "Strait of Hormuz closes to 10% capacity"))
