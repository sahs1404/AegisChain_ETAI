"""
AegisChain — Policy Copilot (Gemini version)

Same role as explain.py: narrate the optimizer's before/after decision
card for a judge/operations-director audience. Swapped to Google's
Gemini API since it has a genuinely free tier (no credit card),
generous enough for a hackathon's rehearsal + live demo volume.

Setup:
    1. Get a free key at https://aistudio.google.com/apikey (no card needed)
    2. export GEMINI_API_KEY=your_key_here
    3. pip install google-genai --break-system-packages

Free tier reference (verify current limits at ai.google.dev/pricing,
these change without much notice): Gemini 3.0 Flash, ~1,500 requests/day,
1M token context. Comfortably covers a hackathon's usage.
"""

import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()  # reads .env in the current or parent directory if present

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY (get one free at https://aistudio.google.com/apikey)")
        _client = genai.Client(api_key=api_key)
    return _client


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


def explain(baseline: dict, disrupted: dict, scenario_desc: str, model: str = "gemini-flash-latest") -> str:
    prompt = build_prompt(baseline, disrupted, scenario_desc)
    client = _get_client()
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text


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

    if not os.environ.get("GEMINI_API_KEY"):
        print("Set GEMINI_API_KEY to test this live (free key: https://aistudio.google.com/apikey)\n")
        print("Prompt preview:\n")
        print(build_prompt(baseline, disrupted, "Strait of Hormuz closes to 10% capacity"))
    else:
        print(explain(baseline, disrupted, "Strait of Hormuz closes to 10% capacity"))
