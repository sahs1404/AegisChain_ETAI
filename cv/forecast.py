"""
AegisChain — Congestion Forecast (lightweight, honest scope)

Does NOT ingest live news/weather feeds — that's flagged as a roadmap
item in the pitch, not built here (would blow the 3-day budget).

What this DOES do: takes a short history of congestion readings for a
port and extrapolates 24h/48h/72h ahead using simple linear trend
fitting. That's enough to justify "the system reroutes proactively,
not just reactively" without pretending to have a forecasting model
that would take weeks to validate properly.
"""

import numpy as np


def forecast_congestion(history: list[float], horizons_hours=(24, 48, 72)) -> dict:
    """
    history: recent congestion_index readings, oldest first, assumed to
             be evenly spaced (e.g. one reading per satellite pass /
             per few hours — spacing doesn't matter for a linear fit,
             only relative trend does)
    returns: {horizon_hours: projected_index}
    """
    if len(history) < 2:
        # not enough signal to trend — assume flat continuation
        last = history[-1] if history else 1.0
        return {h: last for h in horizons_hours}

    x = np.arange(len(history))
    y = np.array(history)
    slope, intercept = np.polyfit(x, y, 1)

    # project forward. Treat one "step" in history as ~6 hours between
    # satellite passes — tune this to your actual CV pipeline's cadence.
    hours_per_step = 6.0
    last_step = len(history) - 1

    forecast = {}
    for h in horizons_hours:
        steps_ahead = h / hours_per_step
        projected = intercept + slope * (last_step + steps_ahead)
        # congestion index shouldn't sensibly go below 1.0 (no congestion)
        forecast[h] = max(1.0, round(projected, 3))
    return forecast


def should_preempt(history: list[float], threshold: float = 1.3, horizon_hours: int = 48) -> bool:
    """
    Decision rule: if the forecast crosses `threshold` within `horizon_hours`,
    the optimizer should be re-run NOW with that forecasted (not current)
    congestion value — i.e. reroute before the port actually clogs up.
    """
    fc = forecast_congestion(history, horizons_hours=(horizon_hours,))
    return fc[horizon_hours] >= threshold


if __name__ == "__main__":
    # simulated readings showing congestion building up at JNPT
    rising = [1.0, 1.05, 1.1, 1.2, 1.28]
    print("History:", rising)
    print("Forecast:", forecast_congestion(rising))
    print("Should preempt within 48h?", should_preempt(rising))

    flat = [1.0, 1.02, 0.99, 1.01, 1.0]
    print("\nHistory:", flat)
    print("Forecast:", forecast_congestion(flat))
    print("Should preempt within 48h?", should_preempt(flat))
