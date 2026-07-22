"""
AegisChain — Port Congestion Index

INTERFACE CONTRACT (for whoever builds the real YOLO pipeline):
    get_congestion_index(port_name: str) -> float

    Returns a multiplier >= 1.0 applied to cost/delay on routes through
    that port in optimizer/model.py's `port_congestion` dict.
        1.0  = normal operations
        1.2  = moderate congestion (+20% effective cost/delay)
        1.5+ = severe congestion

REAL PIPELINE TO BUILD (separate from this file, in Colab):
    1. Dataset: Airbus Ship Detection Challenge (Kaggle) or HRSC2016
    2. Fine-tune YOLOv8n (ultralytics) — nano model, trains fast, good
       enough for "count ships in a bounding region"
    3. For a chosen port (e.g. JNPT Mumbai), pull recent Sentinel-2 imagery
       via Google Earth Engine or Sentinel Hub (free tier) cropped to the
       port's anchorage zone
    4. Run inference -> ship count in anchorage
    5. Map ship count -> congestion_index via a simple calibration:
       e.g. index = 1.0 + max(0, (ship_count - baseline_count)) * 0.05
       (baseline_count = typical/median ship count for that port, which
       you estimate from a handful of historical images)
    6. Export get_congestion_index() with the SAME signature as below,
       reading from a small JSON/dict of {port: latest_count} that your
       inference script updates.

Until that's ready, this file returns synthetic-but-plausible values so
the optimizer and dashboard can be built and demoed end-to-end today.
"""

import random

PORTS = ["JNPT", "Kandla", "Paradip Port"]

# manually settable scenario state for the demo — the dashboard's
# "what-if" slider should just set this dict directly
_MOCK_STATE = {p: 1.0 for p in PORTS}


def set_mock_congestion(port_name: str, multiplier: float):
    """Used by the dashboard's what-if controls / disruption scenario button."""
    if port_name not in PORTS:
        raise ValueError(f"Unknown port: {port_name}")
    _MOCK_STATE[port_name] = multiplier


def get_congestion_index(port_name: str, mode: str = "mock") -> float:
    """
    mode="mock": returns whatever was last set via set_mock_congestion
                 (defaults to 1.0, i.e. no congestion)
    mode="live": placeholder for real YOLO pipeline — currently raises,
                 replace this branch once the CV model is ready
    """
    if port_name not in PORTS:
        raise ValueError(f"Unknown port: {port_name}")

    if mode == "mock":
        return _MOCK_STATE[port_name]

    if mode == "live":
        raise NotImplementedError(
            "Wire this to the real YOLO ship-count -> congestion mapping "
            "described in the module docstring."
        )

    raise ValueError(f"Unknown mode: {mode}")


def get_all_congestion(mode: str = "mock") -> dict:
    """Convenience: congestion dict ready to pass straight into
    optimizer.model.solve_scenario(port_congestion=...)"""
    return {p: get_congestion_index(p, mode=mode) for p in PORTS}


if __name__ == "__main__":
    print("Default (no congestion):", get_all_congestion())
    set_mock_congestion("JNPT", 1.35)
    print("After setting JNPT congestion:", get_all_congestion())
