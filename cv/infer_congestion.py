"""
AegisChain — Real Ship Detection & Congestion Inference

Uses an actual YOLOv8 model trained specifically on satellite ship
imagery (single class: 'boat'), not generic COCO pretraining. Source
model: robmarkcole/kaggle-ships-in-satellite-imagery-with-YOLOv8
(trained on the Kaggle "Ships in Google Earth" dataset).

This is a genuine starting point, not a mock — it already detects real
ships in real satellite crops. Your teammate's job is to point this at
imagery of YOUR specific ports (JNPT, Kandla, Paradip) and calibrate
the baseline_count per port — the model itself doesn't need retraining
unless accuracy on your specific port imagery turns out to need it.
"""

from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "weights" / "ship_yolov8n.pt"
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = YOLO(str(MODEL_PATH))
    return _model


def count_ships(image_path: str, conf: float = 0.25) -> int:
    """Run detection on one image, return the number of ships found."""
    model = _get_model()
    results = model.predict(image_path, conf=conf, verbose=False)
    return len(results[0].boxes)


def image_to_congestion_index(image_path: str, baseline_count: int, conf: float = 0.25) -> dict:
    """
    Converts a raw ship count into the same congestion_index multiplier
    format used by optimizer/model.py's port_congestion parameter.

    baseline_count: the typical/median ship count for THIS port on a
    normal day — you calibrate this per port from a handful of historical
    images (e.g. median of 10-20 past readings). This is the one number
    your CV teammate needs to set per port; everything else is automatic.
    """
    count = count_ships(image_path, conf=conf)
    if baseline_count <= 0:
        index = 1.0
    else:
        excess_ratio = max(0, count - baseline_count) / baseline_count
        index = round(1.0 + excess_ratio, 3)  # 50% more ships than normal -> 1.5x multiplier
    return {
        "ship_count": count,
        "baseline_count": baseline_count,
        "congestion_index": index,
    }


if __name__ == "__main__":
    sample = str(Path(__file__).parent / "sample_images" / "val_preds.jpeg")
    # baseline_count is illustrative here — in production this comes from
    # calibrating against that port's typical historical ship count
    result = image_to_congestion_index(sample, baseline_count=20)
    print(f"Ships detected: {result['ship_count']}")
    print(f"Baseline (normal day): {result['baseline_count']}")
    print(f"Congestion index: {result['congestion_index']}")
    print(f"\n-> Feed this into optimizer/model.py as: "
          f"port_congestion={{'JNPT': {result['congestion_index']}}}")
