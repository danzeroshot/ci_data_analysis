from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-card", default="models/schedule_risk_model_card.json")
    parser.add_argument("--metrics", default="models/schedule_risk_training_metrics.json")
    args = parser.parse_args()
    card = json.loads(Path(args.model_card).read_text(encoding="utf-8"))
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    print(json.dumps({
        "model_version": card["model_version"],
        "status": card["status"],
        "metrics": metrics,
        "limitations": card.get("limitations", []),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

