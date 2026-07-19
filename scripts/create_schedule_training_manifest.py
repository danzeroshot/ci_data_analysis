from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schedule_risk_agent.training_pipeline.feature_qualification import (
    generate_feature_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    generate_feature_manifest(args.schema, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
