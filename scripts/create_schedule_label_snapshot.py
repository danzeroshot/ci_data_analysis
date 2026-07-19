from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schedule_risk_agent.training_pipeline.snapshots import (
    create_label_snapshot_from_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--target-version", default="schedule-delay-v1")
    args = parser.parse_args()
    output = create_label_snapshot_from_csv(
        args.source_csv, args.output_root, args.target_version
    )
    print(output)


if __name__ == "__main__":
    main()
