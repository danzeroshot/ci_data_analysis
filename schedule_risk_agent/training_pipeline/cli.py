from __future__ import annotations

import argparse
import json
from pathlib import Path

from .configuration import load_run_config
from .feature_qualification import generate_feature_manifest
from .release import promote_customer_bundle, promote_run, rollback_release
from .snapshots import create_label_snapshot_from_csv
from .stages import compare_bundle, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Schedule Risk training pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("run", "qualify", "tune"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True, type=Path)

    release = subparsers.add_parser("release")
    release.add_argument("--run-dir", required=True, type=Path)
    release.add_argument("--output-root", required=True, type=Path)
    release.add_argument("--production", action="store_true")

    customer_release = subparsers.add_parser("customer-release")
    customer_release.add_argument("--candidate", required=True, type=Path)
    customer_release.add_argument("--output-root", required=True, type=Path)
    customer_release.add_argument("--customer", required=True)
    customer_release.add_argument("--production", action="store_true")

    rollback = subparsers.add_parser("rollback")
    rollback.add_argument("--output-root", required=True, type=Path)
    rollback.add_argument("--model-version", required=True)

    labels = subparsers.add_parser("snapshot-labels")
    labels.add_argument("--source-csv", required=True, type=Path)
    labels.add_argument("--output-root", required=True, type=Path)
    labels.add_argument("--target-version", required=True)
    snowflake_labels = subparsers.add_parser("snapshot-labels-snowflake")
    snowflake_labels.add_argument("--sql", required=True, type=Path)
    snowflake_labels.add_argument("--output-root", required=True, type=Path)
    snowflake_labels.add_argument("--target-version", required=True)

    manifest = subparsers.add_parser("generate-manifest")
    manifest.add_argument("--schema", required=True, type=Path)
    manifest.add_argument("--output", required=True, type=Path)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--bundle", required=True, type=Path)
    compare.add_argument("--features", required=True, type=Path)
    compare.add_argument("--feature-manifest", required=True, type=Path)
    compare.add_argument("--labels", required=True, type=Path)
    compare.add_argument("--label-manifest", required=True, type=Path)
    compare.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command in {"run", "qualify", "tune"}:
        stop_after = None if args.command == "run" else args.command
        output = run_pipeline(args.config, stop_after=stop_after)
    elif args.command == "release":
        output = promote_run(args.run_dir, args.output_root, args.production)
    elif args.command == "customer-release":
        output = promote_customer_bundle(
            args.candidate, args.output_root, args.customer, args.production
        )
    elif args.command == "rollback":
        output = rollback_release(args.output_root, args.model_version)
    elif args.command == "snapshot-labels":
        output = create_label_snapshot_from_csv(
            args.source_csv, args.output_root, args.target_version
        )
    elif args.command == "snapshot-labels-snowflake":
        output = create_label_snapshot_from_snowflake(
            args.sql, args.output_root, args.target_version
        )
    elif args.command == "generate-manifest":
        generate_feature_manifest(args.schema, args.output)
        output = args.output
    elif args.command == "compare":
        output = compare_bundle(
            args.bundle,
            args.features,
            args.feature_manifest,
            args.labels,
            args.label_manifest,
            args.output,
        )
    else:
        raise RuntimeError("Unsupported command")
    print(json.dumps({"status": "succeeded", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
