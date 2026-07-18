from __future__ import annotations

import argparse
import hashlib
import json
from io import StringIO
from datetime import datetime, timezone
from uuid import uuid4

from .config import Settings
from .feature_schema import FeatureSchema
from .feature_store import LocalFeaturePublisher
from .feature_validation import validate_build
from .snowflake_access import connect


BUILD_TABLE = "SCHEDULE_PROJECT_FEATURES_BUILD"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["local", "snowflake"])
    args = parser.parse_args()
    settings = Settings.load()
    target = args.target or settings.feature_publish_target
    if target != "local":
        raise NotImplementedError(
            "Persistent publisher requires client schema privileges; use existing "
            "schedule_risk_feature_store_refresh_current.sql after deployment configuration"
        )
    schema = FeatureSchema.load(settings.feature_schema_path)
    sql = settings.feature_sql_path.read_text(encoding="utf-8")
    build_id = "schedule-features-{}-{}".format(
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), uuid4().hex[:8]
    )
    connection = connect(settings)
    try:
        for statement_index, cursor in enumerate(connection.execute_stream(StringIO(sql)), start=1):
            print("statement={} query_id={}".format(statement_index, cursor.sfqid), flush=True)
        validation = validate_build(connection, BUILD_TABLE, build_id, schema)
        publisher = LocalFeaturePublisher(settings.feature_snapshot_root, schema)
        manifest = publisher.publish(
            connection,
            BUILD_TABLE,
            validation,
            hashlib.sha256(sql.encode("utf-8")).hexdigest(),
            schema.sha256(),
        )
        print(json.dumps({
            "status": "published",
            "target": target,
            "manifest": manifest,
            "validation": validation.as_dict(),
        }, indent=2, sort_keys=True))
    finally:
        connection.close()


if __name__ == "__main__":
    main()

