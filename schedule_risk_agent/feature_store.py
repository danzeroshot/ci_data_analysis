from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Dict, List
from uuid import uuid4

import joblib
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .errors import FeatureDataStale, FeatureRepositoryNotReady, FeatureVersionMismatch
from .feature_schema import FeatureSchema


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temp), str(path))


class LocalFeaturePublisher:
    def __init__(self, root: Path, schema: FeatureSchema, retain_count: int = 3):
        self.root = root
        self.retain_count = retain_count
        self.schema = schema

    def publish(self, connection, table_name, validation, sql_sha256, schema_sha256):
        self.root.mkdir(parents=True, exist_ok=True)
        snapshots = self.root / "snapshots"
        snapshots.mkdir(exist_ok=True)
        build_id = validation.build_id
        final_dir = snapshots / build_id
        if final_dir.exists():
            raise ValueError("Snapshot already exists: " + build_id)
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=str(snapshots)))
        parquet_path = staging / "features.parquet"
        cursor = connection.cursor()
        writer = None
        rows_written = 0
        try:
            projection_columns = ["CUSTOMERNAME", "PROJECTID", "FEATUREASOFUTC", "FEATURESCHEMAVERSION", "KEYWORDMANIFESTVERSION"] + self.schema.ordered_features
            projection = ", ".join("\"{}\"".format(name) for name in projection_columns)
            cursor.execute("SELECT " + projection + " FROM " + table_name + " ORDER BY CustomerName, ProjectID")
            columns = [item[0] for item in cursor.description]
            while True:
                rows = cursor.fetchmany(250)
                if not rows:
                    break
                frame = pd.DataFrame.from_records(rows, columns=columns)
                for feature_name in self.schema.ordered_features:
                    frame[feature_name] = pd.to_numeric(frame[feature_name], errors="coerce").astype("float64")
                frame["CUSTOMERNAME"] = frame["CUSTOMERNAME"].astype(str)
                frame["PROJECTID"] = frame["PROJECTID"].astype(str)
                frame["FEATUREASOFUTC"] = pd.to_datetime(frame["FEATUREASOFUTC"], utc=True)
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(str(parquet_path), table.schema, compression="zstd")
                elif table.schema != writer.schema:
                    table = table.cast(writer.schema)
                writer.write_table(table)
                rows_written += len(frame)
            if writer is not None:
                writer.close()
                writer = None
            if rows_written != validation.row_count:
                raise ValueError(
                    "Exported row count {} != validated {}".format(
                        rows_written, validation.row_count
                    )
                )
            metadata = pq.read_metadata(str(parquet_path))
            if metadata.num_rows != validation.row_count:
                raise ValueError("Parquet verification row count mismatch")
            manifest = {
                "build_id": build_id,
                "published_at_utc": datetime.now(timezone.utc).isoformat(),
                "feature_as_of_utc": validation.validated_at_utc,
                "row_count": validation.row_count,
                "column_count": validation.column_count,
                "customers": validation.customers,
                "feature_schema_version": validation.feature_schema_version,
                "keyword_manifest_version": validation.keyword_manifest_version,
                "sql_sha256": sql_sha256,
                "feature_schema_sha256": schema_sha256,
                "features_file": "features.parquet",
            }
            (staging / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            (staging / "validation.json").write_text(
                json.dumps(validation.as_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            checksums = {
                name: _sha256(staging / name)
                for name in ("features.parquet", "manifest.json", "validation.json")
            }
            (staging / "checksums.sha256").write_text(
                "".join("{}  {}\n".format(value, name) for name, value in checksums.items()),
                encoding="utf-8",
            )
            (staging / "COMPLETE").write_text("", encoding="ascii")
            os.replace(str(staging), str(final_dir))
            _atomic_json(self.root / "current.json", {
                "build_id": build_id,
                "relative_path": "snapshots/" + build_id,
                "published_at_utc": manifest["published_at_utc"],
            })
            self._cleanup()
            return manifest
        finally:
            if writer is not None:
                writer.close()
            cursor.close()
            if staging.exists():
                shutil.rmtree(str(staging), ignore_errors=True)

    def _cleanup(self):
        current = json.loads((self.root / "current.json").read_text(encoding="utf-8"))
        keep_current = current["build_id"]
        dirs = sorted(
            [path for path in (self.root / "snapshots").iterdir() if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        keep = {path.name for path in dirs[: self.retain_count]} | {keep_current}
        for path in dirs:
            if path.name not in keep and not path.name.startswith(".staging-"):
                shutil.rmtree(str(path))


class LocalFeatureRepository:
    def __init__(self, root: Path, schema: FeatureSchema, max_age_hours: int = 24):
        self.root = root.resolve()
        self.schema = schema
        self.max_age_hours = max_age_hours
        self._lock = RLock()
        self._frame = None
        self._manifest = None

    def open(self):
        self.refresh_if_changed(force=True)
        return self._manifest

    def refresh_if_changed(self, force=False):
        pointer_path = self.root / "current.json"
        if not pointer_path.exists():
            raise FeatureRepositoryNotReady("No current local feature snapshot")
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        if not force and self._manifest and pointer["build_id"] == self._manifest["build_id"]:
            return False
        snapshot = (self.root / pointer["relative_path"]).resolve()
        if self.root not in snapshot.parents or not (snapshot / "COMPLETE").exists():
            raise FeatureRepositoryNotReady("Current snapshot path is invalid")
        manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
        checksums = {}
        for line in (snapshot / "checksums.sha256").read_text(encoding="utf-8").splitlines():
            checksum, name = line.split("  ", 1)
            checksums[name] = checksum
        for name, expected in checksums.items():
            if _sha256(snapshot / name) != expected:
                raise FeatureRepositoryNotReady("Snapshot checksum mismatch: " + name)
        if manifest["feature_schema_version"] != self.schema.feature_schema_version:
            raise FeatureVersionMismatch("Feature schema version mismatch")
        if manifest["keyword_manifest_version"] != self.schema.keyword_manifest_version:
            raise FeatureVersionMismatch("Keyword manifest version mismatch")
        columns = ["CUSTOMERNAME", "PROJECTID", "FEATUREASOFUTC"] + self.schema.ordered_features
        frame = pd.read_parquet(snapshot / "features.parquet", columns=columns)
        frame.columns = [str(column).upper() for column in frame.columns]
        self.schema.validate_columns(frame.columns)
        frame["CUSTOMERNAME"] = frame["CUSTOMERNAME"].astype(str)
        frame["PROJECTID"] = frame["PROJECTID"].astype(str)
        frame = frame.set_index(["CUSTOMERNAME", "PROJECTID"], drop=False)
        with self._lock:
            self._frame = frame
            self._manifest = manifest
        return True

    def fetch(self, keys):
        if self._frame is None:
            raise FeatureRepositoryNotReady("Feature repository is not open")
        published = pd.Timestamp(self._manifest["published_at_utc"])
        now = pd.Timestamp.now(tz="UTC")
        if published.tzinfo is None:
            published = published.tz_localize("UTC")
        age_hours = (now - published).total_seconds() / 3600
        if age_hours > self.max_age_hours:
            raise FeatureDataStale(
                "Feature snapshot is older than allowed",
                {"feature_age_hours": age_hours},
            )
        found = []
        missing = []
        with self._lock:
            for customer, project in keys:
                key = (str(customer), str(project))
                if key in self._frame.index:
                    row = self._frame.loc[key]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    found.append(row)
                else:
                    missing.append(key)
        return pd.DataFrame(found), missing, dict(self._manifest, feature_age_hours=age_hours)

