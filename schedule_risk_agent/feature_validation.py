from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List

from .errors import FeatureValidationError
from .feature_schema import FeatureSchema


@dataclass
class ValidationResult:
    status: str
    build_id: str
    validated_at_utc: str
    row_count: int
    column_count: int
    duplicate_project_keys: int
    customers: List[str]
    feature_schema_version: str
    keyword_manifest_version: str
    warnings: List[str]

    def as_dict(self) -> Dict:
        return self.__dict__.copy()


def validate_build(connection, table_name: str, build_id: str, schema=None):
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT * FROM " + table_name + " LIMIT 0")
        columns = [item[0] for item in cursor.description]
        required_metadata = {
            "CUSTOMERNAME", "PROJECTID", "FEATUREASOFUTC",
            "FEATURESCHEMAVERSION", "KEYWORDMANIFESTVERSION",
        }
        missing_metadata = sorted(required_metadata - {c.upper() for c in columns})
        if missing_metadata:
            raise FeatureValidationError(
                "Build is missing metadata columns", {"missing": missing_metadata}
            )
        if schema:
            schema.validate_columns(columns)

        cursor.execute("SELECT COUNT(*) FROM " + table_name)
        row_count = int(cursor.fetchone()[0])
        if row_count == 0:
            raise FeatureValidationError("Feature build contains zero rows")

        cursor.execute(
            "SELECT COUNT(*) FROM (SELECT CustomerName, ProjectID FROM "
            + table_name
            + " GROUP BY CustomerName, ProjectID HAVING COUNT(*) > 1)"
        )
        duplicate_count = int(cursor.fetchone()[0])
        if duplicate_count:
            raise FeatureValidationError(
                "Feature build has duplicate project keys",
                {"duplicate_project_keys": duplicate_count},
            )

        cursor.execute(
            "SELECT DISTINCT CustomerName FROM " + table_name + " ORDER BY CustomerName"
        )
        customers = [str(row[0]) for row in cursor.fetchall()]

        cursor.execute(
            "SELECT MIN(FeatureSchemaVersion), MAX(FeatureSchemaVersion), "
            "MIN(KeywordManifestVersion), MAX(KeywordManifestVersion) FROM " + table_name
        )
        min_schema, max_schema, min_keyword, max_keyword = cursor.fetchone()
        if min_schema != max_schema or min_keyword != max_keyword:
            raise FeatureValidationError("Build contains mixed feature versions")
        if schema and min_schema != schema.feature_schema_version:
            raise FeatureValidationError(
                "Feature schema version mismatch",
                {"build": min_schema, "model": schema.feature_schema_version},
            )
        if schema and min_keyword != schema.keyword_manifest_version:
            raise FeatureValidationError(
                "Keyword manifest version mismatch",
                {"build": min_keyword, "model": schema.keyword_manifest_version},
            )

        return ValidationResult(
            status="passed",
            build_id=build_id,
            validated_at_utc=datetime.now(timezone.utc).isoformat(),
            row_count=row_count,
            column_count=len(columns),
            duplicate_project_keys=0,
            customers=customers,
            feature_schema_version=str(min_schema),
            keyword_manifest_version=str(min_keyword),
            warnings=[],
        )
    finally:
        cursor.close()

