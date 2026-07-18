from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .errors import FeatureVersionMismatch


@dataclass(frozen=True)
class FeatureSchema:
    feature_schema_version: str
    keyword_manifest_version: str
    ordered_features: List[str]
    class_labels: List[str]

    @classmethod
    def load(cls, path: Path) -> "FeatureSchema":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            feature_schema_version=data["feature_schema_version"],
            keyword_manifest_version=data["keyword_manifest_version"],
            ordered_features=list(data["ordered_features"]),
            class_labels=list(data["class_labels"]),
        )

    def validate_columns(self, columns) -> None:
        available = {str(column).upper() for column in columns}
        missing = [name for name in self.ordered_features if name.upper() not in available]
        if missing:
            raise FeatureVersionMismatch(
                "Published features are missing required model columns",
                {"missing_count": len(missing), "sample": missing[:20]},
            )

    def sha256(self) -> str:
        payload = json.dumps({
            "feature_schema_version": self.feature_schema_version,
            "keyword_manifest_version": self.keyword_manifest_version,
            "ordered_features": self.ordered_features,
            "class_labels": self.class_labels,
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

