from __future__ import annotations

import configparser
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .errors import ConfigurationError


ROOT = Path(__file__).resolve().parent.parent


def _clean(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip().strip('"').strip("'")
    if value.lower() in {"", "none", "null", "<none selected>"}:
        return None
    return value


def _file_connection() -> dict:
    path = Path(os.getenv("SNOWFLAKE_CONFIG_FILE", ROOT / "snowflake_access_config.txt.txt"))
    if not path.exists():
        return {}
    parser = configparser.ConfigParser()
    parser.read(str(path))
    if not parser.sections():
        return {}
    return {key: _clean(value) for key, value in parser[parser.sections()[0]].items()}


@dataclass(frozen=True)
class Settings:
    snowflake_account: Optional[str]
    snowflake_user: Optional[str]
    snowflake_role: Optional[str]
    snowflake_warehouse: Optional[str]
    snowflake_database: Optional[str]
    snowflake_schema: Optional[str]
    snowflake_token_file: Path
    feature_publish_target: str
    feature_repository: str
    feature_snapshot_root: Path
    feature_sql_path: Path
    feature_schema_path: Path
    model_path: Path
    model_card_path: Path
    max_feature_age_hours: int
    max_batch_size: int
    snapshot_check_interval_seconds: int

    @classmethod
    def load(cls) -> "Settings":
        file_cfg = _file_connection()
        get = lambda env, key=None: _clean(os.getenv(env)) or file_cfg.get(key or env.lower())
        settings = cls(
            snowflake_account=get("SNOWFLAKE_ACCOUNT", "account"),
            snowflake_user=get("SNOWFLAKE_USER", "user"),
            snowflake_role=get("SNOWFLAKE_ROLE", "role"),
            snowflake_warehouse=get("SNOWFLAKE_WAREHOUSE", "warehouse"),
            snowflake_database=get("SNOWFLAKE_DATABASE", "database"),
            snowflake_schema=get("SNOWFLAKE_SCHEMA", "schema"),
            snowflake_token_file=Path(os.getenv(
                "SNOWFLAKE_TOKEN_FILE", ROOT / "schedule_model_dev-token-secret.txt"
            )),
            feature_publish_target=os.getenv("FEATURE_PUBLISH_TARGET", "local").lower(),
            feature_repository=os.getenv("FEATURE_REPOSITORY", "local").lower(),
            feature_snapshot_root=Path(os.getenv(
                "FEATURE_SNAPSHOT_ROOT", ROOT / "feature_snapshots"
            )),
            feature_sql_path=Path(os.getenv(
                "FEATURE_SQL_PATH", ROOT / "schedule_risk_feature_calculation.sql"
            )),
            feature_schema_path=Path(os.getenv(
                "FEATURE_SCHEMA_PATH", ROOT / "models/schedule_risk_feature_schema.json"
            )),
            model_path=Path(os.getenv(
                "MODEL_PATH", ROOT / "models/schedule_risk_model.joblib"
            )),
            model_card_path=Path(os.getenv(
                "MODEL_CARD_PATH", ROOT / "models/schedule_risk_model_card.json"
            )),
            max_feature_age_hours=int(os.getenv("MAX_FEATURE_AGE_HOURS", "24")),
            max_batch_size=int(os.getenv("MAX_BATCH_SIZE", "500")),
            snapshot_check_interval_seconds=int(
                os.getenv("SNAPSHOT_CHECK_INTERVAL_SECONDS", "60")
            ),
        )
        if settings.feature_publish_target not in {"local", "snowflake"}:
            raise ConfigurationError("FEATURE_PUBLISH_TARGET must be local or snowflake")
        if settings.feature_repository not in {"local", "snowflake"}:
            raise ConfigurationError("FEATURE_REPOSITORY must be local or snowflake")
        return settings

    def require_snowflake(self) -> None:
        missing = [
            name for name, value in {
                "SNOWFLAKE_ACCOUNT": self.snowflake_account,
                "SNOWFLAKE_USER": self.snowflake_user,
                "SNOWFLAKE_ROLE": self.snowflake_role,
                "SNOWFLAKE_WAREHOUSE": self.snowflake_warehouse,
            }.items() if not value
        ]
        if missing:
            raise ConfigurationError("Missing Snowflake settings: " + ", ".join(missing))
        if not self.snowflake_token_file.is_file():
            raise ConfigurationError(
                "Snowflake token file not found: " + str(self.snowflake_token_file)
            )

