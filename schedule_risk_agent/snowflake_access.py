from __future__ import annotations

import snowflake.connector
import os

from .config import Settings


def connect(settings: Settings, token=None):
    settings.require_snowflake()
    token_value = token or settings.snowflake_token_file.read_text(encoding="utf-8").strip()
    args = {
        "account": settings.snowflake_account,
        "user": settings.snowflake_user,
        "role": settings.snowflake_role,
        "warehouse": settings.snowflake_warehouse,
        "authenticator": "PROGRAMMATIC_ACCESS_TOKEN",
        "token": token_value,
        "login_timeout": 20,
        "network_timeout": int(os.getenv("SNOWFLAKE_NETWORK_TIMEOUT_SECONDS", "1800")),
        "application": "schedule_risk_agent",
        "session_parameters": {"QUERY_TAG": "schedule-risk-agent"},
        "database": settings.snowflake_database or os.getenv("SNOWFLAKE_TEMP_DATABASE", "UDOTAIML"),
        "schema": settings.snowflake_schema or os.getenv("SNOWFLAKE_TEMP_SCHEMA", "DBO"),
    }
    if settings.snowflake_database:
        args["database"] = settings.snowflake_database
    if settings.snowflake_schema:
        args["schema"] = settings.snowflake_schema
    return snowflake.connector.connect(**args)

