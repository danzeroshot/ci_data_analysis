#!/usr/bin/env python3
"""Validate Snowflake connectivity using the schedule model PAT and config."""

from __future__ import annotations

import configparser
import sys
from pathlib import Path
from typing import Dict, Optional

import snowflake.connector


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "snowflake_access_config.txt.txt"
TOKEN_PATH = BASE_DIR / "schedule_model_dev-token-secret.txt"
PLACEHOLDER_VALUES = {"", "<none selected>", "none", "null"}


def normalize(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip().strip('"').strip("'")
    if normalized.lower() in PLACEHOLDER_VALUES:
        return None
    return normalized


def load_config(path: Path) -> Dict[str, Optional[str]]:
    parser = configparser.ConfigParser()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    parser.read(path)
    if not parser.sections():
        raise ValueError(f"No connection section found in {path.name}")

    section = parser[parser.sections()[0]]
    config = {key: normalize(value) for key, value in section.items()}
    missing = [key for key in ("account", "user") if not config.get(key)]
    if missing:
        raise ValueError(f"Missing required config value(s): {', '.join(missing)}")
    return config


def load_token(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Token file not found: {path}")
    token = path.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError(f"Token file is empty: {path.name}")
    return token


def main() -> int:
    connection = None
    cursor = None
    try:
        config = load_config(CONFIG_PATH)
        token = load_token(TOKEN_PATH)

        connect_args = {
            "account": config["account"],
            "user": config["user"],
            "authenticator": "PROGRAMMATIC_ACCESS_TOKEN",
            "token": token,
            "login_timeout": 20,
            "network_timeout": 30,
            "application": "schedule_model_connectivity_validator",
        }
        for optional_key in ("role", "warehouse", "database", "schema"):
            if config.get(optional_key):
                connect_args[optional_key] = config[optional_key]

        print("Connecting to Snowflake with PROGRAMMATIC_ACCESS_TOKEN authentication...")
        connection = snowflake.connector.connect(**connect_args)
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT
                1 AS connectivity_check,
                CURRENT_ACCOUNT() AS account_name,
                CURRENT_USER() AS user_name,
                CURRENT_ROLE() AS role_name,
                CURRENT_WAREHOUSE() AS warehouse_name,
                CURRENT_DATABASE() AS database_name,
                CURRENT_SCHEMA() AS schema_name,
                CURRENT_TIMESTAMP() AS server_time
            """
        )
        row = cursor.fetchone()
        columns = [description[0].lower() for description in cursor.description]

        print("Connection succeeded.")
        for name, value in zip(columns, row):
            print(f"  {name}: {value}")
        return 0
    except Exception as exc:
        print(f"Connection validation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
