from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

from .contracts import StageError


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def write_checksums(directory: Path, names: Iterable[str]) -> Dict[str, str]:
    checksums = {}
    for name in sorted(names):
        checksums[name] = sha256_file(Path(directory) / name)
    content = "".join("{}  {}\n".format(value, name) for name, value in checksums.items())
    target = Path(directory) / "checksums.sha256"
    temporary = target.with_name(".checksums.sha256.tmp")
    temporary.write_text(content, encoding="ascii")
    os.replace(str(temporary), str(target))
    return checksums


def verify_checksums(directory: Path, checksum_path: Optional[Path] = None) -> None:
    directory = Path(directory)
    checksum_path = checksum_path or directory / "checksums.sha256"
    if not checksum_path.is_file():
        raise StageError("Missing checksum file: {}".format(checksum_path))
    for line in checksum_path.read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(None, 1)
        name = name.strip()
        actual = sha256_file(directory / name)
        if actual != expected:
            raise StageError("Checksum mismatch for {}".format(name))


def environment_metadata() -> Dict[str, Any]:
    packages = {}
    for module_name in ("numpy", "pandas", "pyarrow", "sklearn", "joblib"):
        try:
            module = __import__(module_name)
            packages[module_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            packages[module_name] = "unavailable: {}".format(type(exc).__name__)
    return {
        "captured_at_utc": utc_now(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }


def make_run_id(run_name: str, config_hash: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "-", run_name)
    return "{}-{}-{}".format(safe_name, compact_utc_now(), config_hash[:8])


@contextmanager
def stage_status(
    run_dir: Path,
    stage_name: str,
    input_hashes: Optional[Dict[str, str]] = None,
) -> Iterator[Path]:
    stage_dir = Path(run_dir) / "stages" / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    status_path = stage_dir / "stage_status.json"
    started = utc_now()
    write_json_atomic(status_path, {
        "schema_version": "schedule-stage-status-v1",
        "stage": stage_name,
        "status": "running",
        "started_at_utc": started,
        "input_hashes": input_hashes or {},
    })
    try:
        yield stage_dir
    except Exception as exc:
        write_json_atomic(status_path, {
            "schema_version": "schedule-stage-status-v1",
            "stage": stage_name,
            "status": "failed",
            "started_at_utc": started,
            "finished_at_utc": utc_now(),
            "input_hashes": input_hashes or {},
            "error_code": getattr(exc, "code", type(exc).__name__),
            "error": str(exc),
        })
        raise
    else:
        write_json_atomic(status_path, {
            "schema_version": "schedule-stage-status-v1",
            "stage": stage_name,
            "status": "succeeded",
            "started_at_utc": started,
            "finished_at_utc": utc_now(),
            "input_hashes": input_hashes or {},
        })
