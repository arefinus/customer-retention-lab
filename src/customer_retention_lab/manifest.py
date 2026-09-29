"""Run manifest writer (shared contract section 5)."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from . import PROJECT_ID


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_mapping(obj: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def make_run_id(mode: str, started_at: str) -> str:
    return f"{started_at.replace(':', '').replace('-', '').replace('+0000', 'Z')}-{mode}"


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    mode: str
    status: str
    started_at: str
    finished_at: str
    configuration_hash: str
    seed: int
    data: Mapping[str, Any]
    metrics_file: str | None = None
    predictions_file: str | None = None
    evidence_status: str = "demo"
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "project_id": PROJECT_ID,
            "run_id": self.run_id,
            "status": self.status,
            "mode": self.mode,
            "evidence_status": self.evidence_status,
            "git_commit": None,
            "data": dict(self.data),
            "configuration_hash": self.configuration_hash,
            "seed": self.seed,
            "environment": {"python": platform.python_version(), "device": "cpu"},
            "metrics_file": self.metrics_file,
            "predictions_file": self.predictions_file,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "notes": list(self.notes),
        }


def write_manifest(path: Path, manifest: RunManifest) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(manifest.to_dict(), fh, sort_keys=False)
    return path
