"""Repository-relative paths; nothing depends on the machine running the code."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
FIXTURE_DIR = REPO_ROOT / "examples" / "fixtures" / "v1"
OUTPUT_DIR = REPO_ROOT / "examples" / "output"
DOCS_FIGURE_DIR = REPO_ROOT / "docs" / "figures"

ACCOUNTS_FIXTURE = FIXTURE_DIR / "accounts.csv"
ACTIVITY_FIXTURE = FIXTURE_DIR / "activity.csv"
