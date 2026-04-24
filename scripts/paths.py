"""
Rutas centralizadas del repositorio.

Útil para scripts y notebooks (añade `scripts` al path: `sys.path.insert(0, str(REPO_ROOT / "scripts"))`).
"""

from __future__ import annotations

from pathlib import Path

# Raíz del repo (directorio que contiene `scripts/` y `notebooks/`).
REPO_ROOT = Path(__file__).resolve().parents[1]

RAW_CSV_DIR = REPO_ROOT / "src/data/raw/csv"
RAW_JPEG_DIR = REPO_ROOT / "src/data/raw/jpeg"
PROCESSED_DIR = REPO_ROOT / "src/data/processed"
REPORTS_MODELS_DIR = REPO_ROOT / "reports/models"

MANIFEST_ALL = PROCESSED_DIR / "manifest_all.csv"
MANIFEST_TRAIN = PROCESSED_DIR / "manifest_train.csv"
MANIFEST_TEST = PROCESSED_DIR / "manifest_test.csv"
