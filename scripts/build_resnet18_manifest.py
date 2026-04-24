"""
Construye manifiestos CSV (train/test/all) a partir de los CSV de CBIS-DDSM y las rutas JPEG locales.

Ejecutar desde la raíz del repo:
    python scripts/build_resnet18_manifest.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from paths import MANIFEST_ALL, MANIFEST_TEST, MANIFEST_TRAIN, PROCESSED_DIR, RAW_CSV_DIR, RAW_JPEG_DIR

CASE_FILES = [
    ("calc", "train", RAW_CSV_DIR / "calc_case_description_train_set.csv"),
    ("calc", "test", RAW_CSV_DIR / "calc_case_description_test_set.csv"),
    ("mass", "train", RAW_CSV_DIR / "mass_case_description_train_set.csv"),
    ("mass", "test", RAW_CSV_DIR / "mass_case_description_test_set.csv"),
]


def pathology_to_binary(pathology: str) -> tuple[str, int] | None:
    p = str(pathology).strip().upper()
    if p == "MALIGNANT":
        return "MALIGNANT", 1
    if p in {"BENIGN", "BENIGN_WITHOUT_CALLBACK"}:
        return "BENIGN", 0
    return None


def load_cases() -> pd.DataFrame:
    rows = []

    for source, split, csv_path in CASE_FILES:
        df = pd.read_csv(csv_path)
        for _, r in df.iterrows():
            mapped = pathology_to_binary(r.get("pathology"))
            if mapped is None:
                continue

            case_key = str(r["image file path"]).split("/")[0]
            label_name, label = mapped

            rows.append(
                {
                    "source": source,
                    "split": split,
                    "case_key": case_key,
                    "patient_id": r.get("patient_id"),
                    "laterality": r.get("left or right breast"),
                    "view": r.get("image view"),
                    "pathology_original": r.get("pathology"),
                    "label_name": label_name,
                    "label": label,
                }
            )

    return pd.DataFrame(rows)


def load_full_mammograms() -> pd.DataFrame:
    dicom = pd.read_csv(RAW_CSV_DIR / "dicom_info.csv", low_memory=False)
    full = dicom[dicom["SeriesDescription"].eq("full mammogram images")].copy()

    # Example in CSV: CBIS-DDSM/jpeg/<uid>/<file>.jpg
    # Local files are under: src/data/raw/jpeg/<uid>/<file>.jpg
    full["image_rel"] = full["image_path"].str.replace("CBIS-DDSM/jpeg/", "", regex=False)
    full["image_path_local"] = full["image_rel"].map(lambda p: (RAW_JPEG_DIR / str(p)).as_posix())

    return full[["PatientID", "image_path_local"]].rename(columns={"PatientID": "case_key"})


def build_manifest() -> pd.DataFrame:
    cases = load_cases()
    full_mammo = load_full_mammograms()

    manifest = cases.merge(full_mammo, on="case_key", how="left")

    # Keep only rows where full mammogram was found on disk.
    manifest = manifest[manifest["image_path_local"].notna()].copy()
    manifest["exists"] = manifest["image_path_local"].map(lambda p: Path(p).exists())
    manifest = manifest[manifest["exists"]].drop(columns=["exists"])

    # Deterministic order for reproducibility.
    manifest = manifest.sort_values(["split", "source", "case_key"]).reset_index(drop=True)
    return manifest


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_manifest = build_manifest()
    train_manifest = all_manifest[all_manifest["split"].eq("train")].copy()
    test_manifest = all_manifest[all_manifest["split"].eq("test")].copy()

    all_manifest.to_csv(MANIFEST_ALL, index=False)
    train_manifest.to_csv(MANIFEST_TRAIN, index=False)
    test_manifest.to_csv(MANIFEST_TEST, index=False)

    print("Saved:")
    print(f"  - {MANIFEST_ALL.as_posix()}")
    print(f"  - {MANIFEST_TRAIN.as_posix()}")
    print(f"  - {MANIFEST_TEST.as_posix()}")

    print("\nCounts by split and label:")
    print(all_manifest.groupby(["split", "label_name"]).size())


if __name__ == "__main__":
    main()
