"""
prepare_cleaned.py
==================

Restructure F:/datasets_cardioAI/BICL_cardioAI/raw into the layout expected by
data/dicom_processor.py:

    cleaned/
      Patient_<ID>/
        dicom/          <- all *.dcm from the 4 vibe_dixon_2echo_abdomen_* series
        other/          <- NIfTI masks from Anatomical/

Uses NTFS hard links when source and destination live on the same volume so no
bytes are copied. Falls back to copy if linking fails.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

RAW_DIR = Path(r"F:\datasets_cardioAI\BICL_cardioAI\raw")
CLEANED_DIR = Path(r"F:\datasets_cardioAI\BICL_cardioAI\cleaned")

NIFTI_FILES = [
    "masked_2echo_fat_errosion.nii.gz",
    "sat_mask.nii.gz",
    "vat_mask.nii.gz",
    "vat_mask_vb.nii.gz",
]


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_patient(patient_dir: Path) -> tuple[int, int]:
    patient_id = patient_dir.name
    out = CLEANED_DIR / f"Patient_{patient_id}"
    dicom_out = out / "dicom"
    other_out = out / "other"
    dicom_out.mkdir(parents=True, exist_ok=True)
    other_out.mkdir(parents=True, exist_ok=True)

    dicom_count = 0
    for series_dir in sorted(patient_dir.iterdir()):
        if not series_dir.is_dir():
            continue
        if "vibe_dixon" not in series_dir.name:
            continue
        for dcm in series_dir.glob("*.dcm"):
            if dcm.name.startswith("._"):
                continue
            link_or_copy(dcm, dicom_out / dcm.name)
            dicom_count += 1

    anat = patient_dir / "Anatomical"
    nifti_count = 0
    if anat.is_dir():
        for name in NIFTI_FILES:
            src = anat / name
            if src.exists():
                link_or_copy(src, other_out / name)
                nifti_count += 1

    return dicom_count, nifti_count


def main() -> int:
    if not RAW_DIR.is_dir():
        print(f"ERROR: raw dir not found: {RAW_DIR}")
        return 1

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    patients = sorted(p for p in RAW_DIR.iterdir() if p.is_dir())
    print(f"Restructuring {len(patients)} patients -> {CLEANED_DIR}")

    total_dcm = 0
    total_nii = 0
    for i, p in enumerate(patients, 1):
        try:
            d, n = prepare_patient(p)
        except Exception as exc:
            print(f"[{i}/{len(patients)}] {p.name} FAILED: {exc}")
            continue
        total_dcm += d
        total_nii += n
        if i % 10 == 0 or i == len(patients):
            print(f"[{i}/{len(patients)}] processed; cumulative dcm={total_dcm} nii={total_nii}")

    print(f"Done. dcm={total_dcm} nii={total_nii}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
