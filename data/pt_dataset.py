"""
data.pt_dataset
===============

PyTorch ``Dataset`` wrapping the 108-patient CardioAI cohort. Each item
pairs a normalised six-biomarker vector with a stack of four MRI sequence
tensors loaded from the preprocessed ``.pt`` files under
``F:\\datasets_cardioAI\\BICL_cardioAI\\cleaned\\Pt``.

Contents
--------
* ``CardioAIDataset`` -- standard ``torch.utils.data.Dataset``; handles
  z-score biomarker normalisation, missing image files, dtype coercion
  and returns a dict of tensors per patient.
* ``custom_collate_fn`` -- pads / centre-crops variable volumes to the
  manuscript's canonical ``(4, 32, 96, 96)`` shape before stacking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class CardioAIDataset(Dataset):
    """108-patient CardioAI paired biomarker + MRI dataset."""

    def __init__(self,
                 biomarkers_file: str = "./dataset_cardioAI.xlsx",
                 images_dir: str = r"F:\datasets_cardioAI\BICL_cardioAI\cleaned\Pt",
                 normalize_biomarkers: bool = True):
        self.images_dir = Path(images_dir)
        self.normalize_biomarkers = normalize_biomarkers

        print(f"Loading biomarkers from {biomarkers_file}")
        self.biomarkers_df = pd.read_excel(biomarkers_file)

        self.biomarker_columns = ['ALT', 'AST', 'GGT', 'CTSD', 'CK18', 'FGF21']

        self.patient_ids = []
        self.valid_indices = []

        for idx, row in self.biomarkers_df.iterrows():
            patient_id = f"Patient_{int(row['Meta_ID'])}"
            pt_file = self.images_dir / f"{patient_id}.pt"

            if pt_file.exists():
                self.patient_ids.append(patient_id)
                self.valid_indices.append(idx)
            else:
                print(f"Warning: No image data found for {patient_id} (file: {pt_file})")

        self.biomarkers_df = self.biomarkers_df.iloc[self.valid_indices].reset_index(drop=True)

        self.biomarker_values = self.biomarkers_df[self.biomarker_columns].values.astype(np.float32)

        if self.normalize_biomarkers:
            self.biomarker_mean = np.mean(self.biomarker_values, axis=0)
            self.biomarker_std = np.std(self.biomarker_values, axis=0)
            self.biomarker_values = (self.biomarker_values - self.biomarker_mean) / (self.biomarker_std + 1e-8)

        print(f"Dataset initialized with {len(self.patient_ids)} patients")

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        patient_id = self.patient_ids[idx]
        biomarkers = torch.from_numpy(self.biomarker_values[idx]).float()
        pt_file = self.images_dir / f"{patient_id}.pt"

        try:
            patient_data = torch.load(pt_file, map_location='cpu')
            dicom_sequences = patient_data['dicom_sequences']

            sequence_tensors = []
            for seq_name in sorted(dicom_sequences.keys()):
                sequence_tensors.append(dicom_sequences[seq_name])

            if len(sequence_tensors) >= 4:
                images = torch.stack(sequence_tensors[:4], dim=0)
            else:
                while len(sequence_tensors) < 4:
                    sequence_tensors.append(sequence_tensors[-1])
                images = torch.stack(sequence_tensors, dim=0)

            images = images.float()
            hff_value = self.biomarkers_df.iloc[idx]['HFF']
            label = self.biomarkers_df.iloc[idx]['Label']

        except Exception as e:
            print(f"Error loading {patient_id}: {e}")
            images = torch.zeros(4, 32, 96, 96, dtype=torch.float32)
            hff_value = 0.0
            label = 0

        return {
            'biomarkers': biomarkers,
            'images': images,
            'patient_id': patient_id,
            'hff': torch.tensor(hff_value, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long),
            'index': torch.tensor(idx, dtype=torch.long),
        }


TARGET_SLICES = 32
TARGET_HEIGHT = 96
TARGET_WIDTH = 96


def _pad_or_crop(tensor: torch.Tensor, dim: int, target: int) -> torch.Tensor:
    """Symmetrically pad or centre-crop ``tensor`` along ``dim`` to ``target`` length."""
    size = tensor.size(dim)
    if size == target:
        return tensor
    if size < target:
        delta = target - size
        pre = delta // 2
        post = delta - pre
        # F.pad expects pairs for the *last* dims in reverse; build a tuple.
        pad = [0] * (2 * tensor.dim())
        # pad positions for dim (counted from the last dim)
        rev = tensor.dim() - 1 - dim
        pad[2 * rev] = pre
        pad[2 * rev + 1] = post
        return F.pad(tensor, pad, mode="constant", value=0)
    # size > target -> centre crop
    start = (size - target) // 2
    return tensor.narrow(dim, start, target)


def custom_collate_fn(batch):
    """Pad/resize variable volumes to the manuscript's (4, 32, 96, 96) shape before stacking."""
    biomarkers = torch.stack([item['biomarkers'] for item in batch])
    patient_ids = [item['patient_id'] for item in batch]
    hff_values = torch.stack([item['hff'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    indices = torch.stack([item['index'] for item in batch])

    images_list = []
    for item in batch:
        image = item['images']
        # Expected shape: (4, D, H, W). Pad / crop each dim to the target grid.
        image = _pad_or_crop(image, dim=1, target=TARGET_SLICES)
        image = _pad_or_crop(image, dim=2, target=TARGET_HEIGHT)
        image = _pad_or_crop(image, dim=3, target=TARGET_WIDTH)
        images_list.append(image)

    images = torch.stack(images_list)

    return {
        'biomarkers': biomarkers,
        'images': images,
        'patient_id': patient_ids,
        'hff': hff_values,
        'label': labels,
        'index': indices,
    }
