"""
data subpackage.

Everything that turns raw inputs (DICOM stacks, biomarker spreadsheets, NIfTI
masks) into tensors or visuals consumable by the ML pipeline.
"""

from .pt_dataset import CardioAIDataset, custom_collate_fn

__all__ = ["CardioAIDataset", "custom_collate_fn"]
