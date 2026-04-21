"""
training subpackage.

Exports :class:`CardioAITrainer` and helpers so callers can do
``from training import CardioAITrainer`` without knowing the submodule.
"""

from .trainer import CardioAITrainer, parse_patient_range, create_custom_split

__all__ = ["CardioAITrainer", "parse_patient_range", "create_custom_split"]
