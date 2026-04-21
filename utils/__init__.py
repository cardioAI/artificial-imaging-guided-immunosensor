"""
utils subpackage.

Public plotting / styling helpers shared by every visual module in the
project. Re-exports the most common symbols from :mod:`utils.styling`
so callers can write ``from utils import save_cardioai_figure`` instead
of the longer qualified path.
"""

from .styling import (
    CardioAIUtils,
    cardioai_utils,
    setup_cardioai_style,
    get_cardioai_colors,
    save_cardioai_figure,
    get_cardioai_colormap,
    setup_clean_axis,
    create_clean_legend,
    enhance_figure,
    create_enhanced_heatmap,
)

__all__ = [
    "CardioAIUtils",
    "cardioai_utils",
    "setup_cardioai_style",
    "get_cardioai_colors",
    "save_cardioai_figure",
    "get_cardioai_colormap",
    "setup_clean_axis",
    "create_clean_legend",
    "enhance_figure",
    "create_enhanced_heatmap",
]
