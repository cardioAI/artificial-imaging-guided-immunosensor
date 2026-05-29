
from typing import List, Optional

import matplotlib.pyplot as plt

class CardioAIUtils:
    def __init__(self):
        pass

    def setup_matplotlib_style(self):
        return

    def get_colors(self, n: int = 10, category: str = "primary") -> List[str]:
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        return [palette[i % len(palette)] for i in range(n)]

    def get_colormap(self, name: str = "cardioai_primary"):
        return plt.get_cmap("viridis")

    def save_figure(self, fig, filepath: str, formats: Optional[List[str]] = None):
        return

    def setup_axis_clean(self, ax, title=None, xlabel=None, ylabel=None):
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

    def create_legend_clean(self, ax, **kwargs):
        return ax.legend(**kwargs)

    def enhance_figure_quality(self, fig):
        return

    def create_enhanced_heatmap(self, data, **kwargs):
        return plt.imshow(data, **kwargs)

cardioai_utils = CardioAIUtils()

def setup_cardioai_style():
    return

def get_cardioai_colors(n: int = 10, category: str = "primary"):
    return cardioai_utils.get_colors(n, category)

def save_cardioai_figure(fig, filepath: str, formats: Optional[List[str]] = None):
    return

def get_cardioai_colormap(name: str = "cardioai_primary"):
    return cardioai_utils.get_colormap(name)

def setup_clean_axis(ax, title=None, xlabel=None, ylabel=None):
    cardioai_utils.setup_axis_clean(ax, title, xlabel, ylabel)

def create_clean_legend(ax, **kwargs):
    return cardioai_utils.create_legend_clean(ax, **kwargs)

def enhance_figure(fig):
    cardioai_utils.enhance_figure_quality(fig)

def create_enhanced_heatmap(data, **kwargs):
    return cardioai_utils.create_enhanced_heatmap(data, **kwargs)
