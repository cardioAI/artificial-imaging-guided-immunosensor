"""
utils.styling
=============

CardioAI plotting, palette, and figure-saving utilities -- the single source
of truth for every visual produced by the project.

Contents
--------
* ``CardioAIUtils`` -- the underlying worker class. Handles palette loading
  from ``palette.jpg`` (42 colours + white/black/grays), fallback palettes
  when the image is unavailable, matplotlib Nature-style configuration,
  multi-format figure saving (EPS / PNG 300 dpi / TIFF 300 dpi), clean
  axis setup (no grid, consistent tick styling) and enhanced heatmap plots.
* Module-level convenience wrappers around a shared singleton:
  ``setup_cardioai_style``, ``get_cardioai_colors``, ``save_cardioai_figure``,
  ``get_cardioai_colormap``, ``setup_clean_axis``, ``create_clean_legend``,
  ``enhance_figure``, ``create_enhanced_heatmap``.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from PIL import Image
import warnings
from typing import List, Tuple, Optional, Dict, Any

# Suppress warnings
warnings.filterwarnings('ignore')


class CardioAIUtils:
    """
    Centralized utilities for CardioAI project
    """

    def __init__(self):
        """Initialize CardioAI utilities"""
        self.palette_loaded = False
        self.setup_matplotlib_style()
        self.load_color_palette()

    def load_color_palette(self):
        """Load and extract 42 colors from palette.jpg"""
        try:
            # Find palette.jpg in current directory or project root
            palette_path = Path("./palette.jpg")
            if not palette_path.exists():
                # Try parent directory
                palette_path = Path("../palette.jpg")
            if not palette_path.exists():
                raise FileNotFoundError("palette.jpg not found")

            # Open and process palette image
            palette_img = Image.open(palette_path)
            palette_array = np.array(palette_img)

            # Extract colors from 6x7 grid (42 colors)
            colors = []
            rows, cols = 6, 7
            h, w = palette_array.shape[:2]
            cell_h, cell_w = h // rows, w // cols

            for row in range(rows):
                for col in range(cols):
                    # Get center pixel of each cell
                    y = row * cell_h + cell_h // 2
                    x = col * cell_w + cell_w // 2
                    color = palette_array[y, x]

                    # Convert to matplotlib format (0-1 range)
                    if len(color) == 3:  # RGB
                        colors.append(tuple(color / 255.0))
                    else:  # RGBA
                        colors.append(tuple(color[:3] / 255.0))

            # Store 42 base colors
            self.base_colors = colors

            # Create extended color palette
            self.create_extended_palette()

            self.palette_loaded = True
            print(f"CardioAI Utils: Loaded {len(self.base_colors)} colors from palette.jpg")

        except Exception as e:
            print(f"Warning: Could not load palette.jpg: {e}")
            # Fallback to a sophisticated color scheme
            self.create_fallback_palette()

    def create_extended_palette(self):
        """Create extended color palette with variations"""
        # Base 42 colors
        colors = list(self.base_colors)

        # Add darker variations (multiply by 0.6)
        dark_colors = [tuple(c * 0.6 for c in color) for color in self.base_colors]

        # Add lighter variations (blend with white)
        light_colors = [tuple(c * 0.7 + 0.3 for c in color) for color in self.base_colors]

        # Add neutral colors (whites, blacks, grays)
        neutral_colors = [
            (1.0, 1.0, 1.0),    # Pure white
            (0.0, 0.0, 0.0),    # Pure black
            (0.05, 0.05, 0.05), # Almost black
            (0.1, 0.1, 0.1),    # Very dark gray
            (0.2, 0.2, 0.2),    # Dark gray
            (0.3, 0.3, 0.3),    # Medium dark gray
            (0.4, 0.4, 0.4),    # Medium gray
            (0.5, 0.5, 0.5),    # Gray
            (0.6, 0.6, 0.6),    # Medium light gray
            (0.7, 0.7, 0.7),    # Light gray
            (0.8, 0.8, 0.8),    # Very light gray
            (0.85, 0.85, 0.85), # Pale gray
            (0.9, 0.9, 0.9),    # Very pale gray
            (0.95, 0.95, 0.95), # Almost white
        ]

        # Organize color categories
        self.primary_colors = colors[:21]        # First 21 for main data
        self.secondary_colors = colors[21:42]    # Remaining 21 for secondary data
        self.dark_variants = dark_colors
        self.light_variants = light_colors
        self.neutral_colors = neutral_colors
        self.all_colors = colors + dark_colors + light_colors + neutral_colors

        print(f"CardioAI Utils: Created extended palette with {len(self.all_colors)} total colors")

    def create_fallback_palette(self):
        """Create fallback color palette if palette.jpg is not available"""
        # Professional color scheme with good contrast
        base_colors = [
            # Teal/Cyan family
            (0.0, 0.7, 0.8),   (0.1, 0.8, 0.9),   (0.2, 0.9, 1.0),
            (0.0, 0.6, 0.7),   (0.1, 0.7, 0.8),   (0.2, 0.8, 0.9),
            (0.0, 0.5, 0.6),
            # Green family
            (0.2, 0.8, 0.3),   (0.3, 0.9, 0.4),   (0.4, 1.0, 0.5),
            (0.1, 0.7, 0.2),   (0.2, 0.8, 0.3),   (0.3, 0.9, 0.4),
            (0.0, 0.6, 0.1),
            # Orange family
            (1.0, 0.6, 0.0),   (1.0, 0.7, 0.1),   (1.0, 0.8, 0.2),
            (0.9, 0.5, 0.0),   (0.9, 0.6, 0.1),   (0.9, 0.7, 0.2),
            (0.8, 0.4, 0.0),
            # Purple family
            (0.6, 0.2, 0.8),   (0.7, 0.3, 0.9),   (0.8, 0.4, 1.0),
            (0.5, 0.1, 0.7),   (0.6, 0.2, 0.8),   (0.7, 0.3, 0.9),
            (0.4, 0.0, 0.6),
            # Red family
            (0.8, 0.2, 0.2),   (0.9, 0.3, 0.3),   (1.0, 0.4, 0.4),
            (0.7, 0.1, 0.1),   (0.8, 0.2, 0.2),   (0.9, 0.3, 0.3),
            (0.6, 0.0, 0.0),
            # Blue family
            (0.2, 0.4, 0.8),   (0.3, 0.5, 0.9),   (0.4, 0.6, 1.0),
            (0.1, 0.3, 0.7),   (0.2, 0.4, 0.8),   (0.3, 0.5, 0.9),
            (0.0, 0.2, 0.6),
        ]

        self.base_colors = base_colors[:42]  # Ensure exactly 42 colors
        self.create_extended_palette()

        print("CardioAI Utils: Using fallback color palette")

    def setup_matplotlib_style(self):
        """Setup matplotlib for Nature journal style"""
        # Reset to default first
        plt.style.use('default')

        # Nature journal parameters
        plt.rcParams.update({
            # Figure and DPI
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'figure.figsize': (7, 5),

            # Fonts - Nature journal standard
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
            'font.size': 8,
            'axes.titlesize': 10,
            'axes.labelsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'figure.titlesize': 12,

            # Lines and markers
            'lines.linewidth': 1.5,
            'lines.markersize': 4,
            'patch.linewidth': 0.5,

            # Axes styling
            'axes.linewidth': 0.8,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': False,  # NO GRIDS
            'axes.axisbelow': True,
            'axes.edgecolor': 'black',
            'axes.labelcolor': 'black',

            # Ticks
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'xtick.minor.width': 0.4,
            'ytick.minor.width': 0.4,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'xtick.color': 'black',
            'ytick.color': 'black',

            # Legend
            'legend.frameon': True,
            'legend.framealpha': 1.0,
            'legend.fancybox': False,
            'legend.edgecolor': 'black',
            'legend.facecolor': 'white',
            'legend.numpoints': 1,
            'legend.scatterpoints': 1,

            # Saving
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,
            'savefig.transparent': False,
            'savefig.facecolor': 'white',
            'savefig.edgecolor': 'none',
        })

        print("CardioAI Utils: Applied Nature journal matplotlib style")

    def get_colors(self, n: int, category: str = 'primary') -> List[Tuple[float, float, float]]:
        """
        Get n colors from specified category

        Args:
            n: Number of colors needed
            category: 'primary', 'secondary', 'dark', 'light', 'neutral', 'all'

        Returns:
            List of RGB tuples
        """
        if not self.palette_loaded:
            self.load_color_palette()

        if category == 'primary':
            colors = self.primary_colors
        elif category == 'secondary':
            colors = self.secondary_colors
        elif category == 'dark':
            colors = self.dark_variants
        elif category == 'light':
            colors = self.light_variants
        elif category == 'neutral':
            colors = self.neutral_colors
        elif category == 'all':
            colors = self.all_colors
        else:
            colors = self.primary_colors

        # Cycle through colors if we need more than available
        selected_colors = []
        for i in range(n):
            selected_colors.append(colors[i % len(colors)])

        return selected_colors

    def get_colormap(self, name: str = 'cardioai_primary') -> LinearSegmentedColormap:
        """
        Create custom colormap from palette colors

        Args:
            name: Colormap name

        Returns:
            Custom colormap
        """
        if name == 'cardioai_primary':
            colors = self.primary_colors[:10]
        elif name == 'cardioai_heatmap':
            colors = [
                '#001A33', '#003366', '#0066CC', '#1A93FF', '#66B7FF',
                '#B3DBFF', '#F0F8FF', '#FFFFFF', '#FFF8F0', '#FFDBDB',
                '#FF9999', '#FF6666', '#FF3333', '#CC0000', '#990000'
            ]
        elif name == 'cardioai_cool_smooth':
            colors = [
                '#08519c', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef',
                '#e0f2e0', '#a1d99b', '#74c476', '#41ab5d', '#238b45',
                '#006d2c',
            ]
        elif name == 'cardioai_embedding':
            colors = [
                '#08519c', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef',
                '#e0f2e0', '#a1d99b', '#74c476', '#41ab5d', '#238b45',
                '#006d2c',
            ]
        elif name == 'cardioai_sequential':
            colors = self.light_variants[:8]
        else:
            colors = self.primary_colors[:10]

        return LinearSegmentedColormap.from_list(name, colors, N=256)

    def save_figure(self, fig, filepath: str, formats: List[str] = None):
        """
        Save figure in multiple formats with proper naming

        Args:
            fig: Matplotlib figure
            filepath: Base filepath without extension
            formats: List of formats ['eps', 'png', 'tiff']
        """
        if formats is None:
            formats = ['eps', 'png', 'tiff']

        # Ensure directory exists
        output_dir = Path(filepath).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        base_path = Path(filepath).with_suffix('')

        # Save in requested formats
        for fmt in formats:
            if fmt == 'eps':
                fig.savefig(f"{base_path}.eps", format='eps', dpi=300)
            elif fmt == 'png':
                fig.savefig(f"{base_path}.png", format='png', dpi=300)
            elif fmt == 'tiff':
                fig.savefig(f"{base_path}.tiff", format='tiff', dpi=300)

        print(f"Saved figure: {base_path.name} ({', '.join(formats)})")

    def setup_axis_clean(self, ax, title: str = None, xlabel: str = None, ylabel: str = None):
        """
        Setup axis with clean Nature journal styling

        Args:
            ax: Matplotlib axis
            title: Axis title
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # No grids
        ax.grid(False)

        # Set labels with proper spacing
        if title:
            ax.set_title(title, fontweight='bold', pad=15)
        if xlabel:
            ax.set_xlabel(xlabel, fontweight='normal', labelpad=8)
        if ylabel:
            ax.set_ylabel(ylabel, fontweight='normal', labelpad=8)

        # Ensure ticks point outward
        ax.tick_params(axis='both', direction='out', length=4, width=0.8)

    def create_legend_clean(self, ax, **kwargs):
        """
        Create clean legend with proper styling

        Args:
            ax: Matplotlib axis
            **kwargs: Additional legend arguments
        """
        legend_defaults = {
            'frameon': True,
            'fancybox': False,
            'framealpha': 1.0,
            'edgecolor': 'black',
            'facecolor': 'white',
            'fontsize': 8
        }
        legend_defaults.update(kwargs)

        legend = ax.legend(**legend_defaults)
        if legend:
            legend.get_frame().set_linewidth(0.5)

        return legend

    def enhance_figure_quality(self, fig):
        """
        Apply final quality enhancements to figure

        Args:
            fig: Matplotlib figure
        """
        # Tight layout with padding
        fig.tight_layout(pad=2.0)

        # Ensure all text is within bounds
        fig.canvas.draw()

        # Final adjustments
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9)

    def create_enhanced_heatmap(self, data, title: str = "Heatmap",
                                vmin: float = None, vmax: float = None,
                                figsize: Tuple[float, float] = (4, 12),
                                cmap: str = 'cardioai_heatmap'):
        """
        Create enhanced heatmap with CardioAI styling

        Args:
            data: 2D array or 1D array (will be reshaped)
            title: Heatmap title
            vmin, vmax: Color scale limits
            figsize: Figure size
            cmap: Colormap name

        Returns:
            fig, ax: Figure and axis objects
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        fig, ax = plt.subplots(figsize=figsize)

        # Use custom colormap if available
        if cmap == 'cardioai_heatmap':
            colormap = self.get_colormap('cardioai_heatmap')
        else:
            colormap = cmap

        # Create heatmap
        im = ax.imshow(data, cmap=colormap, aspect='auto', vmin=vmin, vmax=vmax)

        # Setup clean styling
        self.setup_axis_clean(ax, title=title, xlabel='Embedding Value', ylabel='Dimension')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.05)
        cbar.set_label('Activation', rotation=270, labelpad=20, fontweight='normal')
        cbar.outline.set_linewidth(0.5)

        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        return fig, ax


# Create global instance
cardioai_utils = CardioAIUtils()


# Convenience functions for easy import
def setup_cardioai_style():
    """Setup CardioAI matplotlib style"""
    cardioai_utils.setup_matplotlib_style()


def get_cardioai_colors(n: int = 10, category: str = 'primary'):
    """Get CardioAI colors"""
    return cardioai_utils.get_colors(n, category)


def save_cardioai_figure(fig, filepath: str, formats: List[str] = None):
    """Save figure in CardioAI standard formats"""
    cardioai_utils.save_figure(fig, filepath, formats)


def get_cardioai_colormap(name: str = 'cardioai_primary'):
    """Get CardioAI colormap"""
    return cardioai_utils.get_colormap(name)


def setup_clean_axis(ax, title: str = None, xlabel: str = None, ylabel: str = None):
    """Setup clean axis styling"""
    cardioai_utils.setup_axis_clean(ax, title, xlabel, ylabel)


def create_clean_legend(ax, **kwargs):
    """Create clean legend"""
    return cardioai_utils.create_legend_clean(ax, **kwargs)


def enhance_figure(fig):
    """Enhance figure quality"""
    cardioai_utils.enhance_figure_quality(fig)


def create_enhanced_heatmap(data, **kwargs):
    """Create enhanced heatmap with CardioAI styling"""
    return cardioai_utils.create_enhanced_heatmap(data, **kwargs)


# Print initialization message
print("CardioAI Utils module loaded successfully")
print("Available functions: setup_cardioai_style, get_cardioai_colors, save_cardioai_figure")
print("Available colors: 42 base + variations + neutrals")
print("Available colormaps: cardioai_primary, cardioai_heatmap, cardioai_cool_smooth, cardioai_embedding")
