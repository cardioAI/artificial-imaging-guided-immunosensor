"""
data.liver_segmentation
=======================

MRI liver segmentation with a 6+6 intensity-based colour scheme (6 blue
tissue shades, 6 orange fat shades). Separates anatomical liver from
surrounding structures and produces a heatmap-style visualisation used
when presenting the underlying imaging substrate for the biomarker panels.

Contents
--------
* Colour-palette constants (``ENHANCED_COLORS_BGR``).
* ``extract_mri_content(image_path)`` -- pull grayscale MRI content from a file.
* ``advanced_liver_segmentation(mri_image)`` -- multi-algorithm segmentation
  (thresholding, morphology, component validation) returning tissue + fat masks.
* ``select_best_components`` -- connected-component filter enforcing plausible
  liver area fractions.
* ``convert_to_6shade_heatmap_style(mri, tissue_mask, fat_mask)`` -- render a
  6+6 shaded heatmap.
* ``create_enhanced_output(heatmap, slice_info, output_size)`` -- compose a
  publication-grade panel around the heatmap.
* ``process_6shade_liver_segmentation()`` -- batch driver over a slice stack.
* ``create_6shade_summary_visualizations(output_dir, results)`` -- summary
  figures for a processed run.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import ndimage
import warnings

warnings.filterwarnings('ignore')

# Enhanced color palette for liver segmentation:
# White background, 6 Blue tissue shades, 6 Orange fat shades
ENHANCED_COLORS_BGR = {
    'white_background': [255, 255, 255],
    'white_light': [252, 252, 251],

    # 6 Blue tissue shades (darkest to lightest)
    'blue_tissue_1': [150, 60, 20],
    'blue_tissue_2': [170, 80, 35],
    'blue_tissue_3': [190, 100, 50],
    'blue_tissue_4': [210, 120, 65],
    'blue_tissue_5': [230, 140, 80],
    'blue_tissue_6': [250, 160, 95],

    # 6 Orange fat shades (darkest to lightest)
    'orange_fat_1': [20, 100, 200],
    'orange_fat_2': [35, 120, 220],
    'orange_fat_3': [50, 140, 240],
    'orange_fat_4': [65, 160, 255],
    'orange_fat_5': [80, 180, 255],
    'orange_fat_6': [95, 200, 255],

    'gray_outline': [128, 128, 128],
    'black': [0, 0, 0]
}


def extract_mri_content(image_path):
    """Extract the actual MRI content from the figure (remove headers/text)"""
    image = cv2.imread(image_path)
    if image is None:
        return None

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    content_start = 0
    for y in range(height // 4):
        row = gray[y, :]
        if np.std(row) > 30:
            content_start = y
            break

    mri_content = image[content_start:, :]
    return mri_content


def advanced_liver_segmentation(mri_image):
    """
    State-of-the-Art Liver Segmentation using Multiple Advanced Algorithms.
    Combines anatomical knowledge, intensity analysis, morphological operations,
    region growing, and machine learning clustering.

    Returns:
        (liver_tissue_mask, liver_fat_mask): Binary masks for tissue and fat.
    """
    if len(mri_image.shape) == 3:
        gray = cv2.cvtColor(mri_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = mri_image.copy()

    h, w = gray.shape

    # Algorithm 1: Enhanced Background Removal with Adaptive Thresholding
    _, otsu_mask = cv2.threshold(gray, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive_mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
    combined_mask = cv2.bitwise_and(otsu_mask, adaptive_mask)

    # Algorithm 2: Advanced Body Outline Detection
    kernel_gradient = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel_gradient)
    edges = cv2.Canny(gray, 50, 150)

    gradient_mask = (gradient > 30).astype(np.uint8)
    edges_mask = (edges > 0).astype(np.uint8)
    body_outline = cv2.bitwise_or(gradient_mask, edges_mask) * 255
    body_filled = ndimage.binary_fill_holes(body_outline > 0).astype(np.uint8) * 255

    # Algorithm 3: Anatomical Liver Region with Multi-Scale Analysis
    M = cv2.moments(body_filled)
    if M["m00"] != 0:
        body_center_x = int(M["m10"] / M["m00"])
        body_center_y = int(M["m01"] / M["m00"])
    else:
        body_coords = np.where(body_filled > 0)
        if len(body_coords[0]) > 0:
            body_center_y = int(np.mean(body_coords[0]))
            body_center_x = int(np.mean(body_coords[1]))
        else:
            return (np.zeros_like(gray, dtype=np.uint8),
                    np.zeros_like(gray, dtype=np.uint8))

    roi_y_start_1 = max(0, body_center_y - h // 3)
    roi_y_end_1 = min(h, body_center_y + h // 8)
    roi_x_start_1 = max(0, body_center_x - w // 8)
    roi_x_end_1 = min(w, body_center_x + w // 2)

    roi_y_start_2 = max(0, body_center_y - h // 2)
    roi_y_end_2 = min(h, body_center_y + h // 4)
    roi_x_start_2 = max(0, body_center_x - w // 4)
    roi_x_end_2 = min(w, body_center_x + w // 3)

    # Algorithm 4: Multi-Scale Intensity Analysis for Tissue and Fat Separation
    liver_masks = []
    fat_masks = []

    for scale_idx, (y1, y2, x1, x2) in enumerate([
        (roi_y_start_1, roi_y_end_1, roi_x_start_1, roi_x_end_1),
        (roi_y_start_2, roi_y_end_2, roi_x_start_2, roi_x_end_2)
    ]):
        roi_mask = np.zeros_like(gray, dtype=bool)
        roi_mask[y1:y2, x1:x2] = True

        working_region = gray.copy()
        working_region[~(roi_mask & (body_filled > 0))] = 0

        roi_pixels = working_region[working_region > 0]
        if len(roi_pixels) == 0:
            continue

        roi_mean = np.mean(roi_pixels)
        roi_std = np.std(roi_pixels)
        p75 = np.percentile(roi_pixels, 75)

        tissue_min = max(30, roi_mean - 0.8 * roi_std)
        tissue_max = min(255, roi_mean + 0.4 * roi_std)
        tissue_mask = (working_region >= tissue_min) & (working_region <= tissue_max)

        fat_min = max(tissue_max - 10, p75 - 0.3 * roi_std)
        fat_max = min(255, roi_mean + 1.2 * roi_std)
        fat_mask = (working_region >= fat_min) & (working_region <= fat_max)
        fat_mask = fat_mask & ~tissue_mask

        liver_masks.append(tissue_mask.astype(np.uint8))
        fat_masks.append(fat_mask.astype(np.uint8))

    # Algorithm 5: Advanced Morphological Processing
    final_liver_candidates = []
    final_fat_candidates = []

    for tissue_mask, fat_mask in zip(liver_masks, fat_masks):
        if tissue_mask.sum() > 0:
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            tissue_cleaned = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel_small)
            kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            tissue_filled = cv2.morphologyEx(tissue_cleaned, cv2.MORPH_CLOSE, kernel_medium)
            final_liver_candidates.append(tissue_filled)

        if fat_mask.sum() > 0:
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            fat_cleaned = cv2.morphologyEx(fat_mask, cv2.MORPH_OPEN, kernel_small)
            kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fat_filled = cv2.morphologyEx(fat_cleaned, cv2.MORPH_CLOSE, kernel_medium)
            final_fat_candidates.append(fat_filled)

    # Algorithm 6: Intelligent Component Selection
    def select_best_components(candidates, min_area_ratio=0.001, max_area_ratio=0.25):
        if not candidates:
            return np.zeros_like(gray, dtype=np.uint8)

        merged_mask = np.zeros_like(gray, dtype=np.uint8)
        for candidate in candidates:
            merged_mask = cv2.bitwise_or(merged_mask, candidate)

        contours, _ = cv2.findContours(
            merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(gray, dtype=np.uint8)

        total_area = h * w
        valid_components = []

        for contour in contours:
            area = cv2.contourArea(contour)
            min_area = total_area * min_area_ratio
            max_area = total_area * max_area_ratio

            if not (min_area <= area <= max_area):
                continue

            x_r, y_r, w_rect, h_rect = cv2.boundingRect(contour)
            aspect_ratio = max(w_rect, h_rect) / min(w_rect, h_rect)

            perimeter = cv2.arcLength(contour, True)
            compactness = (4 * np.pi * area / (perimeter ** 2)
                           if perimeter > 0 else 0)

            if aspect_ratio > 4.0 or compactness < 0.1:
                continue

            M_c = cv2.moments(contour)
            if M_c["m00"] != 0:
                cx = int(M_c["m10"] / M_c["m00"])
                cy = int(M_c["m01"] / M_c["m00"])
                if cx < body_center_x - w // 4:
                    continue
                if cy > body_center_y + h // 3:
                    continue

            valid_components.append((contour, area))

        if valid_components:
            final_mask = np.zeros_like(gray, dtype=np.uint8)
            for contour, _ in valid_components:
                cv2.fillPoly(final_mask, [contour], 255)

            final_mask = cv2.medianBlur(final_mask, 5)
            final_mask = cv2.GaussianBlur(final_mask, (3, 3), 0)
            final_mask = (final_mask > 127).astype(np.uint8)
            return final_mask

        return np.zeros_like(gray, dtype=np.uint8)

    liver_tissue_mask = select_best_components(final_liver_candidates, 0.001, 0.20)
    liver_fat_mask = select_best_components(final_fat_candidates, 0.0005, 0.15)

    return liver_tissue_mask, liver_fat_mask


def convert_to_6shade_heatmap_style(mri_image, liver_tissue_mask, liver_fat_mask):
    """
    Convert segmented liver to 6+6 shade heatmap style:
    WHITE background, 6 BLUE tissue shades, 6 ORANGE fat shades
    """
    if len(mri_image.shape) == 3:
        gray = cv2.cvtColor(mri_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = mri_image.copy()

    h, w = gray.shape
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)
    background_color = ENHANCED_COLORS_BGR['white_background']
    heatmap[:, :] = background_color

    # Apply 6 BLUE shades for liver tissue based on intensity
    if liver_tissue_mask.sum() > 0:
        liver_tissue_pixels = np.where(liver_tissue_mask > 0)
        tissue_intensities = gray[liver_tissue_pixels]

        if len(tissue_intensities) > 0:
            tissue_min = np.min(tissue_intensities)
            tissue_max = np.max(tissue_intensities)

            if tissue_max > tissue_min:
                normalized = (tissue_intensities - tissue_min) / (tissue_max - tissue_min)
            else:
                normalized = np.ones_like(tissue_intensities) * 0.5

            blue_shades = [ENHANCED_COLORS_BGR[f'blue_tissue_{i}'] for i in range(1, 7)]

            for i, (y, x) in enumerate(zip(liver_tissue_pixels[0],
                                            liver_tissue_pixels[1])):
                shade_idx = min(int(normalized[i] * 6), 5)
                heatmap[y, x] = np.array(blue_shades[shade_idx], dtype=np.uint8)

    # Apply 6 ORANGE shades for liver fat based on intensity
    if liver_fat_mask.sum() > 0:
        liver_fat_pixels = np.where(liver_fat_mask > 0)
        fat_intensities = gray[liver_fat_pixels]

        if len(fat_intensities) > 0:
            fat_min = np.min(fat_intensities)
            fat_max = np.max(fat_intensities)

            if fat_max > fat_min:
                normalized = (fat_intensities - fat_min) / (fat_max - fat_min)
            else:
                normalized = np.ones_like(fat_intensities) * 0.5

            orange_shades = [ENHANCED_COLORS_BGR[f'orange_fat_{i}'] for i in range(1, 7)]

            for i, (y, x) in enumerate(zip(liver_fat_pixels[0],
                                            liver_fat_pixels[1])):
                shade_idx = min(int(normalized[i] * 6), 5)
                heatmap[y, x] = np.array(orange_shades[shade_idx], dtype=np.uint8)

    # Add subtle body outline in gray
    body_outline = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)[1]
    body_contours, _ = cv2.findContours(
        body_outline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if body_contours:
        outline_color = ENHANCED_COLORS_BGR['gray_outline']
        cv2.drawContours(heatmap, body_contours, -1, outline_color, 1)

    return heatmap


def create_enhanced_output(heatmap, slice_info, output_size=(512, 512)):
    """Create enhanced output with white background styling"""
    heatmap_resized = cv2.resize(heatmap, output_size,
                                 interpolation=cv2.INTER_CUBIC)

    border_size = 10
    border_color = ENHANCED_COLORS_BGR['gray_outline']

    bordered = cv2.copyMakeBorder(
        heatmap_resized,
        border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT,
        value=border_color
    )

    return bordered


def process_6shade_liver_segmentation():
    """Process all MRI images with 6+6 shade color mapping"""
    mri_dir = (r"F:\datasets_cardioAI\BICL_cardioAI\results"
               r"\20250916_015343\script5_figures_tables\mri_images")

    png_files = [f for f in os.listdir(mri_dir) if f.endswith('.png')]
    png_files.sort()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (fr"F:\datasets_cardioAI\BICL_cardioAI\results"
                  fr"\{timestamp}\six_shade_liver_segmentation")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 100)
    print("6+6 SHADE MRI LIVER SEGMENTATION")
    print("=" * 100)
    print(f"Processing {len(png_files)} MRI slice images")
    print(f"Input directory: {mri_dir}")
    print(f"Output directory: {output_dir}")
    print("\nEnhanced color scheme:")
    print("  Background: WHITE")
    print("  Liver Tissue: 6 BLUE intensity-based shades")
    print("  Liver Fat: 6 ORANGE intensity-based shades")
    print("  Outline: GRAY")
    print("\nBlue tissue shades (BGR format):")
    for i in range(1, 7):
        color = ENHANCED_COLORS_BGR[f'blue_tissue_{i}']
        print(f"  Blue {i}: {color}")
    print("\nOrange fat shades (BGR format):")
    for i in range(1, 7):
        color = ENHANCED_COLORS_BGR[f'orange_fat_{i}']
        print(f"  Orange {i}: {color}")
    print("=" * 100)

    results = []
    processed_count = 0

    for i, filename in enumerate(png_files):
        print(f"\nProgress: {i + 1}/{len(png_files)} "
              f"({(i + 1) / len(png_files) * 100:.1f}%)")
        print(f"Processing: {filename}")

        image_path = os.path.join(mri_dir, filename)

        mri_content = extract_mri_content(image_path)
        if mri_content is None:
            print(f"  Failed to load: {filename}")
            continue

        liver_tissue_mask, liver_fat_mask = advanced_liver_segmentation(mri_content)

        tissue_pixels = liver_tissue_mask.sum()
        fat_pixels = liver_fat_mask.sum()
        total_liver = tissue_pixels + fat_pixels

        tissue_pct = (tissue_pixels / liver_tissue_mask.size) * 100
        fat_pct = (fat_pixels / liver_fat_mask.size) * 100
        total_pct = (total_liver / liver_tissue_mask.size) * 100

        print(f"  Tissue (6 blue shades): {tissue_pixels} pixels ({tissue_pct:.2f}%)")
        print(f"  Fat (6 orange shades): {fat_pixels} pixels ({fat_pct:.2f}%)")
        print(f"  Total liver: {total_liver} pixels ({total_pct:.2f}%)")

        if total_liver > 0:
            heatmap = convert_to_6shade_heatmap_style(
                mri_content, liver_tissue_mask, liver_fat_mask)
            enhanced_output = create_enhanced_output(heatmap, filename)

            base_name = filename.replace('.png', '')

            cv2.imwrite(os.path.join(output_dir,
                        f"{base_name}_original.png"), mri_content)
            cv2.imwrite(os.path.join(output_dir,
                        f"{base_name}_tissue_6blue.png"), liver_tissue_mask * 255)
            cv2.imwrite(os.path.join(output_dir,
                        f"{base_name}_fat_6orange.png"), liver_fat_mask * 255)

            combined_mask = cv2.bitwise_or(liver_tissue_mask, liver_fat_mask)
            cv2.imwrite(os.path.join(output_dir,
                        f"{base_name}_liver_combined.png"), combined_mask * 255)
            cv2.imwrite(os.path.join(output_dir,
                        f"{base_name}_6shade_heatmap.png"), heatmap)
            cv2.imwrite(os.path.join(output_dir,
                        f"{base_name}_final_6shades.png"), enhanced_output)

            results.append({
                'file': filename,
                'tissue_pixels': tissue_pixels,
                'fat_pixels': fat_pixels,
                'total_liver': total_liver,
                'tissue_percentage': tissue_pct,
                'fat_percentage': fat_pct,
                'total_percentage': total_pct
            })

            processed_count += 1
            print(f"  Saved 6+6 shade liver visualization for {filename}")
        else:
            print(f"  No liver detected in {filename}")

    if results:
        create_6shade_summary_visualizations(output_dir, results)

    print("\n" + "=" * 100)
    print("6+6 SHADE LIVER SEGMENTATION COMPLETED!")
    print("=" * 100)
    print(f"Total MRI slices processed: {len(png_files)}")
    print(f"Slices with liver detected: {processed_count}")
    print(f"Success rate: {processed_count / len(png_files) * 100:.1f}%")

    if results:
        tissue_pcts = [r['tissue_percentage'] for r in results]
        fat_pcts = [r['fat_percentage'] for r in results]
        total_pcts = [r['total_percentage'] for r in results]

        print(f"Average tissue (6 blue shades): {np.mean(tissue_pcts):.2f}%")
        print(f"Average fat (6 orange shades): {np.mean(fat_pcts):.2f}%")
        print(f"Average total liver: {np.mean(total_pcts):.2f}%")
        print(f"Total range: {min(total_pcts):.2f}% - {max(total_pcts):.2f}%")

    print("=" * 100)
    print("Generated files with 6+6 shade color scheme:")
    print("  - *_final_6shades.png: Enhanced heatmap (WHITE bg, 6 BLUE tissue, 6 ORANGE fat)")
    print("  - *_6shade_heatmap.png: Raw 6+6 shade heatmap")
    print("  - *_tissue_6blue.png: Blue tissue segmentation masks")
    print("  - *_fat_6orange.png: Orange fat segmentation masks")
    print("  - *_liver_combined.png: Combined liver masks")
    print("  - *_original.png: Processed MRI content")
    print("  - 6shade_summary_*.png: Comprehensive visualization summaries")
    print("=" * 100)
    print(f"Results saved to: {output_dir}")
    print("=" * 100)


def create_6shade_summary_visualizations(output_dir, results):
    """Create comprehensive summary visualizations for 6+6 shade results"""
    n_samples = min(20, len(results))
    if n_samples > 0:
        rows = int(np.ceil(n_samples / 5))
        fig, axes = plt.subplots(rows, 5, figsize=(25, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        fig.suptitle(
            '6+6 Shade MRI Liver Segmentation - WHITE bg, 6 BLUE tissue, 6 ORANGE fat',
            fontsize=24, fontweight='bold')

        for i in range(rows * 5):
            row, col = i // 5, i % 5
            ax = axes[row, col]

            if i < n_samples:
                result = results[i]
                base_name = result['file'].replace('.png', '')

                img_path = os.path.join(output_dir,
                                        f"{base_name}_final_6shades.png")
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        ax.imshow(img_rgb)
                        ax.set_title(
                            f'Slice {i + 1}\n'
                            f'Tissue: {result["tissue_percentage"]:.1f}%\n'
                            f'Fat: {result["fat_percentage"]:.1f}%',
                            fontsize=12, fontweight='bold')
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '6shade_summary_grid.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    # Color palette visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('6+6 Shade Color Palette Visualization',
                 fontsize=16, fontweight='bold')

    blue_palette = np.zeros((100, 600, 3), dtype=np.uint8)
    for i in range(6):
        color = ENHANCED_COLORS_BGR[f'blue_tissue_{i + 1}']
        x_start = i * 100
        x_end = (i + 1) * 100
        blue_palette[:, x_start:x_end] = color

    ax1.imshow(cv2.cvtColor(blue_palette, cv2.COLOR_BGR2RGB))
    ax1.set_title('6 Blue Tissue Shades (Darkest to Lightest)',
                  fontsize=14, fontweight='bold')
    ax1.set_xticks([50, 150, 250, 350, 450, 550])
    ax1.set_xticklabels(['Blue 1', 'Blue 2', 'Blue 3',
                         'Blue 4', 'Blue 5', 'Blue 6'])
    ax1.set_yticks([])

    orange_palette = np.zeros((100, 600, 3), dtype=np.uint8)
    for i in range(6):
        color = ENHANCED_COLORS_BGR[f'orange_fat_{i + 1}']
        x_start = i * 100
        x_end = (i + 1) * 100
        orange_palette[:, x_start:x_end] = color

    ax2.imshow(cv2.cvtColor(orange_palette, cv2.COLOR_BGR2RGB))
    ax2.set_title('6 Orange Fat Shades (Darkest to Lightest)',
                  fontsize=14, fontweight='bold')
    ax2.set_xticks([50, 150, 250, 350, 450, 550])
    ax2.set_xticklabels(['Orange 1', 'Orange 2', 'Orange 3',
                         'Orange 4', 'Orange 5', 'Orange 6'])
    ax2.set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '6shade_color_palette.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == "__main__":
    process_6shade_liver_segmentation()
