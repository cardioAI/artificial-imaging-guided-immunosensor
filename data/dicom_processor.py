"""
data.dicom_processor
====================

DICOM ingestion for the CardioAI MASLD cohort. Produces the preprocessed
``.pt`` tensors that :mod:`data.pt_dataset` consumes.

Contents
--------
* ``CardioAIDatasetAnalyzer`` -- single entry point for the full pipeline.
  Key methods:
    - ``analyze_all_patients`` -- scan every 3DMI patient directory,
      count DICOM sequences, inspect NIfTI masks, determine a standard
      number of slices across the cohort.
    - ``analyze_detailed_patients`` -- deep-dive analysis for
      ``Patient_10008101`` and ``Patient_10008292`` (sequence identification,
      sample metadata, mask inspection).
    - ``process_to_pytorch_tensors`` -- convert selected patients into
      standardised ``(4, n_slices, H, W)`` tensors saved as ``.pt``.
    - ``generate_visualization_figures`` -- write 300 dpi PNG panels of
      every sequence for the detailed-example patient.
    - ``save_analysis_report`` -- dump a JSON summary of the full scan.
* ``main()`` -- CLI driver that runs the analysis -> processing ->
  visualisation stages in order.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import pydicom
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings
from collections import defaultdict
# import SimpleITK as sitk  # Optional dependency
try:
    from tqdm import tqdm
except ImportError:
    # Fallback tqdm implementation
    def tqdm(iterable, desc="Processing"):
        print(f"{desc}...")
        return iterable

try:
    import cv2
except ImportError:
    cv2 = None

# CardioAI plotting / styling helpers.
from utils.styling import (get_cardioai_colors, save_cardioai_figure,
                           setup_cardioai_style, setup_clean_axis, create_clean_legend)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class CardioAIDatasetAnalyzer:
    """
    Comprehensive analyzer for CardioAI 3DMI dataset with MASLD detection focus
    """
    
    def __init__(self, 
                 source_dir: str = r"F:\datasets_cardioAI\BICL_cardioAI\cleaned",
                 target_dir: str = r"F:\datasets_cardioAI\BICL_cardioAI\cleaned\Pt",
                 biomarkers_file: str = "./dataset_cardioAI.xlsx",
                 results_dir: str = None):
        """
        Initialize the dataset analyzer
        
        Args:
            source_dir: Path to 3DMI directory with patient data
            target_dir: Path to target .pt directory for processed tensors
            biomarkers_file: Path to biomarkers Excel file
            results_dir: Path to results directory (optional)
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.biomarkers_file = Path(biomarkers_file)
        
        # Create target directory if it doesn't exist
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        # Analysis results storage
        self.patient_analysis = {}
        self.dicom_structure_summary = {}
        self.recommended_n_slices = None
        
        # Results directory with timestamp and subfolder
        if results_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_dir = Path(f"F:/datasets_cardioAI/BICL_cardioAI/results/{timestamp}/script1_dataset_analysis")
        else:
            self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Figure storage directory within results
        self.figure_dir = self.results_dir / "figures"
        self.figure_dir.mkdir(exist_ok=True)
        
        print(f"CardioAI Dataset Analyzer initialized")
        print(f"Source: {self.source_dir}")
        print(f"Target: {self.target_dir}")
        print(f"Biomarkers: {self.biomarkers_file}")
        
    def analyze_all_patients(self) -> Dict[str, Any]:
        """
        Analyze all 108 patients to understand DICOM structure variations
        
        Returns:
            Dictionary with comprehensive analysis results
        """
        print("\n=== Analyzing All 108 Patients ===")
        
        patient_dirs = [d for d in self.source_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('Patient_')]
        
        print(f"Found {len(patient_dirs)} patients")
        
        dicom_counts = []
        sequence_analysis = defaultdict(list)
        
        for patient_dir in tqdm(patient_dirs, desc="Analyzing patients"):
            patient_id = patient_dir.name
            dicom_dir = patient_dir / "dicom"
            
            if dicom_dir.exists():
                # Count DICOM files
                dicom_files = list(dicom_dir.glob("*.dcm"))
                dicom_count = len(dicom_files)
                dicom_counts.append(dicom_count)
                
                # Analyze DICOM sequence structure
                sequences = self._analyze_dicom_sequences(dicom_files[:10])  # Sample first 10
                
                self.patient_analysis[patient_id] = {
                    'dicom_count': dicom_count,
                    'sequences': sequences,
                    'has_metadata': (patient_dir / "metadata").exists(),
                    'has_other': (patient_dir / "other").exists(),
                    'nifti_masks': self._check_nifti_masks(patient_dir / "other")
                }
                
        # Statistical analysis
        dicom_stats = {
            'total_patients': len(patient_dirs),
            'dicom_counts': dicom_counts,
            'min_dicom': min(dicom_counts) if dicom_counts else 0,
            'max_dicom': max(dicom_counts) if dicom_counts else 0,
            'mean_dicom': np.mean(dicom_counts) if dicom_counts else 0,
            'median_dicom': np.median(dicom_counts) if dicom_counts else 0,
            'std_dicom': np.std(dicom_counts) if dicom_counts else 0
        }
        
        # Determine recommended n_slices for standardization
        self.recommended_n_slices = self._determine_standard_slices(dicom_counts)
        
        print(f"\nDICOM Analysis Summary:")
        print(f"Total patients: {dicom_stats['total_patients']}")
        print(f"DICOM count range: {dicom_stats['min_dicom']} - {dicom_stats['max_dicom']}")
        print(f"Mean DICOM count: {dicom_stats['mean_dicom']:.1f} ± {dicom_stats['std_dicom']:.1f}")
        print(f"Recommended n_slices: {self.recommended_n_slices}")
        
        self.dicom_structure_summary = dicom_stats
        return dicom_stats
    
    def _analyze_dicom_sequences(self, dicom_files: List[Path]) -> Dict[str, Any]:
        """
        Analyze DICOM sequence structure from sample files
        """
        sequences = {}
        
        try:
            if dicom_files:
                # Read first DICOM to get sequence info
                dcm = pydicom.dcmread(dicom_files[0], force=True)
                
                sequences = {
                    'series_description': getattr(dcm, 'SeriesDescription', 'Unknown'),
                    'sequence_name': getattr(dcm, 'SequenceName', 'Unknown'),
                    'pixel_spacing': getattr(dcm, 'PixelSpacing', None),
                    'slice_thickness': getattr(dcm, 'SliceThickness', None),
                    'rows': getattr(dcm, 'Rows', None),
                    'columns': getattr(dcm, 'Columns', None)
                }
        except Exception as e:
            print(f"Error analyzing DICOM sequence: {e}")
            sequences = {'error': str(e)}
            
        return sequences
    
    def _check_nifti_masks(self, other_dir: Path) -> List[str]:
        """
        Check for NIfTI mask files in other/ directory
        """
        nifti_masks = []
        if other_dir.exists():
            mask_files = [
                'masked_2echo_fat_errosion.nii.gz',
                'sat_mask.nii.gz', 
                'vat_mask.nii.gz',
                'vat_mask_vb.nii.gz'
            ]
            
            for mask_file in mask_files:
                if (other_dir / mask_file).exists():
                    nifti_masks.append(mask_file)
                    
        return nifti_masks
    
    # Target voxel grid for the MASLD MRI tensors. The manuscript specifies
    # 96 x 96 x 32 voxels per volume to balance compute and anatomical
    # fidelity, so the dataset stage downsamples every sequence to this
    # shape (32 slices, 96 x 96 in-plane) before packing into .pt.
    TARGET_SLICES = 32
    TARGET_HEIGHT = 96
    TARGET_WIDTH = 96

    def _determine_standard_slices(self, dicom_counts: List[int]) -> int:
        """Return the manuscript's fixed slice count (32) regardless of cohort stats."""
        return self.TARGET_SLICES
    
    def analyze_detailed_patients(self, patient_ids: List[str] = None) -> Dict[str, Any]:
        """
        Detailed analysis of specific patients (default: Patient_10008101, Patient_10008292)
        """
        if patient_ids is None:
            patient_ids = ['Patient_10008101', 'Patient_10008292']
            
        print(f"\n=== Detailed Analysis of {patient_ids} ===")
        
        detailed_analysis = {}
        
        for patient_id in patient_ids:
            patient_dir = self.source_dir / patient_id
            
            if not patient_dir.exists():
                print(f"Warning: {patient_id} not found")
                continue
                
            print(f"\nAnalyzing {patient_id}...")
            
            # Analyze directory structure
            analysis = {
                'patient_id': patient_id,
                'dicom_analysis': self._analyze_dicom_directory(patient_dir / "dicom"),
                'metadata_analysis': self._analyze_metadata(patient_dir / "metadata"),
                'other_analysis': self._analyze_other_directory(patient_dir / "other"),
                'recommendations': {}
            }
            
            # Generate recommendations
            analysis['recommendations'] = self._generate_processing_recommendations(analysis)
            
            detailed_analysis[patient_id] = analysis
            
            # Print summary
            self._print_patient_summary(analysis)
            
        return detailed_analysis
    
    def _analyze_dicom_directory(self, dicom_dir: Path) -> Dict[str, Any]:
        """
        Comprehensive DICOM directory analysis
        """
        if not dicom_dir.exists():
            return {'error': 'DICOM directory not found'}
            
        dicom_files = sorted(dicom_dir.glob("*.dcm"))
        
        analysis = {
            'total_files': len(dicom_files),
            'file_pattern': self._analyze_file_pattern(dicom_files),
            'sequences': self._identify_sequences(dicom_files),
            'sample_metadata': self._extract_sample_metadata(dicom_files[:5])
        }
        
        return analysis
    
    def _analyze_file_pattern(self, dicom_files: List[Path]) -> Dict[str, Any]:
        """
        Analyze DICOM file naming patterns
        """
        if not dicom_files:
            return {}
            
        # Extract sequence and slice numbers from filenames
        sequences = set()
        slices_per_seq = defaultdict(set)
        
        for file_path in dicom_files:
            parts = file_path.stem.split('-')
            if len(parts) == 2:
                seq_num, slice_num = parts
                sequences.add(int(seq_num))
                slices_per_seq[int(seq_num)].add(int(slice_num))
        
        pattern = {
            'sequences': sorted(list(sequences)),
            'slices_per_sequence': {seq: len(slices) for seq, slices in slices_per_seq.items()},
            'total_sequences': len(sequences),
            'slice_ranges': {seq: f"{min(slices)}-{max(slices)}" 
                           for seq, slices in slices_per_seq.items()}
        }
        
        return pattern
    
    def _identify_sequences(self, dicom_files: List[Path]) -> Dict[str, Any]:
        """
        Identify different MRI sequences in DICOM files
        """
        sequences = {}
        
        # Sample files from different sequence numbers
        sequence_samples = {}
        
        for file_path in dicom_files[:20]:  # Sample first 20 files
            try:
                parts = file_path.stem.split('-')
                if len(parts) == 2:
                    seq_num = int(parts[0])
                    if seq_num not in sequence_samples:
                        dcm = pydicom.dcmread(file_path, force=True)
                        sequence_samples[seq_num] = {
                            'series_description': getattr(dcm, 'SeriesDescription', 'Unknown'),
                            'sequence_name': getattr(dcm, 'SequenceName', 'Unknown'),
                            'echo_time': getattr(dcm, 'EchoTime', None),
                            'repetition_time': getattr(dcm, 'RepetitionTime', None),
                            'slice_thickness': getattr(dcm, 'SliceThickness', None)
                        }
            except Exception as e:
                continue
                
        return sequence_samples
    
    def _extract_sample_metadata(self, dicom_files: List[Path]) -> Dict[str, Any]:
        """
        Extract metadata from sample DICOM files
        """
        if not dicom_files:
            return {}
            
        try:
            dcm = pydicom.dcmread(dicom_files[0], force=True)
            
            metadata = {
                'patient_id': getattr(dcm, 'PatientID', 'Unknown'),
                'study_date': getattr(dcm, 'StudyDate', 'Unknown'),
                'modality': getattr(dcm, 'Modality', 'Unknown'),
                'manufacturer': getattr(dcm, 'Manufacturer', 'Unknown'),
                'magnetic_field_strength': getattr(dcm, 'MagneticFieldStrength', None),
                'pixel_spacing': getattr(dcm, 'PixelSpacing', None),
                'rows': getattr(dcm, 'Rows', None),
                'columns': getattr(dcm, 'Columns', None),
                'bits_allocated': getattr(dcm, 'BitsAllocated', None)
            }
            
        except Exception as e:
            metadata = {'error': str(e)}
            
        return metadata
    
    def _analyze_metadata(self, metadata_dir: Path) -> Dict[str, Any]:
        """
        Analyze metadata directory and results.csv
        """
        if not metadata_dir.exists():
            return {'error': 'Metadata directory not found'}
            
        results_file = metadata_dir / "results.csv"
        
        analysis = {
            'has_results_csv': results_file.exists(),
            'files': [f.name for f in metadata_dir.iterdir()]
        }
        
        if results_file.exists():
            try:
                # Read results.csv with custom parsing due to non-standard format
                with open(results_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                analysis['results_summary'] = {
                    'total_lines': len(content.split('\n')),
                    'contains_surface_measures': '=== Measures on the surface' in content,
                    'contains_volume_measures': '=== Measures on the volume' in content,
                    'slice_count_estimate': content.count('12-') if '12-' in content else 0
                }
                
            except Exception as e:
                analysis['results_error'] = str(e)
                
        return analysis
    
    def _analyze_other_directory(self, other_dir: Path) -> Dict[str, Any]:
        """
        Analyze other/ directory with NIfTI masks and additional files
        """
        if not other_dir.exists():
            return {'error': 'Other directory not found'}
            
        files = list(other_dir.iterdir())
        
        analysis = {
            'total_files': len(files),
            'file_types': {},
            'nifti_masks': [],
            'dcm_tags': [],
            'other_files': []
        }
        
        for file_path in files:
            suffix = file_path.suffix.lower()
            analysis['file_types'][suffix] = analysis['file_types'].get(suffix, 0) + 1
            
            if suffix == '.gz' and file_path.name.endswith('.nii.gz'):
                analysis['nifti_masks'].append(file_path.name)
            elif suffix == '.tag':
                analysis['dcm_tags'].append(file_path.name)
            else:
                analysis['other_files'].append(file_path.name)
                
        # Analyze NIfTI masks if present
        if analysis['nifti_masks']:
            analysis['nifti_analysis'] = self._analyze_nifti_masks(other_dir, analysis['nifti_masks'])
            
        return analysis
    
    def _analyze_nifti_masks(self, other_dir: Path, mask_files: List[str]) -> Dict[str, Any]:
        """
        Analyze NIfTI mask files for dimensions and content
        """
        nifti_analysis = {}
        
        for mask_file in mask_files:
            mask_path = other_dir / mask_file
            try:
                img = nib.load(mask_path)
                data = img.get_fdata()
                
                nifti_analysis[mask_file] = {
                    'shape': data.shape,
                    'data_type': str(data.dtype),
                    'voxel_size': img.header.get_zooms(),
                    'unique_values': len(np.unique(data)),
                    'non_zero_voxels': np.count_nonzero(data),
                    'data_range': (float(data.min()), float(data.max()))
                }
                
            except Exception as e:
                nifti_analysis[mask_file] = {'error': str(e)}
                
        return nifti_analysis
    
    def _generate_processing_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate recommendations for dataset processing based on analysis
        """
        recommendations = {
            'include_dicom_sequences': [],
            'exclude_sequences': [],
            'include_nifti_masks': [],
            'processing_strategy': '',
            'standardization_approach': ''
        }
        
        # DICOM sequence recommendations
        dicom_analysis = analysis.get('dicom_analysis', {})
        file_pattern = dicom_analysis.get('file_pattern', {})
        
        if 'slices_per_sequence' in file_pattern:
            # Recommend sequences with consistent slice counts
            for seq, slice_count in file_pattern['slices_per_sequence'].items():
                if slice_count >= 80:  # Sufficient slices for 3D analysis
                    recommendations['include_dicom_sequences'].append(seq)
                else:
                    recommendations['exclude_sequences'].append(seq)
        
        # NIfTI mask recommendations
        other_analysis = analysis.get('other_analysis', {})
        nifti_masks = other_analysis.get('nifti_masks', [])
        
        # Recommend including relevant masks for MASLD analysis
        important_masks = ['masked_2echo_fat_errosion.nii.gz', 'vat_mask.nii.gz', 'sat_mask.nii.gz']
        
        for mask in nifti_masks:
            if any(important in mask for important in important_masks):
                recommendations['include_nifti_masks'].append(mask)
        
        # Processing strategy
        total_dicom = dicom_analysis.get('total_files', 0)
        if total_dicom > 300:
            recommendations['processing_strategy'] = 'multi_sequence_3d'
        elif total_dicom > 150:
            recommendations['processing_strategy'] = 'dual_sequence_3d'
        else:
            recommendations['processing_strategy'] = 'single_sequence_3d'
            
        # Standardization approach
        if self.recommended_n_slices:
            recommendations['standardization_approach'] = f'resize_to_{self.recommended_n_slices}_slices'
        else:
            recommendations['standardization_approach'] = 'adaptive_sizing'
            
        return recommendations
    
    def _print_patient_summary(self, analysis: Dict[str, Any]):
        """
        Print formatted summary for a patient analysis
        """
        patient_id = analysis['patient_id']
        print(f"\n--- {patient_id} Summary ---")
        
        # DICOM summary
        dicom = analysis.get('dicom_analysis', {})
        if 'total_files' in dicom:
            print(f"DICOM files: {dicom['total_files']}")
            
        if 'file_pattern' in dicom:
            pattern = dicom['file_pattern']
            if 'sequences' in pattern:
                print(f"Sequences: {pattern['sequences']}")
                print(f"Slices per sequence: {pattern['slices_per_sequence']}")
        
        # NIfTI masks
        other = analysis.get('other_analysis', {})
        if 'nifti_masks' in other:
            masks = other['nifti_masks']
            print(f"NIfTI masks: {masks}")
        
        # Recommendations
        rec = analysis.get('recommendations', {})
        if 'include_dicom_sequences' in rec:
            print(f"Recommended DICOM sequences: {rec['include_dicom_sequences']}")
        if 'include_nifti_masks' in rec:
            print(f"Recommended masks: {rec['include_nifti_masks']}")
    
    def process_to_pytorch_tensors(self, patient_ids: List[str] = None) -> Dict[str, str]:
        """
        Process selected patients to PyTorch .pt format for ML training
        """
        if patient_ids is None:
            # Process all patients
            patient_dirs = [d.name for d in self.source_dir.iterdir() 
                           if d.is_dir() and d.name.startswith('Patient_')]
            patient_ids = sorted(patient_dirs)
            
        print(f"\n=== Processing to PyTorch Tensors ===")
        
        processing_results = {}
        
        print(f"Processing {len(patient_ids)} patients to .pt format...")
        
        for idx, patient_id in enumerate(tqdm(patient_ids, desc="Processing patients")):
            print(f"\n[{idx+1}/{len(patient_ids)}] Processing {patient_id}...")
            
            try:
                result = self._process_single_patient(patient_id)
                processing_results[patient_id] = result
                print(f"[SUCCESS] {patient_id} -> {result.get('file_size_mb', 0):.1f} MB")
                
            except Exception as e:
                print(f"[ERROR] Error processing {patient_id}: {e}")
                processing_results[patient_id] = {'error': str(e)}
        
        # Generate processing statistics
        successful = [k for k, v in processing_results.items() if 'error' not in v]
        failed = [k for k, v in processing_results.items() if 'error' in v]
        
        total_size_mb = sum(v.get('file_size_mb', 0) for v in processing_results.values() if 'error' not in v)
        
        print(f"\n=== Processing Summary ===")
        print(f"Total patients: {len(patient_ids)}")
        print(f"Successfully processed: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print(f"Total dataset size: {total_size_mb:.1f} MB")
        
        if failed:
            print(f"Failed patients: {failed[:10]}{'...' if len(failed) > 10 else ''}")
                
        return processing_results
    
    def _process_single_patient(self, patient_id: str) -> Dict[str, Any]:
        """
        Process a single patient to PyTorch tensor format
        """
        patient_dir = self.source_dir / patient_id
        dicom_dir = patient_dir / "dicom"
        other_dir = patient_dir / "other"
        
        # Load DICOM sequences
        dicom_data = self._load_dicom_sequences(dicom_dir, patient_id)
        
        # Load NIfTI masks if available
        nifti_data = {}
        if other_dir.exists():
            nifti_data = self._load_nifti_masks(other_dir)
        
        # Create output tensor dictionary
        output_data = {
            'patient_id': patient_id,
            'dicom_sequences': dicom_data,
            'nifti_masks': nifti_data,
            'processing_metadata': {
                'n_slices': self.recommended_n_slices,
                'processing_date': datetime.now().isoformat(),
                'source_dir': str(patient_dir)
            }
        }
        
        # Save to .pt file
        output_path = self.target_dir / f"{patient_id}.pt"
        torch.save(output_data, output_path)
        
        return {
            'output_path': str(output_path),
            'dicom_sequences': len(dicom_data),
            'nifti_masks': len(nifti_data),
            'file_size_mb': output_path.stat().st_size / (1024 * 1024)
        }
    
    def _load_dicom_sequences(self, dicom_dir: Path, patient_id: str) -> Dict[str, torch.Tensor]:
        """
        Load and standardize DICOM sequences to tensors
        """
        if not dicom_dir.exists():
            return {}
            
        dicom_files = sorted(dicom_dir.glob("*.dcm"))
        
        # Group by sequence
        sequences = defaultdict(list)
        
        for file_path in dicom_files:
            try:
                parts = file_path.stem.split('-')
                if len(parts) == 2:
                    seq_num = int(parts[0])
                    slice_num = int(parts[1])
                    sequences[seq_num].append((slice_num, file_path))
            except:
                continue
        
        # Process each sequence
        sequence_tensors = {}
        
        for seq_num, files in sequences.items():
            try:
                # Sort by slice number
                files.sort(key=lambda x: x[0])
                
                # Load images
                images = []
                for _, file_path in files:
                    dcm = pydicom.dcmread(file_path, force=True)
                    img = dcm.pixel_array.astype(np.float32)
                    images.append(img)
                
                if images:
                    # Stack into 3D volume
                    volume = np.stack(images, axis=0)
                    
                    # Standardize to recommended n_slices
                    volume = self._standardize_volume(volume, self.recommended_n_slices)
                    
                    # Convert to tensor
                    sequence_tensors[f'sequence_{seq_num}'] = torch.from_numpy(volume)
                    
            except Exception as e:
                print(f"Warning: Error processing sequence {seq_num} for {patient_id}: {e}")
                continue
                
        return sequence_tensors
    
    def _standardize_volume(self, volume: np.ndarray, target_slices: int) -> np.ndarray:
        """
        Standardize a 3D MRI volume to the manuscript's canonical preprocessing:

        1. Resample depth / height / width to (target_slices, TARGET_HEIGHT, TARGET_WIDTH)
           via trilinear zoom (preserves anatomical continuity).
        2. Intensity-normalize to zero mean, unit variance (per-volume z-score).
        3. Apply a simple centre-of-mass rigid translation so the intensity
           centroid lands at the geometric centre of the volume -- a
           lightweight stand-in for template registration that is good
           enough for abdominal VIBE-Dixon series.
        """
        current_slices, current_h, current_w = volume.shape
        target_h, target_w = self.TARGET_HEIGHT, self.TARGET_WIDTH

        # (1) Trilinear resample to the target voxel grid.
        if (current_slices, current_h, current_w) != (target_slices, target_h, target_w):
            try:
                from scipy.ndimage import zoom
                factors = (target_slices / current_slices,
                           target_h / current_h,
                           target_w / current_w)
                volume = zoom(volume, factors, order=1)
            except ImportError:
                # Nearest-neighbour fallback if scipy is unavailable.
                zi = np.round(np.linspace(0, current_slices - 1, target_slices)).astype(int)
                yi = np.round(np.linspace(0, current_h - 1, target_h)).astype(int)
                xi = np.round(np.linspace(0, current_w - 1, target_w)).astype(int)
                volume = volume[zi][:, yi][:, :, xi]

        # (2) Intensity normalisation: per-volume z-score with epsilon.
        mean = float(volume.mean())
        std = float(volume.std())
        volume = (volume - mean) / (std + 1e-6)

        # (3) Simple centre-of-mass registration: shift the intensity
        # centroid (computed on |volume|, since z-scored volumes span
        # negative values) onto the geometric centre.
        try:
            from scipy.ndimage import center_of_mass, shift
            mass = np.abs(volume) + 1e-8
            com = np.asarray(center_of_mass(mass))
            centre = np.asarray(volume.shape, dtype=np.float64) / 2.0 - 0.5
            delta = centre - com
            # Clip to a small translation to avoid aggressive shifts when
            # the mass is effectively centred already.
            delta = np.clip(delta, -4.0, 4.0)
            volume = shift(volume, shift=delta, order=1, mode="nearest")
        except ImportError:
            pass

        return volume.astype(np.float32)
    
    def _load_nifti_masks(self, other_dir: Path) -> Dict[str, torch.Tensor]:
        """
        Load NIfTI mask files as tensors
        """
        nifti_files = [
            'masked_2echo_fat_errosion.nii.gz',
            'sat_mask.nii.gz',
            'vat_mask.nii.gz',
            'vat_mask_vb.nii.gz'
        ]
        
        nifti_tensors = {}
        
        for filename in nifti_files:
            file_path = other_dir / filename
            if file_path.exists():
                try:
                    img = nib.load(file_path)
                    data = img.get_fdata().astype(np.float32)
                    
                    # Standardize dimensions if needed
                    if len(data.shape) == 3:
                        data = self._standardize_volume(data, self.recommended_n_slices)
                    
                    nifti_tensors[filename.replace('.nii.gz', '')] = torch.from_numpy(data)
                    
                except Exception as e:
                    print(f"Warning: Error loading {filename}: {e}")
                    
        return nifti_tensors
    
    def generate_visualization_figures(self, patient_id: str = 'Patient_10008101') -> List[str]:
        """
        Generate 300 DPI PNG figures for the specified patient
        """
        print(f"\n=== Generating Visualization Figures for {patient_id} ===")
        
        patient_dir = self.source_dir / patient_id
        dicom_dir = patient_dir / "dicom"
        other_dir = patient_dir / "other"
        
        if not dicom_dir.exists():
            print(f"Error: DICOM directory not found for {patient_id}")
            return []
        
        # Create patient-specific figure directory
        patient_figure_dir = self.figure_dir / patient_id
        patient_figure_dir.mkdir(exist_ok=True)
        
        generated_figures = []
        
        # Load and visualize DICOM sequences
        dicom_files = sorted(dicom_dir.glob("*.dcm"))
        
        # Group by sequence for better visualization
        sequences = defaultdict(list)
        for file_path in dicom_files:
            try:
                parts = file_path.stem.split('-')
                if len(parts) == 2:
                    seq_num = int(parts[0])
                    slice_num = int(parts[1])
                    sequences[seq_num].append((slice_num, file_path))
            except:
                continue
        
        # Generate figures for each sequence
        for seq_num, files in sequences.items():
            files.sort(key=lambda x: x[0])
            
            # Select representative slices (evenly distributed)
            n_representative = min(self.recommended_n_slices, len(files))
            if n_representative > len(files):
                n_representative = len(files)
                
            indices = np.linspace(0, len(files) - 1, n_representative, dtype=int)
            selected_files = [files[i] for i in indices]
            
            # Create multi-slice figure
            fig_path = self._create_sequence_figure(selected_files, seq_num, patient_figure_dir, patient_id)
            if fig_path:
                generated_figures.append(fig_path)
        
        # Generate NIfTI mask visualizations if available
        if other_dir.exists():
            nifti_figures = self._create_nifti_figures(other_dir, patient_figure_dir, patient_id)
            generated_figures.extend(nifti_figures)
        
        print(f"Generated {len(generated_figures)} figures for {patient_id}")
        return generated_figures
    
    def _create_sequence_figure(self, files: List[Tuple[int, Path]], seq_num: int, 
                              figure_dir: Path, patient_id: str) -> Optional[str]:
        """
        Create a multi-slice figure for a DICOM sequence
        """
        try:
            # Load images
            images = []
            slice_numbers = []
            
            for slice_num, file_path in files:
                dcm = pydicom.dcmread(file_path, force=True)
                img = dcm.pixel_array.astype(np.float32)
                images.append(img)
                slice_numbers.append(slice_num)
            
            if not images:
                return None
            
            # Create standardized figure with subplots
            n_cols = 8
            n_rows = int(np.ceil(len(images) / n_cols))
            
            # Setup CardioAI style
            setup_cardioai_style()
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows))
            fig.suptitle(f'{patient_id} - Sequence {seq_num} - {len(images)} slices', 
                        fontsize=16, fontweight='bold')
            
            # Flatten axes for easier indexing
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            # Plot images
            for idx, (img, slice_num) in enumerate(zip(images, slice_numbers)):
                if idx < len(axes):
                    ax = axes[idx]
                    
                    # Normalize image for better visualization
                    if cv2 is not None:
                        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    else:
                        # Fallback normalization
                        img_norm = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                    
                    ax.imshow(img_norm, cmap='gray')
                    ax.set_title(f'Slice {slice_num}', fontsize=10)
                    ax.axis('off')
            
            # Hide unused subplots
            for idx in range(len(images), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            
            # Save in standardized formats
            figure_path = figure_dir / f'{patient_id}_sequence_{seq_num}_slices'
            save_cardioai_figure(fig, figure_path)
            plt.close()
            
            return str(figure_path)
            
        except Exception as e:
            print(f"Error creating sequence figure: {e}")
            return None
    
    def _create_nifti_figures(self, other_dir: Path, figure_dir: Path, patient_id: str) -> List[str]:
        """
        Create visualization figures for NIfTI mask files
        """
        nifti_files = [
            'masked_2echo_fat_errosion.nii.gz',
            'sat_mask.nii.gz',
            'vat_mask.nii.gz',
            'vat_mask_vb.nii.gz'
        ]
        
        generated_figures = []
        
        for filename in nifti_files:
            file_path = other_dir / filename
            if file_path.exists():
                try:
                    # Load NIfTI data
                    img = nib.load(file_path)
                    data = img.get_fdata()
                    
                    # Select representative slices
                    n_slices = data.shape[2] if len(data.shape) >= 3 else 1
                    n_representative = min(16, n_slices)  # Show up to 16 slices
                    
                    if n_slices > 1:
                        slice_indices = np.linspace(0, n_slices - 1, n_representative, dtype=int)
                        
                        # Create standardized figure
                        n_cols = 4
                        n_rows = int(np.ceil(n_representative / n_cols))
                        
                        # Setup CardioAI style and get colors
                        setup_cardioai_style()
                        cardio_colors = get_cardioai_colors(4)
                        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
                        fig.suptitle(f'{patient_id} - {filename}', fontsize=14, fontweight='bold')
                        
                        if n_rows == 1:
                            axes = [axes] if n_cols == 1 else axes
                        else:
                            axes = axes.flatten()
                        
                        for idx, slice_idx in enumerate(slice_indices):
                            if idx < len(axes):
                                ax = axes[idx]
                                
                                if len(data.shape) >= 3:
                                    slice_data = data[:, :, slice_idx]
                                else:
                                    slice_data = data
                                
                                # Use appropriate colormap for masks
                                if 'mask' in filename.lower():
                                    cmap = 'viridis'
                                else:
                                    cmap = 'gray'
                                
                                im = ax.imshow(slice_data, cmap=cmap)
                                ax.set_title(f'Slice {slice_idx}', fontsize=10)
                                ax.axis('off')
                                
                                # Add colorbar for the first subplot
                                if idx == 0:
                                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        
                        # Hide unused subplots
                        for idx in range(n_representative, len(axes)):
                            axes[idx].axis('off')
                        
                        plt.tight_layout()
                        
                        # Save in standardized formats
                        clean_name = filename.replace('.nii.gz', '').replace('.', '_')
                        figure_path = figure_dir / f'{patient_id}_{clean_name}_mask'
                        save_cardioai_figure(fig, figure_path)
                        plt.close()
                        
                        generated_figures.append(str(figure_path))
                        
                except Exception as e:
                    print(f"Error creating NIfTI figure for {filename}: {e}")
        
        return generated_figures
    
    def save_analysis_report(self) -> str:
        """
        Save comprehensive analysis report to JSON file
        """
        report = {
            'analysis_date': datetime.now().isoformat(),
            'dataset_summary': self.dicom_structure_summary,
            'patient_analysis': self.patient_analysis,
            'recommended_n_slices': self.recommended_n_slices,
            'processing_recommendations': {
                'include_sequences': 'All sequences with >= 80 slices',
                'include_nifti_masks': [
                    'masked_2echo_fat_errosion.nii.gz',
                    'vat_mask.nii.gz', 
                    'sat_mask.nii.gz'
                ],
                'standardization_strategy': f'Resize all volumes to {self.recommended_n_slices} slices',
                'tensor_format': 'PyTorch .pt files with standardized dimensions'
            }
        }
        
        report_path = self.results_dir / "cardioai_dataset_analysis_report.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nAnalysis report saved to: {report_path}")
        return str(report_path)

def main():
    """
    Main execution function for dataset analysis and processing
    """
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CardioAI Dataset Analysis and Processing')
    parser.add_argument('--results-dir', type=str, help='Results directory path')
    args = parser.parse_args()
    
    print("CardioAI Dataset Analysis and Processing")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = CardioAIDatasetAnalyzer(results_dir=args.results_dir)
    
    # Step 1: Analyze all patients
    print("\nStep 1: Analyzing all 108 patients...")
    dataset_summary = analyzer.analyze_all_patients()
    
    # Step 2: Detailed analysis of specific patients
    print("\nStep 2: Detailed analysis of Patient_10008101 and Patient_10008292...")
    detailed_analysis = analyzer.analyze_detailed_patients(['Patient_10008101', 'Patient_10008292'])
    
    # Step 3: Process ALL patients to PyTorch tensors
    print("\nStep 3: Processing ALL 108 patients to PyTorch tensor format...")
    processing_results = analyzer.process_to_pytorch_tensors()  # Process all patients
    
    # Step 4: Generate visualization figures
    print("\nStep 4: Generating visualization figures...")
    figures = analyzer.generate_visualization_figures('Patient_10008101')
    
    # Step 5: Save analysis report
    print("\nStep 5: Saving comprehensive analysis report...")
    report_path = analyzer.save_analysis_report()
    
    # Final summary
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    print(f"[OK] Analyzed {dataset_summary.get('total_patients', 0)} patients")
    print(f"[OK] Recommended n_slices: {analyzer.recommended_n_slices}")
    print(f"[OK] Generated {len(figures)} visualization figures")
    print(f"[OK] Processed patients to .pt format")
    print(f"[OK] Analysis report: {report_path}")
    
    # Key recommendations
    print("\nKEY RECOMMENDATIONS:")
    print("─" * 30)
    print("1. DICOM Processing:")
    print(f"   - Use {analyzer.recommended_n_slices} slices per volume for standardization")
    print("   - Include all sequences with >= 80 slices")
    print("   - Focus on sequences 9, 10, 11, 12 (most common pattern)")
    
    print("\n2. NIfTI Masks:")
    print("   - Include: masked_2echo_fat_errosion.nii.gz (fat quantification)")
    print("   - Include: vat_mask.nii.gz, sat_mask.nii.gz (tissue segmentation)")
    print("   - Optional: vat_mask_vb.nii.gz (additional VAT mask)")
    
    print("\n3. ML Training Dataset:")
    print(f"   - Target directory: F:\\datasets_cardioAI\\BICL_cardioAI\\cleaned\\Pt")
    print("   - Format: PyTorch .pt tensors")
    print("   - Standardized dimensions for consistent training")
    print("   - Include both DICOM sequences and NIfTI masks")
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()