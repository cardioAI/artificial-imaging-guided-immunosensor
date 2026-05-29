
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

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, desc="Processing"):
        print(f"{desc}...")
        return iterable

try:
    import cv2
except ImportError:
    cv2 = None

from utils.styling import (get_cardioai_colors, save_cardioai_figure,
                           setup_cardioai_style, setup_clean_axis, create_clean_legend)

warnings.filterwarnings('ignore')

class CardioAIDatasetAnalyzer:
    
    def __init__(self, 
                 source_dir: str = r"F:\datasets_cardioAI\BICL_cardioAI\cleaned",
                 target_dir: str = r"F:\datasets_cardioAI\BICL_cardioAI\cleaned\Pt",
                 biomarkers_file: str = "/data/biomarkers.xlsx",
                 results_dir: str = None):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.biomarkers_file = Path(biomarkers_file)
        

        self.target_dir.mkdir(parents=True, exist_ok=True)
        

        self.patient_analysis = {}
        self.dicom_structure_summary = {}
        self.recommended_n_slices = None
        

        if results_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_dir = Path(f"./results/{timestamp}/script1_dataset_analysis")
        else:
            self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        

        self.figure_dir = self.results_dir / "figures"
        self.figure_dir.mkdir(exist_ok=True)
        
        print(f"CardioAI Dataset Analyzer initialized")
        print(f"Source: {self.source_dir}")
        print(f"Target: {self.target_dir}")
        print(f"Biomarkers: {self.biomarkers_file}")
        
    def analyze_all_patients(self) -> Dict[str, Any]:
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

                dicom_files = list(dicom_dir.glob("*.dcm"))
                dicom_count = len(dicom_files)
                dicom_counts.append(dicom_count)
                

                sequences = self._analyze_dicom_sequences(dicom_files[:10])
                
                self.patient_analysis[patient_id] = {
                    'dicom_count': dicom_count,
                    'sequences': sequences,
                    'has_metadata': (patient_dir / "metadata").exists(),
                    'has_other': (patient_dir / "other").exists(),
                    'nifti_masks': self._check_nifti_masks(patient_dir / "other")
                }
                

        dicom_stats = {
            'total_patients': len(patient_dirs),
            'dicom_counts': dicom_counts,
            'min_dicom': min(dicom_counts) if dicom_counts else 0,
            'max_dicom': max(dicom_counts) if dicom_counts else 0,
            'mean_dicom': np.mean(dicom_counts) if dicom_counts else 0,
            'median_dicom': np.median(dicom_counts) if dicom_counts else 0,
            'std_dicom': np.std(dicom_counts) if dicom_counts else 0
        }
        

        self.recommended_n_slices = self._determine_standard_slices(dicom_counts)
        
        print(f"\nDICOM Analysis Summary:")
        print(f"Total patients: {dicom_stats['total_patients']}")
        print(f"DICOM count range: {dicom_stats['min_dicom']} - {dicom_stats['max_dicom']}")
        print(f"Mean DICOM count: {dicom_stats['mean_dicom']:.1f} ± {dicom_stats['std_dicom']:.1f}")
        print(f"Recommended n_slices: {self.recommended_n_slices}")
        
        self.dicom_structure_summary = dicom_stats
        return dicom_stats
    
    def _analyze_dicom_sequences(self, dicom_files: List[Path]) -> Dict[str, Any]:
        sequences = {}
        
        try:
            if dicom_files:

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
    

    TARGET_SLICES = 32
    TARGET_HEIGHT = 96
    TARGET_WIDTH = 96

    def _determine_standard_slices(self, dicom_counts: List[int]) -> int:
        return self.TARGET_SLICES
    
    def analyze_detailed_patients(self, patient_ids: List[str] = None) -> Dict[str, Any]:
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
            

            analysis = {
                'patient_id': patient_id,
                'dicom_analysis': self._analyze_dicom_directory(patient_dir / "dicom"),
                'metadata_analysis': self._analyze_metadata(patient_dir / "metadata"),
                'other_analysis': self._analyze_other_directory(patient_dir / "other"),
                'recommendations': {}
            }
            

            analysis['recommendations'] = self._generate_processing_recommendations(analysis)
            
            detailed_analysis[patient_id] = analysis
            

            self._print_patient_summary(analysis)
            
        return detailed_analysis
    
    def _analyze_dicom_directory(self, dicom_dir: Path) -> Dict[str, Any]:
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
        if not dicom_files:
            return {}
            

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
        sequences = {}
        

        sequence_samples = {}
        
        for file_path in dicom_files[:20]:
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
        if not metadata_dir.exists():
            return {'error': 'Metadata directory not found'}
            
        results_file = metadata_dir / "results.csv"
        
        analysis = {
            'has_results_csv': results_file.exists(),
            'files': [f.name for f in metadata_dir.iterdir()]
        }
        
        if results_file.exists():
            try:

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
                

        if analysis['nifti_masks']:
            analysis['nifti_analysis'] = self._analyze_nifti_masks(other_dir, analysis['nifti_masks'])
            
        return analysis
    
    def _analyze_nifti_masks(self, other_dir: Path, mask_files: List[str]) -> Dict[str, Any]:
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
        recommendations = {
            'include_dicom_sequences': [],
            'exclude_sequences': [],
            'include_nifti_masks': [],
            'processing_strategy': '',
            'standardization_approach': ''
        }
        

        dicom_analysis = analysis.get('dicom_analysis', {})
        file_pattern = dicom_analysis.get('file_pattern', {})
        
        if 'slices_per_sequence' in file_pattern:

            for seq, slice_count in file_pattern['slices_per_sequence'].items():
                if slice_count >= 80:
                    recommendations['include_dicom_sequences'].append(seq)
                else:
                    recommendations['exclude_sequences'].append(seq)
        

        other_analysis = analysis.get('other_analysis', {})
        nifti_masks = other_analysis.get('nifti_masks', [])
        

        important_masks = ['masked_2echo_fat_errosion.nii.gz', 'vat_mask.nii.gz', 'sat_mask.nii.gz']
        
        for mask in nifti_masks:
            if any(important in mask for important in important_masks):
                recommendations['include_nifti_masks'].append(mask)
        

        total_dicom = dicom_analysis.get('total_files', 0)
        if total_dicom > 300:
            recommendations['processing_strategy'] = 'multi_sequence_3d'
        elif total_dicom > 150:
            recommendations['processing_strategy'] = 'dual_sequence_3d'
        else:
            recommendations['processing_strategy'] = 'single_sequence_3d'
            

        if self.recommended_n_slices:
            recommendations['standardization_approach'] = f'resize_to_{self.recommended_n_slices}_slices'
        else:
            recommendations['standardization_approach'] = 'adaptive_sizing'
            
        return recommendations
    
    def _print_patient_summary(self, analysis: Dict[str, Any]):
        patient_id = analysis['patient_id']
        print(f"\n--- {patient_id} Summary ---")
        

        dicom = analysis.get('dicom_analysis', {})
        if 'total_files' in dicom:
            print(f"DICOM files: {dicom['total_files']}")
            
        if 'file_pattern' in dicom:
            pattern = dicom['file_pattern']
            if 'sequences' in pattern:
                print(f"Sequences: {pattern['sequences']}")
                print(f"Slices per sequence: {pattern['slices_per_sequence']}")
        

        other = analysis.get('other_analysis', {})
        if 'nifti_masks' in other:
            masks = other['nifti_masks']
            print(f"NIfTI masks: {masks}")
        

        rec = analysis.get('recommendations', {})
        if 'include_dicom_sequences' in rec:
            print(f"Recommended DICOM sequences: {rec['include_dicom_sequences']}")
        if 'include_nifti_masks' in rec:
            print(f"Recommended masks: {rec['include_nifti_masks']}")
    
    def process_to_pytorch_tensors(self, patient_ids: List[str] = None) -> Dict[str, str]:
        if patient_ids is None:

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
        patient_dir = self.source_dir / patient_id
        dicom_dir = patient_dir / "dicom"
        other_dir = patient_dir / "other"
        

        dicom_data = self._load_dicom_sequences(dicom_dir, patient_id)
        

        nifti_data = {}
        if other_dir.exists():
            nifti_data = self._load_nifti_masks(other_dir)
        

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
        

        output_path = self.target_dir / f"{patient_id}.pt"
        torch.save(output_data, output_path)
        
        return {
            'output_path': str(output_path),
            'dicom_sequences': len(dicom_data),
            'nifti_masks': len(nifti_data),
            'file_size_mb': output_path.stat().st_size / (1024 * 1024)
        }
    
    def _load_dicom_sequences(self, dicom_dir: Path, patient_id: str) -> Dict[str, torch.Tensor]:
        if not dicom_dir.exists():
            return {}
            
        dicom_files = sorted(dicom_dir.glob("*.dcm"))
        

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
        

        sequence_tensors = {}
        
        for seq_num, files in sequences.items():
            try:

                files.sort(key=lambda x: x[0])
                

                images = []
                for _, file_path in files:
                    dcm = pydicom.dcmread(file_path, force=True)
                    img = dcm.pixel_array.astype(np.float32)
                    images.append(img)
                
                if images:

                    volume = np.stack(images, axis=0)
                    

                    volume = self._standardize_volume(volume, self.recommended_n_slices)
                    

                    sequence_tensors[f'sequence_{seq_num}'] = torch.from_numpy(volume)
                    
            except Exception as e:
                print(f"Warning: Error processing sequence {seq_num} for {patient_id}: {e}")
                continue
                
        return sequence_tensors
    
    def _standardize_volume(self, volume: np.ndarray, target_slices: int) -> np.ndarray:
        current_slices, current_h, current_w = volume.shape
        target_h, target_w = self.TARGET_HEIGHT, self.TARGET_WIDTH

        if (current_slices, current_h, current_w) != (target_slices, target_h, target_w):
            try:
                from scipy.ndimage import zoom
                factors = (target_slices / current_slices,
                           target_h / current_h,
                           target_w / current_w)
                volume = zoom(volume, factors, order=1)
            except ImportError:

                zi = np.round(np.linspace(0, current_slices - 1, target_slices)).astype(int)
                yi = np.round(np.linspace(0, current_h - 1, target_h)).astype(int)
                xi = np.round(np.linspace(0, current_w - 1, target_w)).astype(int)
                volume = volume[zi][:, yi][:, :, xi]

        mean = float(volume.mean())
        std = float(volume.std())
        volume = (volume - mean) / (std + 1e-6)

        try:
            from scipy.ndimage import center_of_mass, shift
            mass = np.abs(volume) + 1e-8
            com = np.asarray(center_of_mass(mass))
            centre = np.asarray(volume.shape, dtype=np.float64) / 2.0 - 0.5
            delta = centre - com

            delta = np.clip(delta, -4.0, 4.0)
            volume = shift(volume, shift=delta, order=1, mode="nearest")
        except ImportError:
            pass

        return volume.astype(np.float32)
    
    def _load_nifti_masks(self, other_dir: Path) -> Dict[str, torch.Tensor]:
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
                    

                    if len(data.shape) == 3:
                        data = self._standardize_volume(data, self.recommended_n_slices)
                    
                    nifti_tensors[filename.replace('.nii.gz', '')] = torch.from_numpy(data)
                    
                except Exception as e:
                    print(f"Warning: Error loading {filename}: {e}")
                    
        return nifti_tensors
    
    def generate_visualization_figures(self, patient_id: str = 'Patient_10008101') -> List[str]:
        print(f"\n=== Generating Visualization Figures for {patient_id} ===")
        
        patient_dir = self.source_dir / patient_id
        dicom_dir = patient_dir / "dicom"
        other_dir = patient_dir / "other"
        
        if not dicom_dir.exists():
            print(f"Error: DICOM directory not found for {patient_id}")
            return []
        

        patient_figure_dir = self.figure_dir / patient_id
        patient_figure_dir.mkdir(exist_ok=True)
        
        generated_figures = []
        

        dicom_files = sorted(dicom_dir.glob("*.dcm"))
        

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
        

        for seq_num, files in sequences.items():
            files.sort(key=lambda x: x[0])
            

            n_representative = min(self.recommended_n_slices, len(files))
            if n_representative > len(files):
                n_representative = len(files)
                
            indices = np.linspace(0, len(files) - 1, n_representative, dtype=int)
            selected_files = [files[i] for i in indices]
            

            fig_path = self._create_sequence_figure(selected_files, seq_num, patient_figure_dir, patient_id)
            if fig_path:
                generated_figures.append(fig_path)
        

        if other_dir.exists():
            nifti_figures = self._create_nifti_figures(other_dir, patient_figure_dir, patient_id)
            generated_figures.extend(nifti_figures)
        
        print(f"Generated {len(generated_figures)} figures for {patient_id}")
        return generated_figures
    
    def _create_sequence_figure(self, files: List[Tuple[int, Path]], seq_num: int, 
                              figure_dir: Path, patient_id: str) -> Optional[str]:
        try:

            images = []
            slice_numbers = []
            
            for slice_num, file_path in files:
                dcm = pydicom.dcmread(file_path, force=True)
                img = dcm.pixel_array.astype(np.float32)
                images.append(img)
                slice_numbers.append(slice_num)
            
            if not images:
                return None
            

            n_cols = 8
            n_rows = int(np.ceil(len(images) / n_cols))
            

            setup_cardioai_style()
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows))
            fig.suptitle(f'{patient_id} - Sequence {seq_num} - {len(images)} slices', 
                        fontsize=16, fontweight='bold')
            

            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            

            for idx, (img, slice_num) in enumerate(zip(images, slice_numbers)):
                if idx < len(axes):
                    ax = axes[idx]
                    

                    if cv2 is not None:
                        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    else:

                        img_norm = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                    
                    ax.imshow(img_norm, cmap='gray')
                    ax.set_title(f'Slice {slice_num}', fontsize=10)
                    ax.axis('off')
            

            for idx in range(len(images), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            

            figure_path = figure_dir / f'{patient_id}_sequence_{seq_num}_slices'
            save_cardioai_figure(fig, figure_path)
            plt.close()
            
            return str(figure_path)
            
        except Exception as e:
            print(f"Error creating sequence: {e}")
            return None
    
    def _create_nifti_figures(self, other_dir: Path, figure_dir: Path, patient_id: str) -> List[str]:
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

                    img = nib.load(file_path)
                    data = img.get_fdata()
                    

                    n_slices = data.shape[2] if len(data.shape) >= 3 else 1
                    n_representative = min(16, n_slices)
                    
                    if n_slices > 1:
                        slice_indices = np.linspace(0, n_slices - 1, n_representative, dtype=int)
                        

                        n_cols = 4
                        n_rows = int(np.ceil(n_representative / n_cols))
                        

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
                                

                                if 'mask' in filename.lower():
                                    cmap = 'viridis'
                                else:
                                    cmap = 'gray'
                                
                                im = ax.imshow(slice_data, cmap=cmap)
                                ax.set_title(f'Slice {slice_idx}', fontsize=10)
                                ax.axis('off')
                                

                                if idx == 0:
                                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        

                        for idx in range(n_representative, len(axes)):
                            axes[idx].axis('off')
                        
                        plt.tight_layout()
                        

                        clean_name = filename.replace('.nii.gz', '').replace('.', '_')
                        figure_path = figure_dir / f'{patient_id}_{clean_name}_mask'
                        save_cardioai_figure(fig, figure_path)
                        plt.close()
                        
                        generated_figures.append(str(figure_path))
                        
                except Exception as e:
                    print(f"Error creating NIfTI plot for {filename}: {e}")
        
        return generated_figures
    
    def save_analysis_report(self) -> str:
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
    import argparse
    

    parser = argparse.ArgumentParser(description='CardioAI Dataset Analysis and Processing')
    parser.add_argument('--results-dir', type=str, help='Results directory path')
    args = parser.parse_args()
    
    print("CardioAI Dataset Analysis and Processing")
    print("=" * 50)
    

    analyzer = CardioAIDatasetAnalyzer(results_dir=args.results_dir)
    

    print("\nStep 1: Analyzing all 108 patients...")
    dataset_summary = analyzer.analyze_all_patients()
    

    print("\nStep 2: Detailed analysis of Patient_10008101 and Patient_10008292...")
    detailed_analysis = analyzer.analyze_detailed_patients(['Patient_10008101', 'Patient_10008292'])
    

    print("\nStep 3: Processing ALL 108 patients to PyTorch tensor format...")
    processing_results = analyzer.process_to_pytorch_tensors()
    

    print("\nStep 4: Generating visualization figures...")
    figures = analyzer.generate_visualization_figures('Patient_10008101')
    

    print("\nStep 5: Saving comprehensive analysis report...")
    report_path = analyzer.save_analysis_report()
    

    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    print(f"[OK] Analyzed {dataset_summary.get('total_patients', 0)} patients")
    print(f"[OK] Recommended n_slices: {analyzer.recommended_n_slices}")
    print(f"[OK] Generated {len(figures)} visualization figures")
    print(f"[OK] Processed patients to .pt format")
    print(f"[OK] Analysis report: {report_path}")
    

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
    print(f"   - Target directory: ./data/mri_tensors")
    print("   - Format: PyTorch .pt tensors")
    print("   - Standardized dimensions for consistent training")
    print("   - Include both DICOM sequences and NIfTI masks")
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()