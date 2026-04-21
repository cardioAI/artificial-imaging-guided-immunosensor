"""
training.trainer
================

Training loop for the Figure 1b / 1c contrastive framework. Trains the
:class:`models.CardioAIModel` on the 108-patient CardioAI cohort with
symmetric InfoNCE loss and cosine-annealed AdamW, and writes learning
curves and checkpoints under the configured results directory.

Contents
--------
* ``CardioAITrainer`` -- owns optimiser, schedulers, epoch loops and
  checkpointing. Methods: ``train_epoch``, ``validate_epoch``, ``train``,
  ``save_checkpoint``, ``save_results``, ``plot_training_curves``,
  ``_save_progression_snapshot`` (Figure 1c step-wise embeddings).
* ``parse_patient_range`` -- parses ``"1-57"`` / ``"58-77"`` style CLI
  specifications used by ``create_custom_split``.
* ``create_custom_split`` -- splits the dataset into train/val ranges.
* ``main()`` -- CLI entry point wiring dataset + model + trainer.

The dataset class itself lives in :mod:`data.pt_dataset`; this module only
imports it.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
from tqdm import tqdm
import pickle
import torch.nn.functional as F

# Import our model architecture from the reorganised subpackages.
from models import CardioAIModel, create_model
from data.pt_dataset import CardioAIDataset, custom_collate_fn

# CardioAI plotting / styling helpers.
from utils.styling import (get_cardioai_colors, save_cardioai_figure,
                           setup_cardioai_style)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class CardioAITrainer:
    """
    Trainer class for CardioAI contrastive learning model
    """
    
    def __init__(self,
                 model: CardioAIModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 learning_rate: float = 1e-5,
                 weight_decay: float = 1e-4,
                 results_dir: str = None,
                 enable_early_stopping: bool = False,
                 num_epochs: int = 100,
                 progression_epochs: Optional[List[int]] = None):
        """
        Initialize trainer
        
        Args:
            model: CardioAI model instance
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on (cuda/cpu)
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            results_dir: Directory to save results
            enable_early_stopping: Whether to enable early stopping (default: False)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer with appropriate regularization for small dataset
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,  # Use weight_decay as is for stability
            betas=(0.9, 0.999)  # Standard momentum for better convergence
        )
        
        # Store num_epochs for scheduler
        self.num_epochs = num_epochs
        
        # Manuscript: "initial learning rate of 1e-5 with warm-up scheduling".
        # We use a linear warm-up for the first ``warmup_epochs`` epochs
        # (LR ramps from 0 -> base) followed by cosine annealing across the
        # remaining epochs.
        self.warmup_epochs = max(1, int(0.05 * num_epochs))
        self.base_lr = learning_rate
        self.scheduler = optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[
                optim.lr_scheduler.LinearLR(self.optimizer,
                                             start_factor=1e-3,
                                             end_factor=1.0,
                                             total_iters=self.warmup_epochs),
                optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                     T_max=max(1, num_epochs - self.warmup_epochs),
                                                     eta_min=learning_rate * 0.01),
            ],
            milestones=[self.warmup_epochs],
        )

        # Validation-based LR reducer (kept for long-run stability).
        self.val_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.7,
            patience=10,
            min_lr=learning_rate * 0.005,
            threshold=1e-4,
        )

        # Temperature annealing: linearly decay the InfoNCE temperature from
        # its initial value to 10% of it across training. Annealing sharpens
        # the similarity distribution late in training, as per the
        # "temperature annealing" remark in the manuscript's Methods.
        self.initial_temperature = float(self.model.contrastive_loss.temperature.item())
        self.final_temperature = self.initial_temperature * 0.1
        
        # Results directory
        if results_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_dir = Path(f"F:/datasets_cardioAI/BICL_cardioAI/results/{timestamp}/script3_training_pipeline")
        else:
            self.results_dir = Path(results_dir)
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rates': []
        }
        
        # Early stopping parameters (configurable)
        self.enable_early_stopping = enable_early_stopping
        self.best_val_loss = float('inf')
        self.patience = 15 if enable_early_stopping else float('inf')  # Infinite patience when disabled
        self.patience_counter = 0
        self.min_delta = 1e-4  # Minimum improvement threshold

        # Figure 1c progression: checkpoint epochs at which paired embeddings
        # are persisted so the biomarker<->image alignment tightening can be
        # rendered across training. Default mirrors the figure's three steps.
        if progression_epochs is None:
            progression_epochs = self._default_progression_epochs(num_epochs)
        self.progression_epochs = sorted({int(e) for e in progression_epochs if 1 <= int(e) <= num_epochs})
        self.progression_dir = self.results_dir / "progression"
        self.progression_dir.mkdir(parents=True, exist_ok=True)

        print(f"Trainer initialized")
        print(f"  Device: {device}")
        print(f"  Results directory: {self.results_dir}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Early stopping: {'Enabled' if enable_early_stopping else 'Disabled (Full training)'}")
        print(f"  Progression checkpoints (Figure 1c): {self.progression_epochs}")
        
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch in progress_bar:
            # Move data to device
            biomarkers = batch['biomarkers'].to(self.device)
            images = batch['images'].to(self.device)
            
            # Manuscript Methods: "Gaussian noise injection for biomarker
            # inputs of sigma = 0.0005 and spatial transformations for MRI
            # volumes, comprising random rotations (+-5 deg) and
            # translations (+-2 voxels)."
            if self.model.training:
                biomarkers = biomarkers + torch.randn_like(biomarkers) * 0.0005
                images = _augment_mri_volumes(images,
                                              max_rotation_deg=5.0,
                                              max_translation_voxels=2)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            output = self.model(biomarkers, images)
            # Manuscript eq 3: optimise the weighted sum of contrastive,
            # hard-negative, reconstruction (retrieval MSE) and
            # regularization terms rather than the bare contrastive loss.
            loss = output['total_loss']
            accuracy = output['metrics']['accuracy']

            # Backward pass
            loss.backward()

            # Gradient clipping for stability (proper value)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            num_batches += 1

            progress_bar.set_postfix({
                'Total': f"{loss.item():.4f}",
                'NCE': f"{output['contrastive_loss'].item():.4f}",
                'Hard': f"{output['hard_negative_loss'].item():.4f}",
                'Rec': f"{output['reconstruction_loss'].item():.4f}",
                'Acc': f"{accuracy.item():.4f}",
            })
        
        # Calculate epoch metrics
        epoch_loss = total_loss / num_batches
        epoch_accuracy = total_accuracy / num_batches
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_accuracy
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        all_biomarker_embeddings = []
        all_image_embeddings = []
        all_fused_embeddings = []
        all_patient_ids = []
        all_hff_values = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                # Move data to device
                biomarkers = batch['biomarkers'].to(self.device)
                images = batch['images'].to(self.device)
                
                # IMPORTANT: No noise injection during validation (unlike training)
                # Ensure model is in eval mode (no dropout, no noise)
                self.model.eval()
                
                # Forward pass
                output = self.model(biomarkers, images, return_embeddings=True)
                loss = output['total_loss']
                accuracy = output['metrics']['accuracy']
                
                # Store embeddings for analysis
                all_biomarker_embeddings.append(output['biomarker_embedding'].cpu())
                all_image_embeddings.append(output['image_embedding'].cpu())
                all_fused_embeddings.append(output['fused_embedding'].cpu())
                all_patient_ids.extend(batch['patient_id'])
                all_hff_values.append(batch['hff'].cpu())
                all_labels.append(batch['label'].cpu())
                
                # Update metrics
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1
        
        # Calculate validation metrics
        val_loss = total_loss / num_batches
        val_accuracy = total_accuracy / num_batches
        
        # Store validation embeddings for analysis
        self.val_embeddings = {
            'biomarker_embeddings': torch.cat(all_biomarker_embeddings, dim=0),
            'image_embeddings': torch.cat(all_image_embeddings, dim=0),
            'fused_embeddings': torch.cat(all_fused_embeddings, dim=0),
            'patient_ids': all_patient_ids,
            'hff_values': torch.cat(all_hff_values, dim=0),
            'labels': torch.cat(all_labels, dim=0)
        }
        
        return {
            'loss': val_loss,
            'accuracy': val_accuracy
        }
    
    def train(self, num_epochs: int = 200) -> Dict[str, List[float]]:
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs to train
            
        Returns:
            Training history dictionary
        """
        mode_str = "with early stopping" if self.enable_early_stopping else "for full duration (no early stopping)"
        print(f"\nStarting training for {num_epochs} epochs {mode_str}")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            # Linear temperature annealing from initial -> final across
            # training. ``set_temperature`` mutates both the contrastive
            # and hard-negative heads in place (see CardioAIModel).
            if num_epochs > 1:
                progress = epoch / (num_epochs - 1)
                tau = self.initial_temperature + progress * (
                    self.final_temperature - self.initial_temperature
                )
                self.model.set_temperature(tau)

            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Learning rate scheduling - use both schedulers for better convergence
            self.scheduler.step()  # Cosine annealing scheduler
            self.val_scheduler.step(val_metrics['loss'])  # Validation-based scheduler
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_accuracy'].append(train_metrics['accuracy'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['learning_rates'].append(current_lr)
            
            # Print progress
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Train Acc: {train_metrics['accuracy']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"LR: {current_lr:.6f}")
            
            # Early stopping logic (only if enabled)
            if val_metrics['loss'] < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint(epoch, is_best=True)
                print(f"  → New best validation loss: {val_metrics['loss']:.4f}")
                
            elif self.enable_early_stopping:
                self.patience_counter += 1
                print(f"  → No improvement for {self.patience_counter} epochs")
            
            # Early stopping check (only if enabled)
            if self.enable_early_stopping and self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered after {self.patience} epochs without improvement")
                print(f"Best validation loss: {self.best_val_loss:.4f}")
                break
            
            # Save periodic checkpoint
            if (epoch + 1) % 25 == 0:
                self.save_checkpoint(epoch)

            # Figure 1c: persist paired embeddings at the configured
            # progression epochs. Uses the val_embeddings collected above.
            if (epoch + 1) in self.progression_epochs:
                self._save_progression_snapshot(epoch + 1)

        print("=" * 60)
        print("Training completed!")
        
        # Save final results
        self.save_results()
        
        return self.history
    
    @staticmethod
    def _default_progression_epochs(num_epochs: int) -> List[int]:
        """Default Figure 1c checkpoints: first epoch, midpoint, final epoch."""
        if num_epochs <= 0:
            return []
        if num_epochs == 1:
            return [1]
        if num_epochs == 2:
            return [1, 2]
        return [1, max(2, num_epochs // 2), num_epochs]

    def _save_progression_snapshot(self, epoch: int):
        """Persist paired (biomarker, image) validation embeddings for Figure 1c.

        Writes ``progression/step_{k}.pt`` where ``k`` is the 1-based index
        of this checkpoint within ``self.progression_epochs``. The file
        contains the validation embeddings, patient ids, labels, HFF values,
        the epoch number, and the current mean contrastive accuracy.
        """
        if not hasattr(self, "val_embeddings"):
            return
        try:
            step_index = self.progression_epochs.index(epoch) + 1
        except ValueError:
            step_index = epoch

        snapshot = {
            "epoch": epoch,
            "step_index": step_index,
            "total_steps": len(self.progression_epochs),
            "biomarker_embeddings": self.val_embeddings["biomarker_embeddings"].clone(),
            "image_embeddings": self.val_embeddings["image_embeddings"].clone(),
            "fused_embeddings": self.val_embeddings["fused_embeddings"].clone(),
            "patient_ids": list(self.val_embeddings["patient_ids"]),
            "hff_values": self.val_embeddings["hff_values"].clone(),
            "labels": self.val_embeddings["labels"].clone(),
            "val_loss": self.history["val_loss"][-1] if self.history["val_loss"] else None,
            "val_accuracy": self.history["val_accuracy"][-1] if self.history["val_accuracy"] else None,
        }
        snapshot_path = self.progression_dir / f"step_{step_index}.pt"
        torch.save(snapshot, snapshot_path)
        print(f"  -> Progression snapshot saved: {snapshot_path.name} "
              f"(epoch {epoch}, step {step_index}/{len(self.progression_epochs)})")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'val_embeddings': getattr(self, 'val_embeddings', None)
        }
        
        if is_best:
            checkpoint_path = self.results_dir / 'best_model.pth'
            print(f"  → Saving best model to {checkpoint_path}")
        else:
            checkpoint_path = self.results_dir / f'checkpoint_epoch_{epoch+1}.pth'
        
        torch.save(checkpoint, checkpoint_path)
    
    def save_results(self):
        """Save training results and analysis"""
        # Save training history
        history_path = self.results_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            # Convert numpy types to regular Python types for JSON serialization
            history_json = {}
            for key, values in self.history.items():
                history_json[key] = [float(v) for v in values]
            json.dump(history_json, f, indent=2)
        
        # Save validation embeddings
        if hasattr(self, 'val_embeddings'):
            embeddings_path = self.results_dir / 'validation_embeddings.pkl'
            with open(embeddings_path, 'wb') as f:
                pickle.dump(self.val_embeddings, f)

            # Persist bidirectional retrieval recall + mean reciprocal rank
            # on the validation cohort (manuscript requires top-k + MRR).
            try:
                from retrieval.biomarker_to_image import evaluate_retrieval
                retrieval_metrics = evaluate_retrieval(
                    self.val_embeddings['biomarker_embeddings'],
                    self.val_embeddings['image_embeddings'],
                )
                with open(self.results_dir / 'retrieval_metrics.json', 'w') as fm:
                    json.dump(retrieval_metrics, fm, indent=2)
                print(f"Retrieval metrics: "
                      f"MRR={retrieval_metrics['mean_reciprocal_rank']:.4f}, "
                      f"b2i@1={retrieval_metrics['b2i_recall@1']:.3f}, "
                      f"b2i@5={retrieval_metrics['b2i_recall@5']:.3f}")
            except Exception as exc:
                print(f"[WARNING] Retrieval metric calculation failed: {exc}")
        
        # Create training plots
        self.plot_training_curves()
        
        # Save model configuration
        config_path = self.results_dir / 'model_config.json'
        model_config = {
            'model_class': 'CardioAIModel',
            'embed_dim': self.model.embed_dim,
            'fusion_dim': self.model.fusion_dim,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        print(f"\nResults saved to: {self.results_dir}")
    
    def plot_training_curves(self):
        """Create standardized training curve plots with CardioAI styling"""
        # Setup CardioAI style
        setup_cardioai_style()
        
        # Get CardioAI colors
        colors = get_cardioai_colors(8)
        
        # Create standardized figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        ax1, ax2, ax3, ax4 = axes.flatten()
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss curves
        ax1.plot(epochs, self.history['train_loss'], color=colors[0], label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], color=colors[1], label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Contrastive Loss')
        ax1.legend()
        
        # Accuracy curves
        ax2.plot(epochs, self.history['train_accuracy'], color=colors[2], label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, self.history['val_accuracy'], color=colors[3], label='Validation Accuracy', linewidth=2)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Retrieval Accuracy')
        ax2.legend()
        
        # Learning rate
        ax3.plot(epochs, self.history['learning_rates'], color=colors[4], linewidth=2)
        ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        
        # Loss ratio (validation/training)
        loss_ratio = [v/t for v, t in zip(self.history['val_loss'], self.history['train_loss'])]
        ax4.plot(epochs, loss_ratio, color=colors[5], linewidth=2)
        ax4.axhline(y=1.0, color='#000000', linestyle='--', alpha=0.7)
        ax4.set_title('Validation/Training Loss Ratio', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss Ratio')
        
        plt.tight_layout()
        
        # Save in all standard formats (EPS, PNG, TIFF)
        save_cardioai_figure(fig, self.results_dir / 'training_curves')
        plt.close()

def _augment_mri_volumes(images: torch.Tensor,
                         max_rotation_deg: float,
                         max_translation_voxels: int) -> torch.Tensor:
    """In-plane rotation + 3D translation augmentation for MRI volumes.

    ``images`` has shape (B, C, D, H, W). A single random rotation angle
    (around the depth axis) and a random (dz, dy, dx) integer translation
    are sampled per batch; both are applied to every sample to keep paired
    MRI sequences aligned, matching the manuscript's spatial augmentation
    specification.
    """
    if not images.is_floating_point():
        return images
    B, C, D, H, W = images.shape

    theta = (torch.rand(1, device=images.device).item() * 2 - 1) * max_rotation_deg
    tz = int(torch.randint(-max_translation_voxels, max_translation_voxels + 1, (1,)).item())
    ty = int(torch.randint(-max_translation_voxels, max_translation_voxels + 1, (1,)).item())
    tx = int(torch.randint(-max_translation_voxels, max_translation_voxels + 1, (1,)).item())

    # In-plane rotation via affine_grid + grid_sample on each depth slice.
    if abs(theta) > 0.05:
        rad = torch.tensor(theta * 3.141592653589793 / 180.0, device=images.device)
        cos, sin = torch.cos(rad), torch.sin(rad)
        affine = torch.tensor([[[cos, -sin, 0.0], [sin, cos, 0.0]]],
                              device=images.device, dtype=images.dtype)
        # Collapse (B, C, D) into the batch so 2D affine_grid covers (H, W).
        flat = images.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        grid = torch.nn.functional.affine_grid(
            affine.expand(flat.size(0), -1, -1), flat.size(), align_corners=False
        )
        flat = torch.nn.functional.grid_sample(flat, grid, mode='bilinear',
                                               padding_mode='zeros', align_corners=False)
        images = flat.reshape(B, D, C, H, W).permute(0, 2, 1, 3, 4).contiguous()

    # Integer translations along D / H / W.
    if tz or ty or tx:
        images = torch.roll(images, shifts=(tz, ty, tx), dims=(2, 3, 4))
    return images


def parse_patient_range(range_str: str) -> List[int]:
    """
    Parse patient range string into list of patient indices.
    
    Args:
        range_str: String like "1-57" or "58-77" or "1,5,10-15"
        
    Returns:
        List of patient indices (0-based for array indexing)
    """
    indices = []
    
    # Split by commas for multiple ranges
    parts = range_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Range like "1-57"
            start, end = part.split('-')
            start_idx = int(start) - 1  # Convert to 0-based indexing
            end_idx = int(end) - 1      # Convert to 0-based indexing
            indices.extend(list(range(start_idx, end_idx + 1)))
        else:
            # Single number
            indices.append(int(part) - 1)  # Convert to 0-based indexing
    
    # Remove duplicates and sort
    indices = sorted(list(set(indices)))
    
    return indices

def create_custom_split(dataset: CardioAIDataset, 
                       train_patients: str, 
                       val_patients: str) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    """
    Create custom train/validation split based on specified patient ranges.
    
    Args:
        dataset: CardioAI dataset
        train_patients: String specifying training patients (e.g., "1-57")
        val_patients: String specifying validation patients (e.g., "58-77")
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Parse patient ranges
    train_indices = parse_patient_range(train_patients)
    val_indices = parse_patient_range(val_patients)
    
    # Validate indices are within dataset bounds
    max_idx = len(dataset) - 1
    train_indices = [idx for idx in train_indices if 0 <= idx <= max_idx]
    val_indices = [idx for idx in val_indices if 0 <= idx <= max_idx]
    
    print(f"Custom patient split configuration:")
    print(f"  Training patients: {train_patients} -> {len(train_indices)} patients (indices {min(train_indices) if train_indices else 'none'}-{max(train_indices) if train_indices else 'none'})")
    print(f"  Validation patients: {val_patients} -> {len(val_indices)} patients (indices {min(val_indices) if val_indices else 'none'}-{max(val_indices) if val_indices else 'none'})")
    
    # Check for overlap
    overlap = set(train_indices) & set(val_indices)
    if overlap:
        print(f"  Note: {len(overlap)} patients appear in both training and validation sets (indices: {sorted(list(overlap))})")
    
    # Create subsets
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    return train_dataset, val_dataset

def main(num_epochs=200, batch_size=4, output_dir=None, enable_early_stopping=False,
         train_patients="1-57", val_patients="58-77"):
    """
    Main training function for the MASLD contrastive-learning pipeline.

    The 108-patient cohort is split 57 / 20 / 31: the first 57 patients train
    the cross-modal alignment, the next 20 validate it, and the remaining 31
    are held out for the independent test set evaluated downstream via
    artificial-imaging-guided clustering.

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        output_dir: Custom output directory (optional)
        enable_early_stopping: Whether to enable early stopping (default: False)
        train_patients: Patient range for training (default: "1-57")
        val_patients: Patient range for validation (default: "58-77")
    """
    print("CardioAI Contrastive Learning Training")
    print("=" * 50)
    print(f"Training Configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch Size: {batch_size}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    
    # Create dataset
    print("\nLoading dataset...")
    dataset = CardioAIDataset()
    
    print(f"Dataset size: {len(dataset)} patients")
    
    # Use custom patient split based on specified ranges
    train_dataset, val_dataset = create_custom_split(dataset, train_patients, val_patients)
    
    print(f"Final dataset split: {len(train_dataset)} training, {len(val_dataset)} validation")
    
    # Create data loaders
    # batch_size parameter passed from function argument
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=custom_collate_fn
    )
    
    print(f"Data loaders created with batch size: {batch_size}")
    
    # Create model
    print("\nInitializing model...")
    model = create_model()
    
    # Create trainer (disable early stopping for full training by default)
    trainer = CardioAITrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=1e-5,  # Manuscript specifies 1e-5 with warm-up scheduling
        weight_decay=1e-4,   # Manuscript specifies L2 weight decay 1e-4
        results_dir=output_dir,
        enable_early_stopping=enable_early_stopping,  # Configurable early stopping
        num_epochs=num_epochs  # Pass num_epochs for scheduler
    )
    
    # Start training
    history = trainer.train(num_epochs=num_epochs)
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"Best validation accuracy: {max(history['val_accuracy']):.4f}")
    print(f"Results saved to: {trainer.results_dir}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CardioAI Training Pipeline')
    parser.add_argument('--num-epochs', type=int, default=200,
                       help='Number of training epochs (default: 200)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for training (default: 4, reduced for small dataset)')
    parser.add_argument('--enable-early-stopping', action='store_true',
                       help='Enable early stopping (disabled by default for full training)')
    parser.add_argument('--results-dir', type=str, 
                       help='Results directory for output (optional)')
    
    # Patient split arguments for custom training/validation configuration
    parser.add_argument('--train-patients', type=str, default='1-57',
                       help='Range of patients for training (default: 1-57). Format: "start-end" or comma-separated list')
    parser.add_argument('--val-patients', type=str, default='58-77',
                       help='Range of patients for validation (default: 58-77). Format: "start-end" or comma-separated list')
    
    args = parser.parse_args()
    
    main(num_epochs=args.num_epochs, batch_size=args.batch_size, 
         output_dir=args.results_dir, enable_early_stopping=args.enable_early_stopping,
         train_patients=args.train_patients, val_patients=args.val_patients)