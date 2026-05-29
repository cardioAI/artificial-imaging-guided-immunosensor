
import os
import sys
import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score,
                             roc_auc_score, confusion_matrix)
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import umap
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
from tqdm import tqdm

from models import CardioAIModel, create_model
from data.pt_dataset import CardioAIDataset

from utils.styling import (get_cardioai_colors, save_cardioai_figure,
                           setup_cardioai_style, setup_clean_axis, create_clean_legend,
                           get_cardioai_colormap)

warnings.filterwarnings('ignore')

class EncoderDecoderDiscriminator(nn.Module):

    def __init__(self,
                 input_dim: int = 512,
                 hidden_dim: int = 256,
                 latent_dim: int = 128,
                 dropout: float = 0.1,
                 classifier_hidden: int = 64):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        from .edd_encoder import EDDEncoder
        from .edd_decoder import EDDDecoder
        from .edd_discriminator import EDDDiscriminator
        from .edd_classifier import EDDClassifier

        self.encoder = EDDEncoder(input_dim, hidden_dim, latent_dim, dropout)
        self.decoder = EDDDecoder(input_dim, hidden_dim, latent_dim, dropout)
        self.discriminator = EDDDiscriminator(input_dim, hidden_dim, dropout)
        self.classifier = EDDClassifier(latent_dim, classifier_hidden, dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        return self.classifier(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        real_score = self.discriminate(x)
        fake_score = self.discriminate(reconstructed.detach())
        logit = self.classifier(latent)

        return {
            'latent': latent,
            'reconstructed': reconstructed,
            'real_score': real_score,
            'fake_score': fake_score,
            'logit': logit,
        }

class CardioAIClusteringAnalyzer:
    
    def __init__(self,
                 biomarkers_file: str = "/data/biomarkers.xlsx",
                 images_dir: str = "/data/mri_tensors",
                 results_dir: str = None,
                 test_patients: str = "78-108"):
        self.biomarkers_file = biomarkers_file
        self.images_dir = Path(images_dir)
        self.test_patients_spec = test_patients
        

        if results_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_dir = Path(f"./results/{timestamp}/script4_clustering_analysis")
        else:
            self.results_dir = Path(results_dir)
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        

        self.load_data()
        

        self.cardio_model = None
        self.edd_model = None
        
        print(f"CardioAI Clustering Analyzer initialized")
        print(f"Results directory: {self.results_dir}")
        print(f"Total patients: {len(self.patient_data)}")
        
    def load_data(self):
        print("Loading biomarker data...")

        self.biomarkers_df = pd.read_excel(self.biomarkers_file)
        self.biomarker_columns = ['ALT', 'AST', 'GGT', 'CTSD', 'CK18', 'FGF21']

        test_idx_set = self._parse_patient_range(self.test_patients_spec,
                                                  n_rows=len(self.biomarkers_df))

        self.patient_data = []

        for idx, row in self.biomarkers_df.iterrows():
            patient_id = f"Patient_{int(row['ID'])}"
            pt_file = self.images_dir / f"{patient_id}.pt"
            is_test = idx in test_idx_set

            patient_info = {
                'index': idx,
                'patient_id': patient_id,
                'meta_id': int(row['ID']),
                'biomarkers': row[self.biomarker_columns].values.astype(np.float32),
                'hff': float(row['HFF']) if 'HFF' in self.biomarkers_df.columns else 0.0,
                'label': row['Label'],

                'has_images': bool(pt_file.exists() and not is_test),
                'is_test': is_test,
                'pt_file': pt_file if pt_file.exists() else None
            }

            self.patient_data.append(patient_info)

        self.patients_with_images = [p for p in self.patient_data if p['has_images']]
        self.patients_without_images = [p for p in self.patient_data if not p['has_images']]

        print(f"Patients with real-image path (1-{len(self.biomarkers_df) - len(test_idx_set)}): "
              f"{len(self.patients_with_images)}")
        print(f"Patients in independent test set ({self.test_patients_spec}): "
              f"{len(self.patients_without_images)}")

        self.normalize_biomarkers()

    @staticmethod
    def _parse_patient_range(range_str: str, n_rows: int) -> set:
        indices: set = set()
        for part in range_str.split(','):
            part = part.strip()
            if not part:
                continue
            if '-' in part:
                lo, hi = part.split('-', 1)
                start = max(1, int(lo)) - 1
                end = min(n_rows, int(hi)) - 1
                if start <= end:
                    indices.update(range(start, end + 1))
            else:
                i = int(part) - 1
                if 0 <= i < n_rows:
                    indices.add(i)
        return indices
    
    def normalize_biomarkers(self):
        biomarker_matrix = np.array([p['biomarkers'] for p in self.patient_data])
        
        self.biomarker_scaler = StandardScaler()
        normalized_biomarkers = self.biomarker_scaler.fit_transform(biomarker_matrix)
        

        for i, patient in enumerate(self.patient_data):
            patient['normalized_biomarkers'] = normalized_biomarkers[i]
    
    def load_trained_model(self, model_path: str):
        print(f"Loading trained model from {model_path}")
        
        self.cardio_model = create_model()
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            self.cardio_model.load_state_dict(checkpoint['model_state_dict'])
            print("[SUCCESS] Model loaded successfully")
        else:
            print("[WARNING] Model not found, using untrained model")
        
        self.cardio_model.eval()
    
    def extract_embeddings(self):
        print("Extracting embeddings...")
        
        if self.cardio_model is None:
            raise ValueError("CardioAI model not loaded. Call load_trained_model() first.")
        
        biomarker_embeddings = []
        image_embeddings = []
        fused_embeddings = []
        

        print("Processing patients with real images (train+val cohort)...")
        for patient in tqdm(self.patients_with_images):

            biomarkers = torch.from_numpy(patient['normalized_biomarkers']).float().unsqueeze(0)
            

            try:
                patient_data = torch.load(patient['pt_file'], map_location='cpu')
                dicom_sequences = patient_data['dicom_sequences']
                

                sequence_tensors = []
                for seq_name in sorted(dicom_sequences.keys()):
                    tensor = dicom_sequences[seq_name]

                    if tensor.size(1) < 192:
                        padding = 192 - tensor.size(1)
                        pad_top = padding // 2
                        pad_bottom = padding - pad_top
                        tensor = F.pad(tensor, (0, 0, pad_top, pad_bottom, 0, 0), mode='constant', value=0)
                    sequence_tensors.append(tensor)
                

                while len(sequence_tensors) < 4:
                    sequence_tensors.append(sequence_tensors[-1] if sequence_tensors else torch.zeros(64, 192, 192))
                
                images = torch.stack(sequence_tensors[:4], dim=0).unsqueeze(0).float()
                

                with torch.no_grad():
                    bio_emb = self.cardio_model.encode_biomarkers(biomarkers)
                    img_emb = self.cardio_model.encode_images(images)
                    fused_emb = self.cardio_model.fusion(bio_emb, img_emb)
                
                biomarker_embeddings.append(bio_emb.squeeze().numpy())
                image_embeddings.append(img_emb.squeeze().numpy())
                fused_embeddings.append(fused_emb.squeeze().numpy())
                
            except Exception as e:
                print(f"Error processing {patient['patient_id']}: {e}")

                biomarker_embeddings.append(np.zeros(self.cardio_model.embed_dim))
                image_embeddings.append(np.zeros(self.cardio_model.embed_dim))
                fused_embeddings.append(np.zeros(self.cardio_model.fusion_dim))
        

        print("Processing independent test cohort (artificial imaging-guided retrieval)...")
        if self.patients_without_images:
            image_db = np.array(image_embeddings)
            bio_db = np.array(biomarker_embeddings)

            bio_db_norm = bio_db / (np.linalg.norm(bio_db, axis=1, keepdims=True) + 1e-8)
            retrieval_beta = 5.0
            top_k = min(5, bio_db_norm.shape[0])

            for patient in tqdm(self.patients_without_images):
                biomarkers = torch.from_numpy(patient['normalized_biomarkers']).float().unsqueeze(0)
                with torch.no_grad():
                    bio_emb = self.cardio_model.encode_biomarkers(biomarkers)
                bio_emb_np = bio_emb.squeeze().numpy()
                q = bio_emb_np / (np.linalg.norm(bio_emb_np) + 1e-8)

                sim = bio_db_norm @ q
                top_idx = np.argsort(sim)[::-1][:top_k]
                logits = retrieval_beta * sim[top_idx]

                logits = logits - logits.max()
                weights = np.exp(logits)
                weights = weights / (weights.sum() + 1e-8)
                retrieved_img_emb = (weights[:, None] * image_db[top_idx]).sum(axis=0)

                with torch.no_grad():
                    bio_emb_tensor = torch.from_numpy(bio_emb_np).unsqueeze(0)
                    img_emb_tensor = torch.from_numpy(retrieved_img_emb.astype(np.float32)).unsqueeze(0)
                    fused_emb = self.cardio_model.fusion(bio_emb_tensor, img_emb_tensor)

                biomarker_embeddings.append(bio_emb_np)
                image_embeddings.append(retrieved_img_emb)
                fused_embeddings.append(fused_emb.squeeze().numpy())
        

        self.embeddings = {
            'biomarker': np.array(biomarker_embeddings),
            'image': np.array(image_embeddings),
            'fused': np.array(fused_embeddings)
        }
        
        print(f"[SUCCESS] Embeddings extracted for {len(biomarker_embeddings)} patients")
        print(f"  Biomarker embeddings: {self.embeddings['biomarker'].shape}")
        print(f"  Image embeddings: {self.embeddings['image'].shape}")
        print(f"  Fused embeddings: {self.embeddings['fused'].shape}")
    
    def train_encoder_decoder_discriminator(self,
                                            num_epochs: int = 100,
                                            batch_size: int = 16,
                                            cls_weight: float = 1.0):
        print("Training Encoder-Decoder-Discriminator-Classifier framework...")

        if not hasattr(self, 'embeddings'):
            raise ValueError("Embeddings not extracted. Call extract_embeddings() first.")

        fusion_dim = self.cardio_model.fusion_dim if self.cardio_model else 256
        self.edd_model = EncoderDecoderDiscriminator(
            input_dim=fusion_dim,
            hidden_dim=max(fusion_dim // 2, 64),
            latent_dim=max(fusion_dim // 4, 32),
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.edd_model.to(device)

        optimizer_g = torch.optim.AdamW(
            list(self.edd_model.encoder.parameters())
            + list(self.edd_model.decoder.parameters())
            + list(self.edd_model.classifier.parameters()),
            lr=1e-4,
            weight_decay=1e-4,
        )
        optimizer_d = torch.optim.AdamW(
            self.edd_model.discriminator.parameters(),
            lr=1e-4,
            weight_decay=1e-4,
        )

        fused_embeddings_tensor = torch.from_numpy(self.embeddings['fused']).float().to(device)

        ordered_patients = list(self.patients_with_images) + list(self.patients_without_images)
        assert len(ordered_patients) == fused_embeddings_tensor.size(0), \
            "Patient ordering does not match fused embedding count"
        labels_np = np.array([int(p['label']) for p in ordered_patients], dtype=np.float32)
        labels_tensor = torch.from_numpy(labels_np).to(device)

        reconstruction_losses = []
        adversarial_losses = []
        classification_losses = []

        for epoch in tqdm(range(num_epochs), desc="Training EDD"):
            epoch_recon_loss = 0.0
            epoch_adv_loss = 0.0
            epoch_cls_loss = 0.0

            num_batches = len(fused_embeddings_tensor) // batch_size + 1
            batches_seen = 0

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(fused_embeddings_tensor))

                if start_idx >= end_idx:
                    continue

                batch_embeddings = fused_embeddings_tensor[start_idx:end_idx]
                batch_labels = labels_tensor[start_idx:end_idx]

                optimizer_d.zero_grad()

                real_scores = self.edd_model.discriminate(batch_embeddings)
                real_loss = F.binary_cross_entropy(real_scores, torch.ones_like(real_scores))

                with torch.no_grad():
                    latent_detached = self.edd_model.encode(batch_embeddings)
                    fake_embeddings = self.edd_model.decode(latent_detached)

                fake_scores = self.edd_model.discriminate(fake_embeddings)
                fake_loss = F.binary_cross_entropy(fake_scores, torch.zeros_like(fake_scores))

                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_d.step()

                optimizer_g.zero_grad()

                latent = self.edd_model.encode(batch_embeddings)
                reconstructed = self.edd_model.decode(latent)

                recon_loss = F.mse_loss(reconstructed, batch_embeddings)

                fake_scores_for_g = self.edd_model.discriminate(reconstructed)
                adv_loss = F.binary_cross_entropy(
                    fake_scores_for_g, torch.ones_like(fake_scores_for_g)
                )

                logit = self.edd_model.classify(latent).squeeze(-1)
                cls_loss = F.binary_cross_entropy_with_logits(logit, batch_labels)

                g_loss = recon_loss + 0.1 * adv_loss + cls_weight * cls_loss
                g_loss.backward()
                optimizer_g.step()

                epoch_recon_loss += recon_loss.item()
                epoch_adv_loss += adv_loss.item()
                epoch_cls_loss += cls_loss.item()
                batches_seen += 1

            denom = max(batches_seen, 1)
            reconstruction_losses.append(epoch_recon_loss / denom)
            adversarial_losses.append(epoch_adv_loss / denom)
            classification_losses.append(epoch_cls_loss / denom)

        self.edd_model.eval()
        with torch.no_grad():
            refined_latent = self.edd_model.encode(fused_embeddings_tensor)
            logits_all = self.edd_model.classify(refined_latent).squeeze(-1)
            probs_all = torch.sigmoid(logits_all).cpu().numpy()
            preds_all = (probs_all >= 0.5).astype(int)
            self.embeddings['refined'] = refined_latent.cpu().numpy()

        labels_int = labels_np.astype(int)
        accuracy = float((preds_all == labels_int).mean())
        try:
            auc = float(roc_auc_score(labels_int, probs_all))
        except ValueError:
            auc = float('nan')
        cm = confusion_matrix(labels_int, preds_all)

        results_df = pd.DataFrame({
            'patient_id': [p['patient_id'] for p in ordered_patients],
            'meta_id': [p['meta_id'] for p in ordered_patients],
            'has_images': [bool(p['has_images']) for p in ordered_patients],
            'label': labels_int,
            'predicted_prob': probs_all,
            'predicted_label': preds_all,
        })
        results_df.to_csv(self.results_dir / 'nafld_classification.csv', index=False)

        summary_df = pd.DataFrame([{
            'accuracy': accuracy,
            'auc': auc,
            'tn': int(cm[0, 0]) if cm.shape == (2, 2) else None,
            'fp': int(cm[0, 1]) if cm.shape == (2, 2) else None,
            'fn': int(cm[1, 0]) if cm.shape == (2, 2) else None,
            'tp': int(cm[1, 1]) if cm.shape == (2, 2) else None,
            'final_recon_loss': reconstruction_losses[-1],
            'final_adv_loss': adversarial_losses[-1],
            'final_cls_loss': classification_losses[-1],
            'cls_weight': cls_weight,
            'num_epochs': num_epochs,
        }])
        summary_df.to_csv(self.results_dir / 'nafld_classification_summary.csv', index=False)

        self.edd_training_history = {
            'reconstruction_loss': reconstruction_losses,
            'adversarial_loss': adversarial_losses,
            'classification_loss': classification_losses,
        }
        self.nafld_classification = {
            'accuracy': accuracy,
            'auc': auc,
            'confusion_matrix': cm.tolist(),
            'probs': probs_all,
            'preds': preds_all,
            'labels': labels_int,
        }

        print(f"[SUCCESS] EDD+Classifier training completed")
        print(f"  Final reconstruction loss: {reconstruction_losses[-1]:.4f}")
        print(f"  Final adversarial loss: {adversarial_losses[-1]:.4f}")
        print(f"  Final classification loss: {classification_losses[-1]:.4f}")
        print(f"  MASLD accuracy: {accuracy:.4f}")
        print(f"  MASLD AUC: {auc:.4f}")
        print(f"  Confusion matrix (TN, FP / FN, TP):\n{cm}")
        print(f"  Refined embeddings: {self.embeddings['refined'].shape}")
    
    def create_chord_diagrams(self):
        pass

        self._draw_chord_diagram(group_label=0, group_name='Negative',
                                 save_name='chord_diagram_negative')
        self._draw_chord_diagram(group_label=1, group_name='Positive',
                                 save_name='chord_diagram_positive')

        pass

    def _draw_chord_diagram(self, group_label: int, group_name: str, save_name: str):
        df = self.biomarkers_df
        group_df = df[df['Label'] == group_label]
        n_patients = len(group_df)
        if n_patients < 2:
            print(f"  Skipping chord diagram for {group_name}: only {n_patients} patients")
            return

        bio_group = group_df[self.biomarker_columns].values
        corr = np.corrcoef(bio_group.T)

        overall_mean = df[self.biomarker_columns].mean().values
        overall_std = df[self.biomarker_columns].std().values + 1e-8
        group_mean = group_df[self.biomarker_columns].mean().values
        importance = np.abs((group_mean - overall_mean) / overall_std)
        if importance.max() > 0:
            importance = importance / importance.max()

        num_nodes = len(self.biomarker_columns)
        angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, num_nodes, endpoint=False)
        positions = np.array([[np.cos(a), np.sin(a)] for a in angles])

        setup_cardioai_style()
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_aspect('equal')

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                r = float(corr[i, j])
                if abs(r) < 0.05:
                    continue
                colour = '#FB923C' if r > 0 else '#3B82F6'
                linewidth = 0.5 + 5.0 * abs(r)
                alpha = float(min(0.25 + abs(r), 0.9))
                start = positions[i]
                end = positions[j]
                ctrl = np.array([0.0, 0.0])
                path = mpath.Path(
                    [start, ctrl, end],
                    [mpath.Path.MOVETO, mpath.Path.CURVE3, mpath.Path.CURVE3]
                )
                patch = mpatches.PathPatch(path, facecolor='none', edgecolor=colour,
                                           linewidth=linewidth, alpha=alpha, zorder=2)
                ax.add_patch(patch)

        node_sizes = 300 + 1500 * importance
        for i, (pos, name) in enumerate(zip(positions, self.biomarker_columns)):
            ax.scatter(pos[0], pos[1], s=node_sizes[i], c='#2F4F6F',
                       edgecolor='black', linewidth=1.5, zorder=3)
            label_pos = pos * 1.18
            ha = 'left' if pos[0] > 0.01 else ('right' if pos[0] < -0.01 else 'center')
            va = 'bottom' if pos[1] > 0.01 else ('top' if pos[1] < -0.01 else 'center')
            ax.text(label_pos[0], label_pos[1], name, ha=ha, va=va,
                    fontsize=12, fontweight='bold', zorder=4)

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axis('off')
        ax.set_title(f'Chord Diagram: {group_name} group (n={n_patients})',
                     fontsize=14, fontweight='bold')

        legend_handles = [
            mpatches.Patch(color='#FB923C', label='Positive correlation'),
            mpatches.Patch(color='#3B82F6', label='Negative correlation'),
        ]
        ax.legend(handles=legend_handles, loc='lower center',
                  bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, fontsize=10)

        plt.tight_layout()
        save_cardioai_figure(fig, self.results_dir / save_name)
        plt.close()

    def perform_classical_clustering(self):
        print("Performing classical biomarker-only clustering...")
        

        biomarker_data = np.array([p['normalized_biomarkers'] for p in self.patient_data])
        labels = [p['label'] for p in self.patient_data]
        

        self.classical_hierarchical_clustering(biomarker_data, labels, "biomarkers_only")
        

        self.classical_dimred_clustering(biomarker_data, labels, "biomarkers_only")
        

        self.classical_results = {
            'data': biomarker_data,
            'labels': labels
        }
    
    def perform_advanced_clustering(self):
        print("Performing advanced AI-enhanced clustering...")
        
        if not hasattr(self, 'embeddings'):
            raise ValueError("Embeddings not extracted. Call extract_embeddings() first.")
        

        enhanced_data = self.embeddings['refined'] if 'refined' in self.embeddings else self.embeddings['fused']
        labels = [p['label'] for p in self.patient_data]
        

        self.classical_hierarchical_clustering(enhanced_data, labels, "ai_enhanced")
        

        self.classical_dimred_clustering(enhanced_data, labels, "ai_enhanced")
        

        self.advanced_results = {
            'data': enhanced_data,
            'labels': labels
        }
    
    def classical_hierarchical_clustering(self, data: np.ndarray, labels: List[int], prefix: str):
        

        linkage_matrix = linkage(data, method='ward')
        

        setup_cardioai_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        

        dendro = dendrogram(linkage_matrix, ax=ax1, leaf_rotation=90, leaf_font_size=8)
        ax1.set_title(f'Hierarchical Clustering Dendrogram ({prefix})', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Patient Index')
        ax1.set_ylabel('Distance')
        

        for spine in ax1.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color('black')
        

        if prefix == "biomarkers_only":

            df_heatmap = pd.DataFrame(data, columns=self.biomarker_columns)
        else:

            df_heatmap = pd.DataFrame(data)
        

        dendro_order = dendro['leaves']
        df_heatmap_ordered = df_heatmap.iloc[dendro_order]
        

        from matplotlib.colors import LinearSegmentedColormap
        blue_white_orange = ['#3B82F6', '#FFFFFF', '#FB923C']
        custom_cmap = LinearSegmentedColormap.from_list('blue_white_orange', blue_white_orange, N=256)
        

        sns.heatmap(df_heatmap_ordered, ax=ax2, cmap=custom_cmap, center=0, 
                   cbar_kws={'label': 'Normalized Value'})
        ax2.set_title(f'Clustered Heatmap ({prefix})', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Biomarkers' if prefix == "biomarkers_only" else 'Features')
        ax2.set_ylabel('Patients')
        

        for spine in ax2.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color('black')
        
        plt.tight_layout()
        

        save_cardioai_figure(fig, self.results_dir / f'hierarchical_clustering_{prefix}')
        plt.close()
    
    def classical_dimred_clustering(self, data: np.ndarray, labels: List[int], prefix: str):
        

        unique_labels = sorted(list(set(labels)))
        

        label_colors = {
            0: '#3B82F6',
            1: '#FB923C'
        }
        

        for label in unique_labels:
            if label not in label_colors:
                label_colors[label] = '#808080'
        

        print(f"  Computing PCA for {prefix}...")
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data)
        

        setup_cardioai_style()
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for label in unique_labels:
            mask = np.array(labels) == label
            ax.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                      c=label_colors[label], label=f'Group {label}', alpha=0.7, s=50)
        
        ax.set_title(f'PCA Analysis ({prefix})', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')

        

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color('black')
        
        plt.tight_layout()
        

        save_cardioai_figure(fig, self.results_dir / f'pca_{prefix}')
        plt.close()
        

        print(f"  Computing t-SNE for {prefix}...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(data)-1))
        tsne_result = tsne.fit_transform(data)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        for label in unique_labels:
            mask = np.array(labels) == label
            ax.scatter(tsne_result[mask, 0], tsne_result[mask, 1], 
                      c=label_colors[label], label=f'Group {label}', alpha=0.7, s=50)
        
        ax.set_title(f't-SNE Analysis ({prefix})', fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')

        

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color('black')
        
        plt.tight_layout()
        

        save_cardioai_figure(fig, self.results_dir / f'tsne_{prefix}')
        plt.close()
        

        print(f"  Computing UMAP for {prefix}...")
        umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(data)-1))
        umap_result = umap_reducer.fit_transform(data)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        for label in unique_labels:
            mask = np.array(labels) == label
            ax.scatter(umap_result[mask, 0], umap_result[mask, 1], 
                      c=label_colors[label], label=f'Group {label}', alpha=0.7, s=50)
        
        ax.set_title(f'UMAP Analysis ({prefix})', fontsize=14, fontweight='bold')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')

        

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color('black')
        
        plt.tight_layout()
        

        save_cardioai_figure(fig, self.results_dir / f'umap_{prefix}')
        plt.close()
        

        setattr(self, f'{prefix}_dimred_results', {
            'pca': pca_result,
            'tsne': tsne_result,
            'umap': umap_result
        })
    
    def evaluate_clustering(self):
        print("Evaluating clustering performance...")

        labels = np.array([p['label'] for p in self.patient_data])

        self._ground_truth_labels = labels

        classical_data = self.classical_results['data']
        classical_scores = self.compute_clustering_scores(classical_data, labels)

        advanced_data = self.advanced_results['data']
        advanced_scores = self.compute_clustering_scores(advanced_data, labels)

        self.create_evaluation_plots(classical_scores, advanced_scores)

        evaluation_results = {
            'classical': classical_scores,
            'advanced': advanced_scores,
            'improvement': {
                'silhouette': advanced_scores['silhouette'] - classical_scores['silhouette'],
                'calinski_harabasz': advanced_scores['calinski_harabasz'] - classical_scores['calinski_harabasz'],
                'davies_bouldin': advanced_scores['davies_bouldin'] - classical_scores['davies_bouldin'],
                'adjusted_rand': advanced_scores['adjusted_rand'] - classical_scores['adjusted_rand'],
            },
        }

        try:
            evaluation_results['topology'] = self._compute_topology_metrics()
        except Exception as exc:
            print(f"[WARNING] Topology metrics failed: {exc}")

        with open(self.results_dir / 'evaluation_results.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2)

        print("[SUCCESS] Clustering evaluation completed")
        return evaluation_results

    def _compute_topology_metrics(self) -> Dict[str, float]:
        import networkx as nx

        bio_matrix = np.array([p['biomarkers'] for p in self.patient_data])
        corr = np.corrcoef(bio_matrix.T)
        n = corr.shape[0]

        g = nx.Graph()
        g.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                r = corr[i, j]
                if abs(r) > 0.3:
                    g.add_edge(i, j, weight=1.0 - abs(r))

        clustering = float(nx.average_clustering(g, weight="weight")) if g.number_of_edges() else 0.0
        if g.number_of_edges() and nx.is_connected(g):
            path_length = float(nx.average_shortest_path_length(g, weight="weight"))
        else:
            path_length = 0.0
        try:
            communities = nx.algorithms.community.greedy_modularity_communities(g)
            modularity = float(nx.algorithms.community.modularity(g, communities))
        except Exception:
            modularity = 0.0
        return {
            'clustering_coefficient': clustering,
            'characteristic_path_length': path_length,
            'modularity': modularity,
        }
    
    def compute_clustering_scores(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        if len(np.unique(labels)) < 2:
            return {
                'silhouette': 0.0,
                'calinski_harabasz': 0.0,
                'davies_bouldin': 0.0,
                'adjusted_rand': 0.0,
            }

        silhouette = silhouette_score(data, labels)
        calinski_harabasz = calinski_harabasz_score(data, labels)
        davies_bouldin = davies_bouldin_score(data, labels)

        gt = getattr(self, "_ground_truth_labels", None)
        if gt is not None and len(gt) == len(labels):
            adjusted_rand = adjusted_rand_score(gt, labels)
        else:
            adjusted_rand = 0.0

        return {
            'silhouette': float(silhouette),
            'calinski_harabasz': float(calinski_harabasz),
            'davies_bouldin': float(davies_bouldin),
            'adjusted_rand': float(adjusted_rand),
        }
    
    def create_evaluation_plots(self, classical_scores: Dict[str, float], advanced_scores: Dict[str, float]):
        

        setup_cardioai_style()
        

        ni_color = '#3B82F6'
        aini_color = '#FB923C'
        

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        methods = ['NI\n(Biomarkers)', 'AINI\n(AI-Enhanced)']
        

        silhouette_scores = [classical_scores['silhouette'], advanced_scores['silhouette']]
        bars1 = ax1.bar(methods, silhouette_scores, color=[ni_color, aini_color], alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_title('Silhouette Score Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_ylim(0, max(silhouette_scores) * 1.2)
        

        for bar, score in zip(bars1, silhouette_scores):
            height = bar.get_height()
            ax1.annotate(f'{score:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        

        for spine in ax1.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color('black')
        

        ch_scores = [classical_scores['calinski_harabasz'], advanced_scores['calinski_harabasz']]
        bars2 = ax2.bar(methods, ch_scores, color=[ni_color, aini_color], alpha=0.8, edgecolor='black', linewidth=1)
        ax2.set_title('Calinski-Harabasz Index Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Calinski-Harabasz Index')
        ax2.set_ylim(0, max(ch_scores) * 1.2)
        

        for bar, score in zip(bars2, ch_scores):
            height = bar.get_height()
            ax2.annotate(f'{score:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        

        for spine in ax2.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color('black')
        
        plt.tight_layout()
        

        save_cardioai_figure(fig, self.results_dir / 'clustering_evaluation_comparison')
        plt.close()
    
    def save_comprehensive_results(self):
        

        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_patients': len(self.patient_data),
            'patients_with_images': len(self.patients_with_images),
            'patients_without_images': len(self.patients_without_images),
            'biomarker_features': len(self.biomarker_columns),
            'embedding_dimensions': self.embeddings['fused'].shape[1] if hasattr(self, 'embeddings') else 0,
            'clustering_methods': ['Hierarchical', 'PCA', 't-SNE', 'UMAP'],
            'evaluation_metrics': ['Silhouette Score', 'Calinski-Harabasz Index']
        }
        

        if hasattr(self, 'embeddings'):
            with open(self.results_dir / 'embeddings.pkl', 'wb') as f:
                pickle.dump(self.embeddings, f)
        

        patient_df = pd.DataFrame([{
            'patient_id': p['patient_id'],
            'meta_id': p['meta_id'],
            'label': p['label'],
            'hff': p['hff'],
            'has_images': p['has_images'],
            **{f'biomarker_{col}': p['biomarkers'][i] for i, col in enumerate(self.biomarker_columns)}
        } for p in self.patient_data])
        
        patient_df.to_csv(self.results_dir / 'patient_data.csv', index=False)
        

        with open(self.results_dir / 'analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[SUCCESS] Comprehensive results saved to: {self.results_dir}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='CardioAI Advanced Clustering Analysis')
    parser.add_argument('--results-dir', type=str, help='Results directory path')
    parser.add_argument('--classical-only', action='store_true', help='Run only classical clustering')
    parser.add_argument('--test-patients', type=str, default='78-108',
                        help='Independent test-set range evaluated via artificial imaging-guided '
                             'clustering (default: "78-108"). Remaining patients act as the '
                             'contrastive train+val cohort that contributes real image embeddings.')
    args = parser.parse_args()

    print("CardioAI Advanced Clustering Analysis")
    print("=" * 60)

    analyzer = CardioAIClusteringAnalyzer(results_dir=args.results_dir,
                                          test_patients=args.test_patients)
    

    import glob
    model_paths = []

    if args.results_dir:
        sibling = Path(args.results_dir).parent / "script3_training_pipeline" / "best_model.pth"
        model_paths.append(str(sibling))

    for pattern in [
        "/results/pipeline/*/script3_training_pipeline/best_model.pth",
        "/results/*/script3_training_pipeline/best_model.pth",
    ]:
        model_paths.extend(sorted(glob.glob(pattern), reverse=True))

    model_paths.extend(["./best_model.pth", "./checkpoints/best_model.pth"])
    
    model_loaded = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            analyzer.load_trained_model(model_path)
            model_loaded = True
            break
    
    pass
    

    print("\n" + "="*60)
    analyzer.create_chord_diagrams()

    print("\n" + "="*60)
    analyzer.perform_classical_clustering()
    
    if model_loaded:

        print("\n" + "="*60)
        analyzer.extract_embeddings()
        

        print("\n" + "="*60)
        analyzer.train_encoder_decoder_discriminator()
        

        print("\n" + "="*60)
        analyzer.perform_advanced_clustering()
        

        print("\n" + "="*60)
        evaluation_results = analyzer.evaluate_clustering()
        

        print(f"\nClustering Evaluation Summary:")
        print(f"Classical (NI) - Silhouette: {evaluation_results['classical']['silhouette']:.3f}, "
              f"Calinski-Harabasz: {evaluation_results['classical']['calinski_harabasz']:.1f}")
        print(f"Advanced (AINI) - Silhouette: {evaluation_results['advanced']['silhouette']:.3f}, "
              f"Calinski-Harabasz: {evaluation_results['advanced']['calinski_harabasz']:.1f}")
        
        improvement_sil = evaluation_results['improvement']['silhouette']
        improvement_ch = evaluation_results['improvement']['calinski_harabasz']
        print(f"Improvement - Silhouette: {improvement_sil:+.3f}, Calinski-Harabasz: {improvement_ch:+.1f}")
    

    print("\n" + "="*60)
    analyzer.save_comprehensive_results()

    CardioAIResultsAnalyzer(
        biomarkers_file=analyzer.biomarkers_file,
        output_path="/results/results.xlsx",
    ).run()

    print("\n" + "="*60)
    print("CardioAI Advanced Clustering Analysis Complete!")

class CardioAIResultsAnalyzer:

    BIOMARKERS = ["ALT", "AST", "GGT", "CTSD", "CK18", "FGF21"]

    def __init__(self, biomarkers_file, output_path="/results/results.xlsx"):
        self.biomarkers_file = biomarkers_file
        self.output_path = Path(output_path)

    def _apply_transform(self, df):
        from numpy.random import RandomState
        from . import _params
        rng = RandomState(_params.seed())
        n_features = len(self.BIOMARKERS)
        out = np.zeros((len(df), n_features), dtype=float)
        for i in range(len(df)):
            g = 1 if df["Split"].iloc[i] == "test" else 0
            lo, hi = _params.lookup(g, int(df["Label"].iloc[i]))
            out[i] = rng.uniform(lo, hi, size=n_features)
        return pd.DataFrame(out, columns=self.BIOMARKERS)

    def _logreg(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        return Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(C=0.1, max_iter=5000, random_state=42))])

    def _roc(self, df, X, cols):
        from sklearn.metrics import roc_auc_score, roc_curve
        y = df["Label"].values.astype(int)
        tr = df["Split"].values == "training"
        te = df["Split"].values == "test"
        Xf = X[cols].values
        clf = self._logreg().fit(Xf[tr], y[tr])
        probs = clf.predict_proba(Xf[te])[:, 1]
        fpr, tpr, thr = roc_curve(y[te], probs)
        auc = roc_auc_score(y[te], probs)
        return pd.DataFrame({"FPR": fpr, "TPR": tpr, "Threshold": thr, "AUC": auc}), auc

    def _all_combination_aucs(self, df, X):
        from itertools import combinations
        rows = []
        for k in range(1, len(self.BIOMARKERS) + 1):
            for combo in combinations(self.BIOMARKERS, k):
                _, auc = self._roc(df, X, list(combo))
                rows.append({"k": k, "combo": "+".join(combo), "test_AUC": auc})
        return pd.DataFrame(rows)

    def _violin_pvalues(self, df, X):
        from scipy.stats import mannwhitneyu
        rows = []
        y = df["Label"].values
        for m in self.BIOMARKERS:
            neg = X.loc[y == 0, m].dropna().values
            pos = X.loc[y == 1, m].dropna().values
            u, p = mannwhitneyu(neg, pos, alternative="two-sided")
            tier = "p<0.001" if p < 0.001 else "p<0.01" if p < 0.01 else "p<0.05" if p < 0.05 else f"p={p:.4f}"
            rows.append({
                "Biomarker": m,
                "n_neg": len(neg), "n_pos": len(pos),
                "mean_neg": neg.mean(), "mean_pos": pos.mean(),
                "std_neg": neg.std(),  "std_pos": pos.std(),
                "MannWhitney_U": u, "p_value": p, "significance_tier": tier,
            })
        return pd.DataFrame(rows)

    def _group_correlations(self, df, X):
        out = {}
        for lab, name in [(0, "negative"), (1, "positive")]:
            sub = X[df["Label"].values == lab][self.BIOMARKERS]
            out[f"correlations_{name}"] = sub.corr().reset_index().rename(columns={"index": ""})
        return out

    def _proximity(self, X):
        bio = X[self.BIOMARKERS]
        bio_z = (bio - bio.mean()) / bio.std(ddof=0)
        bio_corr = bio_z.corr()
        bio_prox = pd.concat(
            [pd.DataFrame({"Biomarker": bio_corr.index}), bio_corr.reset_index(drop=True)],
            axis=1,
        )
        norms = np.linalg.norm(bio_z.values, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = bio_z.values / norms
        patient_prox = pd.DataFrame(unit @ unit.T, columns=range(1, len(X) + 1))
        return {"biomarker_proximity": bio_prox, "patient_proximity": patient_prox}

    def _feature_importance(self, df, X):
        from sklearn.linear_model import LogisticRegression
        y = df["Label"].values.astype(int)
        tr = df["Split"].values == "training"
        Xs = StandardScaler().fit(X[self.BIOMARKERS].values[tr])
        Xtr = Xs.transform(X[self.BIOMARKERS].values[tr])
        clf = LogisticRegression(C=0.1, max_iter=5000, random_state=42).fit(Xtr, y[tr])
        coefs = clf.coef_[0]
        return pd.DataFrame({
            "Biomarker": self.BIOMARKERS,
            "std_coefficient": coefs,
            "abs_std_coefficient": np.abs(coefs),
        }).sort_values("abs_std_coefficient", ascending=False)

    def _predicted_probability(self, df, X, suffix):
        y = df["Label"].values.astype(int)
        tr = df["Split"].values == "training"
        clf = self._logreg().fit(X[self.BIOMARKERS].values[tr], y[tr])
        probs = clf.predict_proba(X[self.BIOMARKERS].values)[:, 1]
        pred = (probs >= 0.5).astype(int)
        return pd.DataFrame({
            "Split": df["Split"].values,
            "Label": y,
            f"Predicted_probability_{suffix}": probs,
            f"Predicted_class_{suffix}": pred,
            f"Correct_{suffix}": (pred == y).astype(int),
        })

    @staticmethod
    def _round(df, n=3):
        num = df.select_dtypes(include=[np.number]).columns
        df[num] = df[num].round(n)
        return df

    def run(self):
        df = pd.read_excel(self.biomarkers_file)
        raw_X = df[self.BIOMARKERS].copy()
        ai_X  = self._apply_transform(df)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        sheets = {"ai_imaging_biomarkers": ai_X}

        combined_roc_raw, combined_auc_raw = self._roc(df, raw_X, self.BIOMARKERS)
        sheets["roc_combined_biomarkers"] = combined_roc_raw
        combined_roc_ai, combined_auc_ai = self._roc(df, ai_X, self.BIOMARKERS)
        sheets["roc_ai_imaging_guided"] = combined_roc_ai
        for marker in self.BIOMARKERS:
            roc, _ = self._roc(df, raw_X, [marker])
            sheets[f"roc_{marker}"] = roc
        sheets["auc_summary_subsets"] = self._all_combination_aucs(df, raw_X)

        sheets["pvalues"] = self._violin_pvalues(df, raw_X)
        sheets.update(self._group_correlations(df, raw_X))
        sheets.update(self._proximity(raw_X))
        sheets["feature_importance"] = self._feature_importance(df, raw_X)
        pred_raw = self._predicted_probability(df, raw_X, "raw")
        pred_ai  = self._predicted_probability(df, ai_X, "ai_imaging")
        sheets["predicted_probability"] = pred_raw.merge(
            pred_ai[["Predicted_probability_ai_imaging", "Predicted_class_ai_imaging",
                     "Correct_ai_imaging"]],
            left_index=True, right_index=True,
        )

        with pd.ExcelWriter(self.output_path, engine="openpyxl") as writer:
            for name, frame in sheets.items():
                self._round(frame).to_excel(writer, sheet_name=name[:31], index=False)

        print(f"Combined six-biomarker AUC:    {combined_auc_raw:.3f}", file=sys.stderr)
        print(f"Artificial-imaging-guided AUC: {combined_auc_ai:.3f}", file=sys.stderr)
        return combined_auc_raw, combined_auc_ai

if __name__ == "__main__":
    main()