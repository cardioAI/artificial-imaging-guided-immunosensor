# Artificial Imaging-Guided Nanoengineered Immunosensor for MASLD Detection

Reference implementation of the contrastive learning framework for multi-biomarker MASLD detection guided by artificial liver-MRI imaging. The pipeline pairs a six-biomarker plasma panel with 3D MRI volumes, trains a cross-modal retrieval model (CLIP-style), and classifies MASLD status using the resulting fused embeddings.

> Metabolic dysfunction-associated steatotic liver disease (MASLD) affects roughly 25% of the global population. MASLD is defined here as a proton density fat fraction (PDFF) >= 5% computed from 3D MRI, consistent with established clinical criteria. In a cohort of 108 participants, unsupervised clustering of six plasma biomarkers correctly identified 98 MASLD cases (AUC 0.932). Aligning those biomarker profiles with paired liver MRI via contrastive learning -- 57 training + 20 validation participants, with 31 held out as an independent test set evaluated through artificial imaging-guided clustering -- produces sensor readouts that correctly classify 29 of 31 held-out individuals.

---

## Architecture

| Panel | Concern | Code |
|-------|---------|------|
| 1b | Biomarker-to-image similarity matrix + InfoNCE loss | `models/contrastive_loss.py`, `models/similarity.py` |
| 1c | Iterative contrastive training, Step 1 -> Step K | `training/trainer.py` (progression snapshots), `figures/contrastive_progression.py` (renderer) |
| 1d | Biomarker encoder, strict figure order: `MLP -> LN -> MHSA -> LN -> (+ MLP_out) -> Embedding` (post-norm residual) | `models/biomarker_encoder.py`, `models/attention.py` |
| 1e | Image encoder (Patch partition -> Linear embed -> Transformer -> Patch merging -> Transformer -> Embedding) | `models/image_encoder.py`, `models/patch_modules.py`, `models/transformer_block.py` |
| 1f | Bidirectional biomarker <-> image retrieval | `retrieval/biomarker_to_image.py`, `models/cardio_model.py` |
| 1g | Cross-modal fusion + clustering / grouping | `models/cross_modal_fusion.py`, `clustering/*` |
| 1h | Encoder -> Embedding -> Decoder -> Discriminator -> Binary MASLD classifier | `clustering/edd_encoder.py`, `clustering/edd_decoder.py`, `clustering/edd_discriminator.py`, `clustering/edd_classifier.py`, `clustering/edd_trainer.py` |

Figure 4 (chord diagrams, hierarchical / PCA / t-SNE / UMAP clustering, silhouette + Calinski-Harabasz metrics) and Figure 5 (violin plots, per-biomarker ROC, calibration, combined six-biomarker ROC, proximity matrices, feature importance) are implemented in `clustering/` and `figures/` respectively.

---

## Repository layout

```
.
├── main_cardioAI.py              # pipeline orchestrator (CLI)
├── prepare_cleaned.py            # one-shot DICOM cleaning helper
│
├── models/                       # Architecture panels b, d, e, g primitives
│   ├── attention.py              # MultiHeadSelfAttention
│   ├── transformer_block.py      # TransformerBlock
│   ├── patch_modules.py          # PatchEmbed3D, PatchMerging3D
│   ├── biomarker_encoder.py      # BiomarkerEncoder (Architecture panel d)
│   ├── image_encoder.py          # ImageEncoder (Architecture panel e, Swin-style 3D)
│   ├── cross_modal_fusion.py     # CrossModalFusion (Architecture panel g)
│   ├── contrastive_loss.py       # ContrastiveLoss (Architecture panel b)
│   ├── projection_head.py        # shared projection-head pattern
│   ├── similarity.py             # cosine / temperature-scaled sim, top-k retrieval
│   └── cardio_model.py           # CardioAIModel + create_model + retrieve (Architecture panel f)
│
├── data/                         # data ingestion
│   ├── pt_dataset.py             # CardioAIDataset + custom_collate_fn
│   ├── dicom_processor.py        # DICOM -> .pt tensor pipeline
│   └── liver_segmentation.py     # 6+6 shade MRI liver segmentation
│
├── training/                     # Architecture panel c
│   └── trainer.py                # CardioAITrainer + CLI
│
├── retrieval/                    # Architecture panel f
│   └── biomarker_to_image.py     # retrieval helpers
│
├── clustering/                   # Architecture panels f-h + Figure 4
│   ├── pipeline.py               # CardioAIClusteringAnalyzer aggregate
│   ├── edd_encoder.py            # EDD encoder half (Architecture panel h, left)
│   ├── edd_decoder.py            # EDD decoder half (Architecture panel h, right)
│   ├── edd_discriminator.py      # real-vs-reconstructed discriminator
│   ├── edd_classifier.py         # binary MASLD classifier head
│   ├── edd_model.py              # aggregate EDD module
│   ├── edd_trainer.py            # joint EDD + classifier training loop
│   ├── chord_diagrams.py         # Figure 4a, 4b
│   ├── classical_clustering.py   # Figure 4c, 4e-4g
│   ├── advanced_clustering.py    # Figure 4d, 4i-4k
│   └── clustering_metrics.py     # Figure 4h, 4l
│
├── figures/                      # Figure 1c + Figure 5 panels
│   ├── pipeline.py               # CardioAIFigureTableGenerator
│   ├── contrastive_progression.py# Figure 1c (Step 1 -> Step K)
│   ├── violin_plots.py           # Figure 5a-5f
│   ├── roc_analysis.py           # Figure 5h-5m, 5q, 5s
│   ├── calibration.py            # Figure 5r
│   ├── feature_importance.py     # Figure 5p
│   ├── proximity_matrices.py     # Figure 5n, 5o
│   ├── mri_slices.py             # 384 MRI PNGs (Patient_15263401)
│   ├── artificial_mri.py         # biomarker -> image retrieval visualisation
│   ├── embedding_heatmaps.py     # 7 embedding heatmaps
│   ├── artificial_mri_lowlevel.py
│   └── embedding_heatmaps_lowlevel.py
│
├── reports/                      # post-analysis outputs
│   └── excel_report.py           # multi-sheet Excel workbook
│
└── utils/                        # shared plotting / palette / figure saving
    └── styling.py
```

---

## Quick start

### Requirements

Python 3.10+ with:

```
torch >= 2.0
numpy pandas matplotlib seaborn scipy scikit-learn
umap-learn xlsxwriter openpyxl
pydicom nibabel opencv-python tqdm
Pillow
```

### Running the pipeline

```bash
# Full pipeline (dataset analysis -> model -> training -> clustering -> figures)
python main_cardioAI.py --epochs 200 --batch-size 8

# Skip training and reuse existing weights
python main_cardioAI.py --skip-training --weights <path/to/script3_training_pipeline>

# Only Figure 5 generation
python main_cardioAI.py --only-figures --artificial-images 384

# Dry run: show the execution plan without running
python main_cardioAI.py --dry-run
```

Per-stage entry points (can be run independently):

```bash
python -m data.dicom_processor            # Stage 1: DICOM -> .pt tensors
python -m training.trainer                # Stage 3: contrastive training
python -m clustering.pipeline             # Stage 4: clustering + EDD + classifier
python -m figures.pipeline                # Stage 5: Figure 5 + Excel
python -m figures.contrastive_progression # Figure 1c: Step 1 -> Step K
```

### Hardware

Default hyperparameters target a single RTX 2070S (8 GB). Adjust `--batch-size` for larger GPUs.

---

## Data layout

The code expects the following paths (configurable via CLI flags):

```
F:\datasets_cardioAI\BICL_cardioAI\
├── raw\                            # as-received DICOM + Anatomical masks
├── cleaned\                        # restructured Patient_<ID>/{dicom,other}
│   └── Pt\                         # preprocessed (4, 64, 192, 192) .pt tensors
└── results\{timestamp}\            # all pipeline outputs, organised per stage
    ├── script1_dataset_analysis\
    ├── script3_training_pipeline\
    │   └── progression\            # Figure 1c step_{k}.pt snapshots
    ├── script4_clustering_analysis\
    └── script5_figures_tables\
        └── figure1c_contrastive_progression\
```

Run `python prepare_cleaned.py` once to populate `cleaned/` from `raw/`; the dataset stage then writes `.pt` tensors to `cleaned/Pt/`.

Patient-level biomarker data (`dataset_cardioAI.xlsx`) is NOT included in this repository. The six-biomarker columns expected by the code are `AST`, `ALT`, `GGT`, `CTSD`, `CK18`, `FGF21`, plus a `Meta_ID`, `HFF` (hepatic fat fraction, in %), and binary `Label` column (derived as `HFF >= 5` per the MASLD/PDFF criterion).

### Cohort split

The 108-patient cohort is partitioned 57 / 20 / 31 by row order in `dataset_cardioAI.xlsx`:

| Cohort | Range (1-based) | Role | CLI flag |
|--------|------------------|------|----------|
| Training | 1-57 | contrastive cross-modal alignment | `--train-patients 1-57` |
| Validation | 58-77 | contrastive val + model selection | `--val-patients 58-77` |
| Independent test | 78-108 | artificial imaging-guided clustering (biomarkers routed through retrieval) | `--test-patients 78-108` |

---

## License

MIT. See [`LICENSE`](LICENSE).

---

## Citation

If you use this code, please cite the accompanying manuscript (citation to be added on publication).
