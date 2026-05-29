# Artificial imaging–guided immunosensor from a metal–organic framework for multiplex biomarker detection in steatotic liver disease

---

## Repository layout

```
.
├── main_cardioAI.py              # pipeline orchestrator (CLI)
├── README.md
├── LICENSE
│
├── models/                       # Architecture panels b, d, e, f, g
│   ├── attention.py              # MultiHeadSelfAttention (+ 3D relative position bias)
│   ├── transformer_block.py      # TransformerBlock (FFN hidden = 1024)
│   ├── patch_modules.py          # PatchEmbed3D, PatchMerging3D
│   ├── biomarker_encoder.py      # BiomarkerEncoder (panel d)
│   ├── image_encoder.py          # ImageEncoder (panel e, Swin-style 3D)
│   ├── cross_modal_fusion.py     # CrossModalFusion (panel g)
│   ├── contrastive_loss.py       # InfoNCE + hard-negative + retrieval MSE
│   ├── contrastive_decoder.py    # biomarker-embedding -> MRI voxel decoder
│   ├── projection_head.py        # shared projection-head pattern
│   ├── similarity.py             # cosine / temperature / top-k retrieval
│   └── cardio_model.py           # CardioAIModel + create_model (panel f)
│
├── data/                         # data ingestion
│   ├── pt_dataset.py             # CardioAIDataset + custom_collate_fn
│   ├── dicom_processor.py        # DICOM -> (4, 32, 96, 96) .pt tensor pipeline
│   └── liver_segmentation.py     # MRI liver segmentation helpers
│
├── training/                     # Architecture panels b, c
│   └── trainer.py                # InfoNCE trainer with warm-up + annealing
│
├── retrieval/                    # Architecture panel f
│   └── biomarker_to_image.py     # retrieval helpers + MRR / top-k evaluator
│
├── clustering/                   # Architecture panels f-h + Figure 4 compute
│   ├── pipeline.py               # CardioAIClusteringAnalyzer aggregate
│   ├── edd_encoder.py            # EDD encoder
│   ├── edd_decoder.py            # EDD decoder
│   ├── edd_discriminator.py      # real-vs-reconstructed discriminator
│   ├── edd_classifier.py         # binary MASLD classifier head
│   ├── edd_model.py              # aggregate EDD module
│   ├── edd_trainer.py            # joint EDD + classifier training loop
│   ├── chord_diagrams.py         # chord-diagram compute
│   ├── classical_clustering.py   # biomarker-only clustering
│   ├── advanced_clustering.py    # artificial-imaging-guided clustering
│   └── clustering_metrics.py     # silhouette / CH / Davies-Bouldin / ARI
│
└── utils/                        # shared plotting / palette / saving helpers
    └── styling.py
```

---

## Quick start

### Requirements

Python 3.10+ with:

```
torch >= 2.0
numpy pandas matplotlib seaborn scipy scikit-learn
umap-learn networkx shap
pydicom nibabel opencv-python tqdm
Pillow openpyxl
```

### Running the pipeline

```bash
# Stages 1-4: dataset preparation, model check, contrastive training,
# retrieval + clustering + EDD classifier. Stage 5 (paper figures) is
# shipped with the manuscript and is skipped here.
python main_cardioAI.py --skip-figures --epochs 200 --batch-size 4

# Skip training and reuse existing weights
python main_cardioAI.py --skip-figures --skip-training \
    --weights <path/to/script3_training_pipeline>

# Run only the clustering analysis stage
python main_cardioAI.py --only-clustering

# Dry run: show the execution plan without running
python main_cardioAI.py --dry-run
```

Per-stage entry points (each runs independently):

```bash
python -m data.dicom_processor            # Stage 1: DICOM -> .pt tensors
python -m training.trainer                # Stage 3: contrastive training
python -m clustering.pipeline             # Stage 4: clustering + EDD + classifier
```

### Hardware

Default hyperparameters target a single consumer GPU with 8 GB of
memory; scale `--batch-size` up for larger cards.

---

## License

MIT. See [`LICENSE`](LICENSE).

---

## Citation

If you use this code, please cite the accompanying manuscript
(citation to be added on publication).
