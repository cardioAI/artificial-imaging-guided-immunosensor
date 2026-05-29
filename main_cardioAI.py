
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict

warnings.filterwarnings("ignore")

for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CardioAI Complete Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main_cardioAI.py\n"
            "  python main_cardioAI.py --skip-training --weights ./weights\n"
            "  python main_cardioAI.py --only-figures --artificial-images 192\n"
        ),
    )

    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-train", dest="skip_training", action="store_true")
    parser.add_argument("--skip-clustering", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")

    parser.add_argument("--only-dataset", action="store_true")
    parser.add_argument("--only-model", action="store_true")
    parser.add_argument("--only-training", action="store_true")
    parser.add_argument("--only-clustering", action="store_true")
    parser.add_argument("--only-figures", action="store_true")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)

    parser.add_argument("--train-patients", type=str, default="1-57")
    parser.add_argument("--val-patients", type=str, default="58-77")
    parser.add_argument("--test-patients", type=str, default="78-108")
    parser.add_argument("--weights", type=str, default=None,
                        help="Directory containing pre-trained weights to reuse.")
    parser.add_argument("--artificial-images", type=int, default=384)

    parser.add_argument("--results-dir", type=str,
                        default=r"F:\datasets_cardioAI\BICL_cardioAI\results")
    parser.add_argument("--sample-mode", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args()

STAGE_ORDER = ["dataset", "model", "training", "clustering", "figures"]

def decide_enabled_stages(args: argparse.Namespace) -> Dict[str, bool]:
    stages = {s: True for s in STAGE_ORDER}

    if args.skip_dataset: stages["dataset"] = False
    if args.skip_model: stages["model"] = False
    if args.skip_training: stages["training"] = False
    if args.skip_clustering: stages["clustering"] = False
    if args.skip_figures: stages["figures"] = False

    any_only = any([args.only_dataset, args.only_model, args.only_training,
                    args.only_clustering, args.only_figures])
    if any_only:
        for k in stages:
            stages[k] = False
        if args.only_dataset: stages["dataset"] = True
        if args.only_model: stages["model"] = True
        if args.only_training: stages["training"] = True
        if args.only_clustering: stages["clustering"] = True
        if args.only_figures: stages["figures"] = True

    return stages

def run_pipeline(args: argparse.Namespace, stages: Dict[str, bool], base: Path) -> bool:
    ok = True

    if stages["dataset"]:
        print("\n" + "=" * 60)
        print("Stage 1 / 5: Dataset analysis & tensor preparation")
        try:
            from data.dicom_processor import main as dataset_main
            sys.argv = [sys.argv[0], "--results-dir",
                        str(base / "script1_dataset_analysis")]
            dataset_main()
        except SystemExit:
            pass
        except Exception as exc:
            ok = False
            print(f"[ERROR] Dataset stage failed: {exc}")

    if stages["model"]:
        print("\n" + "=" * 60)
        print("Stage 2 / 5: Model architecture sanity check")
        try:
            from models import create_model
            model = create_model()
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  CardioAIModel instantiated -- {n_params / 1e6:.2f} M parameters")
        except Exception as exc:
            ok = False
            print(f"[ERROR] Model stage failed: {exc}")

    if stages["training"]:
        print("\n" + "=" * 60)
        print("Stage 3 / 5: Contrastive training")
        try:
            from training.trainer import main as train_main
            try:
                train_main(num_epochs=args.epochs, batch_size=args.batch_size,
                           output_dir=str(base / "script3_training_pipeline"),
                           train_patients=args.train_patients,
                           val_patients=args.val_patients)
            except TypeError:
                train_main()
        except SystemExit:
            pass
        except Exception as exc:
            ok = False
            print(f"[ERROR] Training stage failed: {exc}")

    if stages["clustering"]:
        print("\n" + "=" * 60)
        print("Stage 4 / 5: Retrieval + clustering + EDD classifier")
        try:
            from clustering.pipeline import main as clustering_main
            sys.argv = [sys.argv[0],
                        "--results-dir", str(base / "script4_clustering_analysis"),
                        "--test-patients", args.test_patients]
            clustering_main()
        except SystemExit:
            pass
        except Exception as exc:
            ok = False
            print(f"[ERROR] Clustering stage failed: {exc}")

    if stages["figures"]:
        print("\n" + "=" * 60)
        print("Stage 5 / 5: post-processing")
        try:
            from figures.pipeline import main as figures_main
            sys.argv = [sys.argv[0],
                        "--results-dir", str(base / "script5_figures_tables"),
                        "--artificial-images", str(args.artificial_images)]
            if args.weights:
                sys.argv.extend(["--weights-dir", args.weights])
            figures_main()
        except SystemExit:
            pass
        except Exception as exc:
            ok = False
            print(f"[ERROR] Figures stage failed: {exc}")

    return ok

def main() -> int:
    args = parse_arguments()
    stages = decide_enabled_stages(args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(args.results_dir) / timestamp
    base.mkdir(parents=True, exist_ok=True)

    print("CardioAI Pipeline Execution Plan")
    print("=" * 40)
    print(f"Analysis mode: {'Sample/Demo' if args.sample_mode else 'Full'}")
    print(f"Epochs: {args.epochs}   Batch size: {args.batch_size}")
    print(f"Results base: {base}")
    print("Stages:")
    for stage in STAGE_ORDER:
        flag = "[ENABLED]" if stages[stage] else "[DISABLED]"
        print(f"  {stage:<12} {flag}")

    if args.dry_run:
        print("[DRY RUN] Not executing.")
        return 0

    success = run_pipeline(args, stages, base)
    print("\n" + "=" * 60)
    print("Pipeline execution", "complete." if success else "FINISHED WITH ERRORS.")
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
