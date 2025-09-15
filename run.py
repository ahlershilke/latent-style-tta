#!/usr/bin/env python3
"""
One-shot pipeline:
1) Hyperparameter tuning (LODO)
2) Training with tuned/global config
3) (Optional) TTA evaluation using saved models and style statistics

Example:
  python run_tune_train.py \
    --data_root /data/ \
    --dataset_name PACS \
    --num_trials 40 \
    --num_epochs 60 \
    --domains 4 \
    --use_mixstyle false \
    --run_tta \
    --tta_modes single_0 selective_0_1 \
    --tta_output_dir experiments/test_results/pacs_run
"""

import argparse
import os
import sys
import shlex
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

def bool_arg(v: str) -> bool:
    return str(v).lower() in {"1", "true", "yes", "y", "t"}

def run(cmd: str, cwd: Path = REPO_ROOT) -> None:
    print(f"\n>>> {cmd}\n")
    subprocess.run(shlex.split(cmd), cwd=str(cwd), check=True)

def main():
    parser = argparse.ArgumentParser(description="Tune -> Train -> (optional) TTA")
    # tune/train
    parser.add_argument("--data_root", required=True, type=str, help="Parent dir that contains PACS/ or VLCS/")
    parser.add_argument("--dataset_name", default="PACS", choices=["PACS", "VLCS"])
    parser.add_argument("--num_trials", type=int, default=30)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--domains", type=int, default=4)
    parser.add_argument("--use_mixstyle", type=bool_arg, default=False)
    parser.add_argument("--extract_stats", type=bool_arg, default=True, help="Should style stats be extracted after training (True/False).")
    parser.add_argument("--hparam_file", type=str, default=None)
    parser.add_argument("--skip_tuning", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 7, 0])

    # TTA
    parser.add_argument("--run_tta", action="store_true", help="Run TTA after training")
    parser.add_argument("--tta_output_dir", type=str, default="experiments/test_results")
    parser.add_argument("--tta_num_classes", type=int, default=None, help="Override class count (PACS 7, VLCS 5)")
    parser.add_argument("--tta_random_trials", type=int, default=1000)
    parser.add_argument("--tta_verbose", action="store_true")

    args = parser.parse_args()
    py = shlex.quote(sys.executable)

    # 1) TUNING
    if not args.skip_tuning:
        cmd_tune = (
            f'{py} {REPO_ROOT / "tune.py"} '
            f'--data_root {shlex.quote(args.data_root)} '
            f'--dataset_name {shlex.quote(args.dataset_name)} '
            f'--num_trials {args.num_trials}'
        )
        run(cmd_tune)

    # 2) Resolve hyperparameter YAML for training
    hparam_file = args.hparam_file
    if hparam_file is None:
        # tuner writes here
        generated = REPO_ROOT / "configs" / "global_config.yaml"
        if generated.exists():
            hparam_file = str(generated)
        else:
            # fallback to repo default if present
            default_path = REPO_ROOT / "configs" / args.dataset_name.lower() / "global_config.yaml"
            if default_path.exists():
                hparam_file = str(default_path)
            else:
                raise FileNotFoundError(
                    "No global_config.yaml found. Either run tuning first or pass --hparam_file."
                )

    # 3) TRAINING
    if not args.skip_training:
        mixstyle_flag = "true" if args.use_mixstyle else "false"
        extract_flag = "true" if args.extract_stats else "false"
        cmd_train = (
            f'{py} {REPO_ROOT / "train.py"} '
            f'--data_root {shlex.quote(args.data_root)} '
            f'--dataset_name {shlex.quote(args.dataset_name)} '
            f'--hparam_file {shlex.quote(hparam_file)} '
            f'--num_epochs {args.num_epochs} '
            f'--domains {args.domains} '
            f'--use_mixstyle {mixstyle_flag} '
            f'--extract_stats {extract_flag}'
        )
        run(cmd_train)

    # 4) TTA (optional)
    if args.run_tta:
        # models_root_path expected by _tta.py -> points to "experiments/train_results/saved_models"
        models_root = REPO_ROOT / "experiments" / "train_results" / "saved_models"
        if not models_root.exists():
            raise FileNotFoundError(
                f"TTA requested, but models root not found: {models_root}.\n"
                "Make sure training completed and saved checkpoints under seed_* folders."
            )

        # dataset-specific default num_classes
        if args.tta_num_classes is None:
            if args.dataset_name.upper() == "PACS":
                num_classes = 7
            elif args.dataset_name.upper() == "VLCS":
                num_classes = 5
            else:
                raise ValueError("Please set --tta_num_classes explicitly.")
        else:
            num_classes = args.tta_num_classes

        modes_str = " ".join(args.tta_modes)
        seeds_str = " ".join(map(str, args.seeds))
        cmd_tta = (
            f'{py} -m models._tta '
            f'--models_root_path {shlex.quote(str(models_root))} '
            f'--data_dir {shlex.quote(args.data_root)} '
            f'--dataset {shlex.quote(args.dataset_name)} '
            f'--num_classes {num_classes} '
            f'--modes {modes_str} '
            f'--seeds {seeds_str} '
            f'--output_dir {shlex.quote(args.tta_output_dir)} '
            f'--random_trials {args.tta_random_trials} '
            f'{"--verbose" if args.tta_verbose else ""}'
        )
        run(cmd_tta)

    # 5) Summary
    print("\n=== Pipeline complete ===")
    print(f"- Tuning results: {REPO_ROOT/'experiments'/'hp_results'}")
    print(f"- Global config:  {hparam_file}")
    print(f"- Training out:   {REPO_ROOT/'experiments'/'train_results'}")
    if args.run_tta:
        print(f"- TTA results:    {args.tta_output_dir}")

if __name__ == "__main__":
    main()
