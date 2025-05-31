import argparse
import os
import random
import numpy as np
import torch
import json
from typing import List, Dict
from models.tuning import HP_Tuner
from data._load_data import get_lodo_splits
from data._datasets import DOMAIN_NAMES
from utils._utils import analyze_and_visualize_studies
from torch.utils.data import DataLoader

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    use_mixstyle = None
    all_splits = get_lodo_splits()
    results: List[Dict] = []
    all_studies = []
    os.makedirs("experiments/hp_results", exist_ok=True)
    
    for fold_idx, (train_data, val_data, test_data) in enumerate(all_splits):
        domain_name = DOMAIN_NAMES['PACS'][fold_idx]
        print(f"\n=== Fold {fold_idx} ({domain_name}) ===")

        save_dir = f"experiments/hp_results/{fold_idx}"
        os.makedirs(save_dir, exist_ok=True)

        tuner = HP_Tuner(
            train_data=train_data,
            val_data=val_data,
            num_classes=7,
            num_domains=len(DOMAIN_NAMES['PACS']) - 1,  # N-1 Domains im Training
            n_trials=args.num_trials,
            save_dir=save_dir,
            fold_info={'fold': fold_idx, 'test_domain': domain_name}
        )

        study = tuner.run()
        all_studies.append(study)

        use_mixstyle = any('mixstyle' in study.best_params for study in all_studies)
        
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        test_metrics = tuner.evaluate_model(
            model=tuner.load_best_model(study),
            test_loader=test_loader,
        )
        
        results.append({
            'test_domain': domain_name,
            'accuracy': test_metrics['accuracy'],
            'loss': test_metrics['loss'],
            'best_params': study.best_params,
            'trials': len(study.trials)
        })

        with open(f"{save_dir}/interim_results_fold_{fold_idx}.json", "w") as f:
            json.dump(results[-1], f, indent=2)

    with open("experiments/hp_results/final_results.json", "w") as f:
        json.dump({
            'per_domain': results,
            'global_best': HP_Tuner.compute_global_best_params("experiments/hp_results", use_mixstyle),
            'config': {
                'num_trials': args.num_trials,
                'batch_size': study.best_params['batch_size'],
                'learning_rate': study.best_params['lr'],
                'weight_decay': study.best_params['weight_decay'],
                'dropout': study.best_params['dropout'],
                'model': 'resnet50',
                'optimizer': study.best_params['optimizer'],
                'scheduler': study.best_params['scheduler'],
                'loss_function': 'CrossEntropyLoss',
                'domains': DOMAIN_NAMES['PACS']
            }
        }, f, indent=2)

    if any(len(study.trials) > 1 for study in all_studies):
        analyze_and_visualize_studies(all_studies, "experiments/hp_results", use_mixstyle)
    else:
        print("Skipping visualizations - not enough trials for analysis")
    
    print("=== All trials completed ===")


if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trials', type=int, default=30,
                       help='Number of trials for hyperparameter tuning.')
    args = parser.parse_args()
    main()