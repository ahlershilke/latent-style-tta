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
from utils._utils import analyze_and_visualize_studies, save_tuning_results
from torch.utils.data import DataLoader

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # 1. Initial LODO tuning for each domain
    print("=== Phase 1: Per-domain hyperparameter tuning ===")
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
            num_domains=len(DOMAIN_NAMES['PACS']) - 1,
            n_trials=args.num_trials,
            save_dir=save_dir,
            fold_info={'fold': fold_idx, 'test_domain': domain_name}
        )

        study = tuner.run()
        all_studies.append(study)
        
        # Evaluate on test domain
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

    # 2. Global parameter optimization
    print("\n=== Phase 2: Global hyperparameter optimization ===")
    use_mixstyle = any('mixstyle' in study.best_params for study in all_studies)

    global_tuner = HP_Tuner(
        train_data=None,
        val_data=None,
        num_classes=7,
        num_domains=len(DOMAIN_NAMES['PACS']) - 1,
        n_trials=0,
        save_dir="experiments/hp_results/global",
        fold_info={'fold': -1, 'test_domain': 'global'}
    )
    
    # Analyze LODO results to find global optimal parameters
    global_results = global_tuner.compute_global_best_params(
        "experiments/hp_results", 
        #use_mixstyle=use_mixstyle
        use_mixstyle=False
    )
    
    # Further refine global parameters
    final_params = global_tuner.run_global_tuning(
        lodo_results_dir="experiments/hp_results",
        n_trials=args.num_trials
    )
    
    # 3. Evaluate global parameters on all domains
    print("\n=== Phase 3: Evaluating global parameters ===")
    global_eval_results = []
    for fold_idx, (_, _, test_data) in enumerate(all_splits):
        domain_name = DOMAIN_NAMES['PACS'][fold_idx]
        print(f"Evaluating on domain: {domain_name}")
        
        eval_tuner = HP_Tuner(
            train_data=None,  # only evaluation
            val_data=None,
            num_classes=7,
            num_domains=len(DOMAIN_NAMES['PACS']) - 1,
            n_trials=0,  # No tuning
            save_dir=f"experiments/hp_results/{fold_idx}",
            fold_info={'fold': fold_idx, 'test_domain': domain_name}
        )
        
        # Create model with global parameters
        model = eval_tuner.create_model(final_params)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        test_metrics = eval_tuner.evaluate_model(model, test_loader)
        
        global_eval_results.append({
            'domain': domain_name,
            'accuracy': test_metrics['accuracy'],
            'loss': test_metrics['loss']
        })

    # 4. Save final results
    final_output = {
        'lodo results': global_results,
        'per_domain_tuning': results,
        'global_parameters': final_params,
        'global_evaluation': global_eval_results,
        'global_stats': {
            'mean_accuracy': np.mean([x['accuracy'] for x in global_eval_results]),
            'std_accuracy': np.std([x['accuracy'] for x in global_eval_results]),
            'worst_domain': min(global_eval_results, key=lambda x: x['accuracy'])['domain'],
            'best_domain': max(global_eval_results, key=lambda x: x['accuracy'])['domain']
        }
    }

    with open("experiments/hp_results/final_global_results.json", "w") as f:
        json.dump(final_output, f, indent=2)

    print("\n=== Final Results ===")
    print(f"Global parameters: {final_params}")
    print(f"Mean accuracy across domains: {final_output['global_stats']['mean_accuracy']:.2%}")
    print(f"Standard deviation: {final_output['global_stats']['std_accuracy']:.4f}")
    print(f"Worst performing domain: {final_output['global_stats']['worst_domain']}")
    print(f"Best performing domain: {final_output['global_stats']['best_domain']}")

    if any(len(study.trials) > 1 for study in all_studies):
        analyze_and_visualize_studies(all_studies, "experiments/hp_results", use_mixstyle)
    else:
        print("Skipping visualizations - not enough trials for analysis")
    
    save_tuning_results(
        config_dir="configs",
        results_dir="experiments/hp_results"
    )

    print("=== All trials completed ===")


if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trials', type=int, default=30,
                       help='Number of trials for hyperparameter tuning.')
    args = parser.parse_args()
    main()