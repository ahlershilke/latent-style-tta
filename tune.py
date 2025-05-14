import argparse
import os
import random
import numpy as np
import torch
from models.tuning import HP_Tuner 
from data._load_data import get_train_val_datasets

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    train_data, val_data, test_data = get_train_val_datasets()
    num_classes = 7
    num_domains = 3 # photo is test domain
    
    tuner = HP_Tuner(
        train_data=train_data,
        val_data=val_data,
        num_classes=num_classes,
        num_domains=num_domains,
        n_trials=args.num_trials
    )
    
    save_dir = "experiments"
    os.makedirs(save_dir, exist_ok=True)

    study = tuner.run(save_dir=save_dir)
    
    print("\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    
    print(f"\nBest Accuracy: {study.best_value:.4f}")

if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trials', type=int, default=50,
                       help='Number of trials for hyperparameter tuning.')
    args = parser.parse_args()
    
    main()