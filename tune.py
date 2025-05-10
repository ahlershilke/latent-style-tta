import argparse
import os
from models.tuning import HP_Tuner 
from data._load_data import get_train_val_datasets

def main():
    train_data, val_data = get_train_val_datasets()
    num_classes = 7
    num_domains = 4
    
    tuner = HP_Tuner(
        train_data=train_data,
        val_data=val_data,
        num_classes=num_classes,
        num_domains=num_domains,
        n_trials=args.num_trials
    )
    
    save_dir = "experiments/hp_results"
    os.makedirs(save_dir, exist_ok=True)

    study = tuner.run(save_dir=save_dir)
    
    print("\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    
    print(f"\nBest Accuracy: {study.best_value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trials', type=int, default=50,
                       help='Number of trials for hyperparameter tuning.')
    args = parser.parse_args()
    
    main()