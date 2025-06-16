import os
import json
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
import optuna
import optuna.visualization as vis
from optuna.trial import TrialState
from optuna.pruners import MedianPruner
from models import resnet18, resnet34, resnet50, resnet101
from data._datasets import DOMAIN_NAMES
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, 
                 patience=3, 
                 delta=0.001, 
                 verbose=False, 
                 path='checkpoint.pt', 
                 trace_func=print
                 ):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.trace_func = trace_func
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class HP_Tuner:
    def __init__(self, train_data, val_data, num_classes: int, num_domains: int, n_trials=50, save_dir="experiments/hp_results", fold_info: Dict[str, Any] = None):
        """
        Initialize the Hyperparameter Tuner
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            num_classes: Number of output classes
            num_domains: Number of domains
            n_trials: Number of optimization trials
        """
        self.train_data = train_data
        self.val_data = val_data
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.n_trials = n_trials
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = save_dir
        self.fold_info = fold_info

    def create_model(self, params_or_trial, verbose=False):
        #model_type = trial.suggest_categorical("model_type", ["resnet18", "resnet34", "resnet50", "resnet101"])
        model_type = "resnet50"
    
        actual_layers = []
        use_mixstyle = False

        is_trial = hasattr(params_or_trial, 'suggest_float')

        mixstyle_p = params_or_trial.suggest_float("mixstyle_p", 0.1, 0.9) if use_mixstyle else 0.0
        mixstyle_alpha = params_or_trial.suggest_float("mixstyle_alpha", 0.1, 0.5) if use_mixstyle else 0.0

        if use_mixstyle:
            mixstyle_layers = params_or_trial.suggest_categorical("mixstyle_layers", [
                    "layer1", "layer2", "layer3", "layer4",
                    "layer1+layer2", "layer1+layer2+layer3", "layer1+layer3",
                    "layer2+layer3", "all", "none"])
        
            if mixstyle_layers == "all":
                actual_layers = ["layer1", "layer2", "layer3", "layer4"]
            elif mixstyle_layers == "none":
                actual_layers = []
            else:
                actual_layers = mixstyle_layers.split("+")
    
        dropout = (
            params_or_trial.suggest_float("dropout", 0.0, 0.5)
            if is_trial else params_or_trial.get("dropout", 0.2)
        )

        model_kwargs = {
            "num_classes": self.num_classes,
            "num_domains": self.num_domains,
            "use_mixstyle": use_mixstyle,
            "mixstyle_layers": actual_layers,
            "mixstyle_p": mixstyle_p,
            "mixstyle_alpha": mixstyle_alpha,
            "dropout_p": dropout,
            "pretrained": True
        }
    
        models = {
            "resnet18": resnet18(**model_kwargs, verbose=verbose),
            "resnet34": resnet34(**model_kwargs, verbose=verbose),
            "resnet50": resnet50(**model_kwargs, verbose=verbose),
            "resnet101": resnet101(**model_kwargs, verbose=verbose)
        }

        return models[model_type]


    def objective(self, trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

        model = self.create_model(trial)
        device = self.device
        model.to(device)

        use_mixstyle = getattr(model, 'use_mixstyle', False)
        #checkpoint_path = Path("mnt/data/hahlers/tuning/woMS") / str(self.fold_info['fold']) / "checkpoints" / f"checkpoint_trial_{trial.number}.pt"
        #checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "AdamW"])
        if optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=lr, 
                                betas=(trial.suggest_float("beta1", 0.8, 0.99), trial.suggest_float("beta2", 0.9, 0.999)),
                                eps=trial.suggest_float("eps", 1e-8, 1e-6),
                                weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr, 
                                momentum=trial.suggest_float("momentum", 0.1, 0.9),
                                weight_decay=weight_decay, 
                                nesterov=trial.suggest_categorical("nesterov", [True, False]))
    
        scheduler_type = trial.suggest_categorical("scheduler", ["StepLR", "ReduceLROnPlateau", "CosineAnnealing"])
        if scheduler_type == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=trial.suggest_int("step_size", 5, 20),
                gamma=trial.suggest_float("gamma", 0.1, 0.9)
            )
        elif scheduler_type == "CosineAnnealing":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=trial.suggest_int("T_max", 10, 50),
                eta_min=trial.suggest_float("eta_min", 1e-6, 1e-3)
            )
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=trial.suggest_float("factor", 0.1, 0.5),
                patience=trial.suggest_int("patience", 2, 5)
            )

        criterion = nn.CrossEntropyLoss()
    
        def collate_fn(batch):
            imgs = torch.stack([item[0] for item in batch])
            labels = torch.tensor([item[1] for item in batch])
            domains = torch.tensor([item[2] for item in batch])
            return imgs, labels, domains

        train_loader = DataLoader(
            self.train_data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn
        )
    
        val_loader = DataLoader(
            self.val_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        early_stopper = EarlyStopping(
            patience=3, 
            delta=0.001,
            verbose=True,
            path=os.path.join(self.save_dir, "checkpoints", f"checkpoint_trial_{trial.number}.pt"),
            #path=os.path.join(checkpoint_path),
            trace_func=print
    )

        best_accuracy = 0
        for epoch in range(20):
            model.train()
            for inputs, labels, domain_idx in train_loader:  # DataLoader returns (input, label, domain_idx)
                inputs, labels, domain_idx = inputs.to(device), labels.to(device), domain_idx.to(device)
            
                optimizer.zero_grad()
                outputs = model(inputs, domain_idx) if use_mixstyle else model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
            model.eval()
            correct = 0
            total = 0
            val_loss = 0
            with torch.no_grad():
                for inputs, labels, domain_idx in val_loader:
                    inputs, labels, domain_idx = inputs.to(device), labels.to(device), domain_idx.to(device)
                    outputs = model(inputs, domain_idx)
                    val_loss += criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        
            accuracy = correct / total
            val_loss /= len(val_loader)

            early_stopper(val_loss, model)  # OR early_stopper(-accuracy, model)
            if early_stopper.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break
        
            if scheduler_type == "ReduceLROnPlateau":
                scheduler.step(accuracy)
            else:
                scheduler.step()
    
            trial.report(accuracy, epoch)
    
            if trial.should_prune():
                raise optuna.TrialPruned()
        
            if accuracy > best_accuracy:
                best_accuracy = accuracy

            self.log_trial_result(trial, accuracy, model)

        return best_accuracy
    

    def save_best_models(self, study, save_dir: str, top_k: int = 5):
        """Save the top-k best models from the study with organized folder structure."""
        try:
            base_dir = os.path.join(save_dir, "hp_results")
            models_dir = os.path.join(base_dir, "models")
            params_dir = os.path.join(base_dir, "params")
            checkpoints_dir = os.path.join(self.save_dir, "checkpoints")
        
            os.makedirs(models_dir, exist_ok=True)
            os.makedirs(params_dir, exist_ok=True)
            os.makedirs(checkpoints_dir, exist_ok=True)

            completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
            sorted_trials = sorted(completed_trials, key=lambda x: x.value, reverse=True)
    
            num_models_to_save = min(top_k, len(sorted_trials))
    
            for i in range(num_models_to_save):
                trial = sorted_trials[i]
                model = self.create_model(trial, verbose=False)
                model.to(self.device)
        
                model_path = os.path.join(models_dir, f"top_{i+1}_model.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'params': trial.params,
                    'value': trial.value,
                    'rank': i+1
                }, model_path)

                params_path = os.path.join(params_dir, f"top_{i+1}_params.json")
                with open(params_path, "w") as f:
                    json.dump({
                        'params': trial.params,
                        'value': trial.value,
                        'rank': i+1
                    }, f, indent=4)
    
            print(f"Saved top {num_models_to_save} models to {models_dir}")
            print(f"Saved parameters to {params_dir}")
        except Exception as e:
            print(f"Error saving best models: {str(e)}")


    def save_best_configs(self, study: optuna.Study, save_dir: str, top_k: int = 5) -> None:
        """
        Save the top-k best configurations from the study to YAML files.
        
        Args:
            study: Optuna study object
            save_dir: directory to save the configurations
            top_k: number of top configurations to save
        """
        try:
            base_dir = Path(save_dir).parent.parent
            fold_config_dir = base_dir / "configs" / f"fold_{self.fold_info['fold']}"
            fold_config_dir.mkdir(parents=True, exist_ok=True)
        
            completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
            sorted_trials = sorted(completed_trials, key=lambda x: x.value, reverse=True)[:top_k]
            
            for i, trial in enumerate(sorted_trials):
                config_data = {
                    "trial_id": trial.number,
                    "validation_accuracy": float(trial.value),
                    "params": trial.params,
                    "fold": self.fold_info['fold'],
                    "test_domain": self.fold_info['test_domain']
                }

                config_path = fold_config_dir / f"top_{i+1}_fold_{self.fold_info['fold']}_config.yaml"
                
                with open(config_path, "w") as f:
                    yaml.safe_dump(config_data, f, sort_keys=False)
        
            print(f"Top {len(sorted_trials)} configs saved in: {fold_config_dir}")
        
        except Exception as e:
            print(f"Failed to save configs: {str(e)}")


    def log_trial_result(self, trial, val_acc, model):
        csv_path = os.path.join(self.save_dir, f"results_trials_fold_{self.fold_info['fold']}.csv")

        clean_params = {k.replace('params_', ''): v for k, v in trial.params.items()}

        trial_data = trial.params.copy()
    
        # Add fold info and accuracy
        trial_data.update({
            **clean_params,
            'fold': self.fold_info['fold'],
            'test_domain': self.fold_info['test_domain'],
            'value': val_acc
        })

        all_params = [
            'fold', 'test_domain', 'lr', 'batch_size', 'weight_decay', 
            'optimizer', 'beta1', 'beta2', 'eps', 'momentum', 'nesterov',
            'scheduler', 'step_size', 'gamma', 'T_max', 'eta_min',
            'factor', 'patience', 'mixstyle_layers', 'mixstyle_p', 
            'mixstyle_alpha', 'dropout', 'val_acc'
        ]

        use_mixstyle = getattr(model, 'use_mixstyle', False)
        if not use_mixstyle:
            all_params = [
                'fold', 'test_domain', 'lr', 'batch_size', 'weight_decay', 
                'optimizer', 'beta1', 'beta2', 'eps', 'momentum', 'nesterov',
                'scheduler', 'step_size', 'gamma', 'T_max', 'eta_min',
                'factor', 'patience', 'dropout', 'val_acc'
            ]

        complete_data = {param: trial_data.get(param, None) for param in all_params}
        df = pd.DataFrame([complete_data])

        write_header = not os.path.exists(csv_path)
        df.to_csv(csv_path, mode='a', header=write_header, index=False)
    

    def run(self, save_dir: str = "experiments/hp_results") -> optuna.Study:
        os.makedirs(save_dir, exist_ok=True)
        fold_save_dir = os.path.join(save_dir, str(self.fold_info['fold']))
        os.makedirs(fold_save_dir, exist_ok=True)
        
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        study = optuna.create_study(direction="maximize", pruner=pruner, sampler=optuna.samplers.TPESampler())

        study.set_user_attr("fold", self.fold_info["fold"])
        study.set_user_attr("test_domain", self.fold_info["test_domain"])

        with tqdm(total=self.n_trials, desc="Hyperparameter-Tuning", unit="trial") as pbar:
            def update_pbar(study, trial):
                pbar.update(1)
                pbar.set_postfix({
                    "best_acc": f"{study.best_value:.4f}",
                    "current_acc": f"{trial.value:.4f}" if trial.value else "None"
                })
        
            study.optimize(self.objective, n_trials=self.n_trials, callbacks=[update_pbar])
            torch.cuda.empty_cache()

        self.save_best_configs(study, save_dir, top_k=5)
    
        trials_df = study.trials_dataframe()
        trials_df.columns = [col.replace('params_', '') for col in trials_df.columns]
        trials_df.to_csv(os.path.join(fold_save_dir, f"all_trials_fold_{self.fold_info['fold']}.csv"), index=False)

        try:
            if len(study.trials) > 1:
                fig1 = vis.plot_param_importances(study)
                fig1.write_html(os.path.join(save_dir, str(self.fold_info['fold']), "param_importances.html"))
            fig2 = vis.plot_optimization_history(study)
            fig2.write_html(os.path.join(save_dir, str(self.fold_info['fold']), "optimization_history.html"))
            print(f"Saved results.")
        except ImportError:
            print("Optuna Visualisation not available.")

        return study


    def load_best_model(self, study):
        """Load the best model from the study."""
        model = self.create_model(study.best_trial)
        checkpoint_path = os.path.join(self.save_dir, 
                                       "checkpoints", 
                                       f"checkpoint_trial_{study.best_trial.number}.pt"
                                       )
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True), strict=False)
        return model.to(self.device)
    

    def evaluate_model(self, model, test_loader):
        """Evaluate the model on the test set."""
        model = model.to(self.device)
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss().to(self.device)

        with torch.no_grad():
            for inputs, labels, _ in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                total_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
        accuracy = correct / total
        avg_loss = total_loss / len(test_loader)
        return {'accuracy': accuracy, 'loss': avg_loss}
    

    @staticmethod
    def compute_global_best_params(save_dir: str, use_mixstyle: bool) -> Dict[str, Any]:
        try:
            # Load all trial data as before
            all_trials = []
            domain_names = DOMAIN_NAMES['PACS']
        
            for fold_dir in Path(save_dir).glob("[0-9]"):
                fold_num = fold_dir.name
                trials_file = fold_dir / f"all_trials_fold_{fold_num}.csv"
            
                if trials_file.exists():
                    df = pd.read_csv(trials_file)
                    df["domain"] = domain_names[int(fold_num)]
                    all_trials.append(df)
        
            combined_df = pd.concat(all_trials)
            accuracy_col = 'value' if 'value' in combined_df.columns else 'val_acc'

            # Define parameter bins for continuous variables
            combined_df['lr_bin'] = pd.cut(combined_df['lr'], 
                                        bins=np.logspace(-5, -2, 6),
                                        labels=['1e-5', '3e-5', '1e-4', '3e-4', '1e-3'])
        
            combined_df['weight_decay_bin'] = pd.cut(combined_df['weight_decay'],
                                                bins=np.logspace(-5, -3, 5),
                                                labels=['1e-5', '3e-5', '1e-4', '3e-4'])
        
            # For mixstyle parameters (if used)
            if use_mixstyle:
                combined_df['mixstyle_p_bin'] = pd.cut(combined_df['mixstyle_p'],
                                                    bins=[0.1, 0.3, 0.5, 0.7, 0.9],
                                                    labels=['0.1-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9'])
        
            # Define grouping columns - now using binned values
            grouping_cols = [
                'lr_bin',
                'batch_size',  # Already categorical
                'weight_decay_bin',
                'optimizer',
                'scheduler'
            ]
        
            if use_mixstyle:
                grouping_cols.extend(['mixstyle_layers', 'mixstyle_p_bin', 'mixstyle_alpha'])
        
            # Group by binned parameters and calculate statistics
            grouped = combined_df.groupby(grouping_cols, observed=False)[accuracy_col]
            stats_df = grouped.agg(['mean', 'std', 'count']).reset_index().sort_values('mean', ascending=False)
        
            # Select best parameter configuration
            best_global = stats_df.iloc[0]
        
            # Calculate statistics about the best parameters
            best_mask = pd.Series(True, index=combined_df.index)
            for col in grouping_cols:
                best_mask &= (combined_df[col] == best_global[col])
        
            # Create results dictionary with representative values
            best_params = {
                'lr': float(combined_df.loc[best_mask, 'lr'].median()),
                'batch_size': int(best_global['batch_size']),
                'weight_decay': float(combined_df.loc[best_mask, 'weight_decay'].median()),
                'optimizer': best_global['optimizer'],
                'scheduler': best_global['scheduler']
            }
            
            # Add optimizer-specific params
            if best_params['optimizer'] in ['Adam', 'AdamW']:
                best_params.update({
                    'beta1': float(combined_df.loc[best_mask, 'beta1'].median()),
                    'beta2': float(combined_df.loc[best_mask, 'beta2'].median()),
                    'eps': float(combined_df.loc[best_mask, 'eps'].median())
                })
            
            elif best_params['optimizer'] == 'SGD':
                best_params.update({
                    'momentum': float(combined_df.loc[best_mask, 'momentum'].median()),
                    'nesterov': bool(combined_df.loc[best_mask, 'nesterov'].mode()[0])
                })
        
            # NEW: Improved scheduler parameter extraction
            scheduler_type = best_params['scheduler']
        
            if scheduler_type == 'StepLR':
                # Filter only trials that use StepLR
                step_trials = combined_df[combined_df['scheduler'] == 'StepLR']
                if not step_trials.empty:
                    best_params.update({
                        'step_size': int(step_trials['step_size'].median()),
                        'gamma': float(step_trials['gamma'].median())
                    })
                else:  # Fallback values if no StepLR trials exist
                    best_params.update({
                        'step_size': 10,
                        'gamma': 0.1
                    })
                
            elif scheduler_type == 'CosineAnnealing':
                cosine_trials = combined_df[combined_df['scheduler'] == 'CosineAnnealing']
                if not cosine_trials.empty:
                    best_params.update({
                        'T_max': int(cosine_trials['T_max'].median()),
                        'eta_min': float(cosine_trials['eta_min'].median())
                    })
                else:
                    best_params.update({
                        'T_max': 30,
                        'eta_min': 1e-4
                    })
                
            elif scheduler_type == 'ReduceLROnPlateau':
                plateau_trials = combined_df[combined_df['scheduler'] == 'ReduceLROnPlateau']
                if not plateau_trials.empty:
                    best_params.update({
                        'patience': int(plateau_trials['patience'].median()),
                        'factor': float(plateau_trials['factor'].median())
                    })
                else:
                    best_params.update({
                        'patience': 3,
                        'factor': 0.1
                    })

            best_per_domain = combined_df.loc[combined_df.groupby('domain')[accuracy_col].idxmax()]
            std_over_domains = best_per_domain[accuracy_col].std()

            global_mean = combined_df[accuracy_col].mean()
            global_std = combined_df[accuracy_col].std()
        
            # Save and return results (same as before)
            global_results = {
                "best_params": best_params,
                "mean_val_acc": float(best_global['mean']),
                "std_per_param": float(best_global['std']) if pd.notna(best_global['std']) else None,
                "std_over_domains": float(std_over_domains),
                "global_mean": float(global_mean),
                "global_std": float(global_std),
                "num_domains": int(best_global['count']),
                "domains_tested": list(combined_df.loc[best_mask, 'domain'].unique())
            }
        
            with open(f"{save_dir}/best_global_params_binned.json", "w") as f:
                json.dump(global_results, f, indent=4)
            
            return global_results

        except Exception as e:
            print(f"Error in binned parameter analysis: {str(e)}")
            return {}
        
    """
    def run_global_tuning(self, lodo_results_dir: str, n_trials: int = 30) -> Dict[str, Any]:
        #Determine optimal global parameters without training
        # 1. Analyze LODO results to get parameter distributions
        lodo_params = self.compute_global_best_params(lodo_results_dir, use_mixstyle=False)
    
        if not lodo_params:
            raise ValueError("Could not load LODO results for global tuning")

        # 2. Create parameter suggestion function
        def suggest_params(trial):
            params = {}
        
            # Learning rate with narrowed range
            params['lr'] = trial.suggest_float(
                "lr",
                max(1e-6, lodo_params['best_params'].get('lr', 1e-4) * 0.5),
                min(1e-2, lodo_params['best_params'].get('lr', 1e-4) * 2),
                log=True
            )
        
            # Fix optimizer to best type from LODO
            params['optimizer'] = lodo_params['best_params'].get('optimizer')
        
            # Constrained parameter ranges
            params['weight_decay'] = trial.suggest_float(
                "weight_decay",
                max(1e-6, lodo_params['best_params'].get('weight_decay', 1e-4) * 0.5),
                min(1e-3, lodo_params['best_params'].get('weight_decay', 1e-4) * 2),
                log=True
            )
        
            # Add other parameters similarly...
            params['batch_size'] = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        
            if params['optimizer'] in ['Adam', 'AdamW']:
                params['beta1'] = trial.suggest_float("beta1", 0.8, 0.99)
                params['beta2'] = trial.suggest_float("beta2", 0.9, 0.999)
                params['eps'] = trial.suggest_float("eps", 1e-8, 1e-6)
            elif params['optimizer'] == 'SGD':
                params['momentum'] = trial.suggest_float("momentum", 0.1, 0.9)
                params['nesterov'] = trial.suggest_categorical("nesterov", [True, False])
            
            # Scheduler-specific parameters
            if params['scheduler'] == 'StepLR':
                params['step_size'] = trial.suggest_int("step_size", 5, 20)
                params['gamma'] = trial.suggest_float("gamma", 0.1, 0.9)
            elif params['scheduler'] == 'CosineAnnealing':
                params['T_max'] = trial.suggest_int("T_max", 10, 50)
                params['eta_min'] = trial.suggest_float("eta_min", 1e-6, 1e-3)
            elif params['scheduler'] == 'ReduceLROnPlateau':
                params['factor'] = trial.suggest_float("factor", 0.1, 0.5)
                params['patience'] = trial.suggest_int("patience", 2, 5)
        
            # Add model architecture parameters
            params['dropout'] = trial.suggest_float(
                "dropout",
                max(0.0, lodo_params['best_params'].get('dropout', 0.1) - 0.1),
                min(0.5, lodo_params['best_params'].get('dropout', 0.1) + 0.1)
            )

            return params

        # 3. Create study and store all suggested parameters
        all_params = []
        study = optuna.create_study(direction="maximize")
    
        for _ in range(n_trials):
            trial = study.ask()
            params = suggest_params(trial)
            all_params.append(params)  # Jetzt wird params gespeichert
            study.tell(trial, 0.0)  # Dummy value

        # 4. Parameter analysis (statt dummy evaluation)
        # Beispiel: Median der vorgeschlagenen Parameter nehmen
        df_params = pd.DataFrame(all_params)

        # Get mode for categorical parameters
        categorical_params = ['optimizer', 'scheduler', 'mixstyle_layers']
        best_categorical = {}
        for param in categorical_params:
            if param in df_params.columns:
                best_categorical[param] = df_params[param].mode()[0] if not df_params[param].mode().empty else lodo_params['best_params'].get(param, 'AdamW')
    
        # Get median for numeric parameters
        numeric_params = [col for col in df_params.columns if col not in categorical_params]
        best_numeric = df_params[numeric_params].median().to_dict()

        # 5. Combine with LODO best params
        final_params = {
            **lodo_params['best_params'],  # Basis-Parameter
            **best_categorical,
            **best_numeric
        }

        # 6. Save results
        os.makedirs(os.path.join(self.save_dir, "global_params"), exist_ok=True)
        with open(os.path.join(self.save_dir, "global_params", "best_global_params.json"), "w") as f:
            json.dump({
                "best_params": final_params,
                "parameter_stats": df_params.describe().to_dict(),
                "lodo_stats": {
                    "mean_val_acc": lodo_params['mean_val_acc'],
                    "std_over_domains": lodo_params['std_over_domains']
                }
            }, f, indent=2)
    
        return final_params
    """

    """
    def run_global_tuning(self, lodo_results_dir: str, n_trials: int = 30) -> Dict[str, Any]:
        #Determine optimal global parameters without training
        # 1. Analyze LODO results to get parameter distributions
        lodo_params = self.compute_global_best_params(lodo_results_dir, use_mixstyle=False)

        if not lodo_params:
            raise ValueError("Could not load LODO results for global tuning")

        # 2. Create parameter suggestion function
        def suggest_params(trial):
            params = {}
    
            # Learning rate with narrowed range
            params['lr'] = trial.suggest_float(
                "lr",
                max(1e-6, lodo_params['best_params'].get('lr', 1e-4) * 0.5),
                min(1e-2, lodo_params['best_params'].get('lr', 1e-4) * 2),
                log=True
            )
    
            # Fix optimizer to best type from LODO
            params['optimizer'] = lodo_params['best_params'].get('optimizer', 'AdamW')
        
            # Fix scheduler to best type from LODO
            params['scheduler'] = lodo_params['best_params'].get('scheduler', 'StepLR')
    
            # Constrained parameter ranges
            params['weight_decay'] = trial.suggest_float(
                "weight_decay",
                max(1e-6, lodo_params['best_params'].get('weight_decay', 1e-4) * 0.5),
                min(1e-3, lodo_params['best_params'].get('weight_decay', 1e-4) * 2),
                log=True
            )
    
            # Batch size
            params['batch_size'] = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    
            # Optimizer-specific parameters
            if params['optimizer'] in ['Adam', 'AdamW']:
                params['beta1'] = trial.suggest_float("beta1", 0.8, 0.99)
                params['beta2'] = trial.suggest_float("beta2", 0.9, 0.999)
                params['eps'] = trial.suggest_float("eps", 1e-8, 1e-6)
            elif params['optimizer'] == 'SGD':
                params['momentum'] = trial.suggest_float("momentum", 0.1, 0.9)
                params['nesterov'] = trial.suggest_categorical("nesterov", [True, False])
        
            # Scheduler-specific parameters
            if params['scheduler'] == 'StepLR':
                params['step_size'] = trial.suggest_int("step_size", 5, 20)
                params['gamma'] = trial.suggest_float("gamma", 0.1, 0.9)
            
            elif params['scheduler'] == 'CosineAnnealing':
                params['T_max'] = trial.suggest_int("T_max", 10, 50)
                params['eta_min'] = trial.suggest_float("eta_min", 1e-6, 1e-3)
            
            elif params['scheduler'] == 'ReduceLROnPlateau':
                params['factor'] = trial.suggest_float("factor", 0.1, 0.5)
                params['patience'] = trial.suggest_int("patience", 2, 5)

            # Add model architecture parameters
            params['dropout'] = trial.suggest_float(
                "dropout",
                max(0.0, lodo_params['best_params'].get('dropout', 0.1) - 0.1),
                min(0.5, lodo_params['best_params'].get('dropout', 0.1) + 0.1)
            )

            return params

        def global_objective(trial):
        # Only suggest parameters, no actual training
            params = suggest_params(trial)
        
            # Calculate a dummy score based on parameter distances from LODO best
            score = 0.0
            for param, value in params.items():
                if param in lodo_params['best_params'] and lodo_params['best_params'][param] is not None:
                    if isinstance(value, (int, float)):
                        lodo_value = lodo_params['best_params'][param]
                        score -= abs(value - lodo_value) / (abs(lodo_value) + 1e-8)
        
            return score

        # 3. Create study and optimize
        study = optuna.create_study(direction="maximize")
    
        # Store all suggested parameters (for analysis later)
        all_params = []
    
        def optimization_callback(study, trial):
            all_params.append(trial.params)
    
        study.optimize(
            global_objective,  # Use the same objective as before
            n_trials=n_trials,
            callbacks=[optimization_callback]
        )

        # 4. Get the best parameters from the study
        final_params = study.best_params
    
        # 5. Save results
        os.makedirs(os.path.join(self.save_dir, "global_params"), exist_ok=True)
        with open(os.path.join(self.save_dir, "global_params", "best_global_params.json"), "w") as f:
            json.dump({
                "best_params": final_params,
                "study_results": {
                    "best_value": study.best_value,
                    "best_trial": study.best_trial.number,
                },
                "lodo_stats": {
                    "mean_val_acc": lodo_params['mean_val_acc'],
                    "std_over_domains": lodo_params['std_over_domains']
                }
            }, f, indent=2)

        return final_params
    """

    def run_global_tuning(self, lodo_results_dir: str, n_trials: int = 30) -> Dict[str, Any]:
        """Determine optimal global parameters without training"""
        # 1. Analyze LODO results to get parameter distributions
        lodo_params = self.compute_global_best_params(lodo_results_dir, use_mixstyle=False)

        if not lodo_params or 'best_params' not in lodo_params:
            raise ValueError("Could not load valid LODO results for global tuning")

        # 2. Create parameter suggestion function
        def suggest_params(trial):
            params = {}
        
            # Learning rate with narrowed range
            lr_default = 1e-4
            lr_min = max(1e-6, lodo_params['best_params'].get('lr', lr_default) * 0.5)
            lr_max = min(1e-2, lodo_params['best_params'].get('lr', lr_default) * 2)
            params['lr'] = trial.suggest_float("lr", lr_min, lr_max, log=True)
        
            # Fix optimizer to best type from LODO
            params['optimizer'] = lodo_params['best_params'].get('optimizer', 'AdamW')
        
            # Fix scheduler to best type from LODO
            params['scheduler'] = lodo_params['best_params'].get('scheduler', 'StepLR')

            # Constrained parameter ranges
            wd_default = 1e-4
            wd_min = max(1e-6, lodo_params['best_params'].get('weight_decay', wd_default) * 0.5)
            wd_max = min(1e-3, lodo_params['best_params'].get('weight_decay', wd_default) * 2)
            params['weight_decay'] = trial.suggest_float("weight_decay", wd_min, wd_max, log=True)
        
            # Batch size
            params['batch_size'] = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        
            # Optimizer-specific parameters
            if params['optimizer'] in ['Adam', 'AdamW']:
                params['beta1'] = trial.suggest_float("beta1", 0.8, 0.99)
                params['beta2'] = trial.suggest_float("beta2", 0.9, 0.999)
                params['eps'] = trial.suggest_float("eps", 1e-8, 1e-6)
            elif params['optimizer'] == 'SGD':
                params['momentum'] = trial.suggest_float("momentum", 0.1, 0.9)
                params['nesterov'] = trial.suggest_categorical("nesterov", [True, False])
        
            # Scheduler-specific parameters
            if params['scheduler'] == 'StepLR':
                params['step_size'] = trial.suggest_int("step_size", 5, 20)
                params['gamma'] = trial.suggest_float("gamma", 0.1, 0.9)
            elif params['scheduler'] == 'CosineAnnealing':
                params['T_max'] = trial.suggest_int("T_max", 10, 50)
                params['eta_min'] = trial.suggest_float("eta_min", 1e-6, 1e-3)
            elif params['scheduler'] == 'ReduceLROnPlateau':
                params['factor'] = trial.suggest_float("factor", 0.1, 0.5)
                params['patience'] = trial.suggest_int("patience", 2, 5)

            # Add model architecture parameters
            dropout_default = 0.1
            dropout_min = max(0.0, lodo_params['best_params'].get('dropout', dropout_default) - 0.1)
            dropout_max = min(0.5, lodo_params['best_params'].get('dropout', dropout_default) + 0.1)
            params['dropout'] = trial.suggest_float("dropout", dropout_min, dropout_max)

            return params

        def global_objective(trial):
            try:
                params = suggest_params(trial)
            
                # Calculate a score based on parameter distances from LODO best
                score = 0.0
                valid_params = 0
            
                for param, value in params.items():
                    if param in lodo_params['best_params'] and lodo_params['best_params'][param] is not None:
                        if isinstance(value, (int, float)):
                            lodo_value = lodo_params['best_params'][param]
                            if not np.isnan(lodo_value):
                                score -= abs(value - lodo_value) / (abs(lodo_value) + 1e-8)
                                valid_params += 1
            
                # Normalize score by number of valid parameters
                if valid_params > 0:
                    return score / valid_params
                return 0.0  # Fallback if no valid parameters
        
            except Exception as e:
                print(f"Error in global objective: {str(e)}")
                return 0.0  # Return neutral score on error

        # 3. Create study and optimize
        study = optuna.create_study(direction="maximize")

        try:
            study.optimize(
                global_objective,
                n_trials=n_trials,
                callbacks=[]
            )
        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            # Fallback to LODO params if optimization fails
            return lodo_params['best_params']

        # 4. Get the best parameters from the study or fallback to LODO params
        if len(study.trials) > 0 and study.best_trial is not None:
            final_params = study.best_params
        else:
            print("No valid trials completed, using LODO params as fallback")
            final_params = lodo_params['best_params']

        # 5. Save results
        os.makedirs(os.path.join(self.save_dir, "global_params"), exist_ok=True)
        with open(os.path.join(self.save_dir, "global_params", "best_global_params.json"), "w") as f:
            json.dump({
                "best_params": final_params,
                "study_results": {
                    "best_value": study.best_value if study.best_trial else None,
                    "best_trial": study.best_trial.number if study.best_trial else None,
                },
                "lodo_stats": {
                    "mean_val_acc": lodo_params.get('mean_val_acc', None),
                    "std_over_domains": lodo_params.get('std_over_domains', None)
                }
            }, f, indent=2)

        return final_params