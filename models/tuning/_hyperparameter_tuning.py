import os
import json
import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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

    def create_model(self, trial, verbose=False):
        #model_type = trial.suggest_categorical("model_type", ["resnet18", "resnet34", "resnet50", "resnet101"])
        model_type = "resnet50"
    
        actual_layers = []
        use_mixstyle = False
        mixstyle_p = trial.suggest_float("mixstyle_p", 0.1, 0.9) if use_mixstyle else 0.0
        mixstyle_alpha = trial.suggest_float("mixstyle_alpha", 0.1, 0.5) if use_mixstyle else 0.0

        if use_mixstyle:
            mixstyle_layers = trial.suggest_categorical("mixstyle_layers", [
                    "layer1", "layer2", "layer3", "layer4",
                    "layer1+layer2", "layer1+layer2+layer3", "layer1+layer3",
                    "layer2+layer3", "all", "none"])
        
            if mixstyle_layers == "all":
                actual_layers = ["layer1", "layer2", "layer3", "layer4"]
            elif mixstyle_layers == "none":
                actual_layers = []
            else:
                actual_layers = mixstyle_layers.split("+")
    
        dropout = trial.suggest_float("dropout", 0.0, 0.5)

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
            'val_acc': val_acc
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
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()

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
        """
        Calculates the average validation accuracy across all domains 
        and saves the best global hyperparameters.
    
        Args:
            save_dir: base directory where the results are saved
    
        Returns:
            dict with the best global hyperparameters and their mean accuracy
        """
        
        all_trials = []
    
        for fold_dir in Path(save_dir).glob("[0-9]"):  # search for directories named with single digit (0-9)
            fold_num = fold_dir.name
            trials_file = fold_dir / f"all_trials_fold_{fold_num}.csv"
        
            if trials_file.exists():
                df = pd.read_csv(trials_file)
                df["domain"] = DOMAIN_NAMES['PACS'][int(fold_num)]
                all_trials.append(df)
            
        if not all_trials:
            raise ValueError("Couldn't find domain results!")
    
        combined_df = pd.concat(all_trials)
    
        accuracy_col = 'value'
        param_cols = ['lr', 'batch_size', 'weight_decay', 'optimizer',
                      'scheduler', 'dropout', 'mixstyle_layers', 'mixstyle_p', 'mixstyle_alpha']
        
        if use_mixstyle == False:
            param_cols = ['lr', 'batch_size', 'weight_decay', 'optimizer', 'scheduler', 'dropout']
        
        missing_cols = [col for col in param_cols if col not in combined_df.columns]
        if missing_cols:
            raise ValueError(f"Missing parameter cols in the data: {missing_cols}")
    
        # mean, std and count of domains for each parameter combination
        best_global = (
            combined_df.groupby(param_cols)
            [accuracy_col].agg(["mean", "std", "count"])
            .sort_values("mean", ascending=False)
            .reset_index()
            .iloc[0]  # best parameter set (highest mean accuracy)
        )
        
        best_per_fold = combined_df.loc[combined_df.groupby('domain')[accuracy_col].idxmax()]
        std_over_folds = best_per_fold[accuracy_col].std()

        global_mean = combined_df[accuracy_col].mean()
        global_std = combined_df[accuracy_col].std()

        global_results = {
            "best_params": best_global[param_cols].to_dict(),
            "mean_val_acc": float(best_global["mean"]),
            "std_per_param": float(best_global["std"]),
            "std_over_folds": float(std_over_folds),
            "global_mean": float(global_mean),
            "global_std": float(global_std),
            "num_domains": int(best_global["count"])
        }

        with open(f"{save_dir}/best_global_params.json", "w") as f:
            json.dump(global_results, f, indent=4)
    
        print(
            f"\nGlobal best Hyperparameters: "
            f"Mean Accuracy = {best_global['mean']:.2%} ± "
            f"{best_global['std']:.2%} (per param) / "
            f"{std_over_folds:.2%} (over folds)"
        )
        
        return global_results
    