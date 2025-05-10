import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
import optuna.visualization as vis
from optuna.trial import TrialState
from optuna.pruners import MedianPruner
from models import resnet18, resnet34, resnet50, resnet101
from tqdm import tqdm


class HP_Tuner:
    def __init__(self, train_data, val_data, num_classes: int, num_domains: int, n_trials=50):
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

    def create_model(self, trial):
        model_type = trial.suggest_categorical("model_type", ["resnet18", "resnet34", "resnet50", "resnet101"])
    
        #use_mixstyle = trial.suggest_categorical("use_mixstyle", [True, False])
        use_mixstyle = True

        if use_mixstyle:
            mixstyle_layers = trial.suggest_categorical("mixstyle_layers", [
                "layer1", "layer2", "layer3", "layer4",
                "layer1+layer2", "layer1+layer2+layer3", "layer1+layer3",
                "layer2+layer3", "all", "none"])
            mixstyle_p = trial.suggest_float("mixstyle_p", 0.1, 0.9) if use_mixstyle else 0.0
            mixstyle_alpha = trial.suggest_float("mixstyle_alpha", 0.1, 0.5) if use_mixstyle else 0.0
        else:
            mixstyle_layers = []
            mixstyle_p = 0.0
            mixstyle_alpha = 0.0
            mixstyle_layers = trial.suggest_categorical("mixstyle_layers", [
                "layer1", "layer2", "layer3", "layer4",
                "layer1+2", "layer1+2+3", "layer1+3",
                "layer2+3", "all", "none"])
        
        if mixstyle_layers == "all":
            actual_layers = ["layer1", "layer2", "layer3", "layer4"]
        elif mixstyle_layers == "none":
            actual_layers = []
        else:
            actual_layers = mixstyle_layers.split("+")

        mixstyle_p = trial.suggest_float("mixstyle_p", 0.1, 0.9) if use_mixstyle else 0.0
        mixstyle_alpha = trial.suggest_float("mixstyle_alpha", 0.1, 0.5) if use_mixstyle else 0.0
    
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
            "resnet18": resnet18(**model_kwargs),
            "resnet34": resnet34(**model_kwargs),
            "resnet50": resnet50(**model_kwargs),
            "resnet101": resnet101(**model_kwargs)
        }

        return models[model_type]

    def objective(self, trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

        model = self.create_model(trial)
        device = self.device
        model.to(device)
    
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

        best_accuracy = 0
        for epoch in range(20):
            model.train()
            for inputs, labels, domain_idx in train_loader:  # DataLoader returns (input, label, domain_idx)
                inputs, labels, domain_idx = inputs.to(device), labels.to(device), domain_idx.to(device)
            
                optimizer.zero_grad()
                outputs = model(inputs, domain_idx)
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
        
            if scheduler_type == "ReduceLROnPlateau":
                scheduler.step(accuracy)
            else:
                scheduler.step()
    
            trial.report(accuracy, epoch)
    
            if trial.should_prune():
                raise optuna.TrialPruned()
        
            if accuracy > best_accuracy:
                best_accuracy = accuracy

        return best_accuracy
    

    def save_best_model(self, study, save_dir: str):
        """Save the best model from the study."""
        try:
            best_model = self.create_model(study.best_trial)
            best_model.to(self.device)
        
            model_path = f"{save_dir}/best_model.pth"
            torch.save({
                'model_state_dict': best_model.state_dict(),
                'best_params': study.best_params,
                'best_value': study.best_value
            }, model_path)
        
            print(f"Best model saved to {model_path}")
        except Exception as e:
            print(f"Error saving best model: {str(e)}")
    

    def run(self, save_dir: str = "results"):
        os.makedirs(save_dir, exist_ok=True)
    
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        study = optuna.create_study(direction="maximize", pruner=pruner, sampler=optuna.samplers.TPESampler())
        
        with tqdm(total=self.n_trials, desc="Hyperparameter-Tuning", unit="trial") as pbar:
            def update_pbar(study, trial):
                pbar.update(1)
                pbar.set_postfix({
                    "best_acc": f"{study.best_value:.4f}",
                    "current_acc": f"{trial.value:.4f}" if trial.value else "None"
                })
        
            study.optimize(self.objective, n_trials=self.n_trials, callbacks=[update_pbar])

        best_params = study.best_params
        with open(f"{save_dir}/best_params.json", "w") as f:
            json.dump(best_params, f, indent=4)
    
        trials_df = study.trials_dataframe()
        trials_df.to_csv(f"{save_dir}/all_trials.csv", index=False)

        print(f"\nSaved results in: {save_dir}/:")
        print(f"- best_params.json (best hyperparams)")
        print(f"- all_trials.csv (all trials)")

        try:
            fig1 = vis.plot_optimization_history(study)
            fig1.write_html(f"{save_dir}/optimization_history.html")
            if len(study.trials) > 1:
                fig2 = vis.plot_param_importances(study)
                fig2.write_html(f"{save_dir}/param_importances.html")
        except ImportError:
            print("Optuna Visualisation not available.")

        return study
