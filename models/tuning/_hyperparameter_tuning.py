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


class HP_Tuner:
    def __init__(self, train_data, val_data, model_name, num_classes: int, num_domains: int):
        self.train_data = train_data
        self.val_data = val_data
        self.model_name = model_name
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.n_trials = 50  # Anzahl der Versuche für Optuna
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_model(self, trial):
        model_type = trial.suggest_categorical("model_type", ["resnet18", "resnet34", "resnet50", "resnet101"])
    
        use_mixstyle = trial.suggest_categorical("use_mixstyle", [True, False])
        mixstyle_layers = trial.suggest_categorical("mixstyle_layers", 
                                [["layer1"], ["layer2"], ["layer3"], ["layer4"], 
                                ["layer1", "layer2"], ["layer1", "layer2", "layer3"], 
                                ["layer1", "layer3"], ["layer2", "layer3"], 
                                ["layer1", "layer2", "layer3", "layer4"], ["layer1", "layer2", "layer4"],
                                ["layer1", "layer3", "layer4"], ["layer2", "layer3", "layer4"], []])
        mixstyle_p = trial.suggest_float("mixstyle_p", 0.1, 0.9) if use_mixstyle else 0.0
        mixstyle_alpha = trial.suggest_float("mixstyle_alpha", 0.1, 0.5) if use_mixstyle else 0.0
    
        model_kwargs = {
            "num_classes": self.num_classes,
            "num_domains": self.num_domains,
            "use_mixstyle": use_mixstyle,
            "mixstyle_layers": mixstyle_layers,
            "mixstyle_p": mixstyle_p,
            "mixstyle_alpha": mixstyle_alpha,
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
        # Hyperparameter
        lr = trial.suggest_float("lr", 1e-5, 1e-4, 1e-3, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    
        # Modell und Gerät
        model = self.create_model(trial)
        device = self.device
        model.to(device)
    
        # Optimierer
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "AdamW"])
        if optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=lr, 
                                betas=(trial.suggest_float("beta1", 0.8, 0.99), trial.suggest_float("beta2", 0.9, 0.999)),
                                eps=trial.suggest_float("eps", 1e-8, 1e-6),
                                weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-3))
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr, 
                                momentum=trial.suggest_float("momentum", 0.1, 0.9),
                                weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-3))
    
        # Learning Rate Scheduler
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
    
        # Beispiel-Dataloader (mit Domain-Labels)
        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_data, batch_size=batch_size)

        # Training Loop
        best_accuracy = 0
        for epoch in range(20):
            model.train()
            for inputs, labels, domain_idx in train_loader:  # Annahme: DataLoader gibt (input, label, domain_idx) zurück
                inputs, labels, domain_idx = inputs.to(device), labels.to(device), domain_idx.to(device)
            
                optimizer.zero_grad()
                outputs = model(inputs, domain_idx)  # Wichtig: Domain-Index übergeben!
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
            # Validation
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
        
            # Scheduler Update
            if scheduler_type == "ReduceLROnPlateau":
                scheduler.step(accuracy)
            else:
                scheduler.step()
        
            # Berichte an Optuna
            trial.report(accuracy, epoch)
        
            # Pruning
            if trial.should_prune():
                raise optuna.TrialPruned()
        
            if accuracy > best_accuracy:
                best_accuracy = accuracy

        return best_accuracy
    
    def run(self, save_dir: str = "results"):
        os.makedirs(save_dir, exist_ok=True)
    
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        study = optuna.create_study(direction="maximize", pruner=pruner, sampler=optuna.samplers.TPESampler())
        study.optimize(self.objective, n_trials=self.n_trials)

        best_params = study.best_params
        with open(f"{save_dir}/best_params.json", "w") as f:
            json.dump(best_params, f, indent=4)
    
        trials_df = study.trials_dataframe()
        trials_df.to_csv(f"{save_dir}/all_trials.csv", index=False)

        print(f"\nSaved results in: {save_dir}/:")
        print(f"- best_params.json (best hyperparams)")
        print(f"- all_trials.csv (all trials)")

        # Visualisierung
        try:
            fig1 = vis.plot_optimization_history(study)
            fig2 = vis.plot_param_importances(study)
            fig1.write_html(f"{save_dir}/optimization_history.html")
            fig2.write_html(f"{save_dir}/param_importances.html")
        except ImportError:
            print("Optuna Visualisation not available.")

        return study
