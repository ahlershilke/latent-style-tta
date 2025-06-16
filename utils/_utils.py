import os
import shutil
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.tuning._hyperparameter_tuning import EarlyStopping, HP_Tuner
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import optuna


def analyze_and_visualize_studies(all_studies, save_dir, use_mixstyle: bool):
    """
    Analyze and visualize results from multiple Optuna studies.
    Args:
        all_studies (List[optuna.Study]): List of Optuna study objects.
        save_dir (str): Directory to save the results and visualizations.
    """
    os.makedirs(save_dir, exist_ok=True)

    all_results = []
    param_importance_aggregated = {}
    
    all_param_keys = [
        'lr', 'batch_size', 'weight_decay', 'optimizer', 'scheduler', 'dropout',
        'mixstyle_layers', 'mixstyle_p', 'mixstyle_alpha', 'beta1', 'beta2',
        'momentum', 'eps', 'nesterov', 'step_size', 'T_max', 'eta_min', 'factor',
        'patience', 'gamma'
    ]

    if use_mixstyle == False:
        all_param_keys = [
            'lr', 'batch_size', 'weight_decay', 'optimizer', 'scheduler', 'dropout',
            'beta1', 'beta2', 'momentum', 'eps', 'nesterov', 'step_size', 'T_max', 
            'eta_min', 'factor', 'patience', 'gamma'
        ]
    
    for study in all_studies:
        best_trial = study.best_trial

        trial_dict = {
            "fold": study.user_attrs.get("fold", "N/A"),
            "test_domain": study.user_attrs.get("test_domain", "N/A"),
            "best_accuracy": best_trial.value,
        }

        for key in all_param_keys:
            trial_dict[key] = best_trial.params.get(key, None)

        all_results.append(trial_dict)
        
        # aggregate parameter importances
        for param, importance in optuna.importance.get_param_importances(study).items():
            param_importance_aggregated[param] = param_importance_aggregated.get(param, 0) + importance

    # normalize parameter importances
    total_importance = sum(param_importance_aggregated.values())
    param_importance_normalized = {k: v / total_importance for k, v in param_importance_aggregated.items()}

    # 2. Visualization (Plotly)
    # a) Optimization History (all folds)
    fig_history = make_subplots()
    for i, study in enumerate(all_studies):
        df = study.trials_dataframe()
        fig_history.add_scatter(
            x=df["number"], 
            y=df["value"], 
            mode="lines+markers",
            name=f"Fold {i} ({study.user_attrs.get('test_domain', 'N/A')})",
            opacity=0.7
        )
    fig_history.update_layout(
        title="Optimization History (All Folds)",
        xaxis_title="Trial Number",
        yaxis_title="Validation Accuracy",
        hovermode="x unified"
    )
    fig_history.write_html(os.path.join(save_dir, "global_optimization_history.html"))

    # b) Parameter Importances (aggregated across all folds)
    fig_importance = go.Figure(go.Bar(
        x=list(param_importance_normalized.values()),
        y=list(param_importance_normalized.keys()),
        orientation='h',
        marker_color='skyblue'
    ))
    fig_importance.update_layout(
        title="Global Parameter Importances (Mean Across Folds)",
        xaxis_title="Normalized Importance",
        yaxis_title="Parameter",
        height=600
    )
    fig_importance.write_html(os.path.join(save_dir, "global_param_importances.html"))

    # 3. Statistical Summary
    # a) CSV
    df_results = pd.DataFrame(all_results)
    csv_path = os.path.join(save_dir, "global_best_trials.csv")
    df_results.to_csv(csv_path, index=False)

    # b) JSON
    summary_df = pd.DataFrame([{
        "accuracy_mean": float(df_results["best_accuracy"].mean()),
        "accuracy_std": float(df_results["best_accuracy"].std()),
        "accuracy_min": float(df_results["best_accuracy"].min()),
        "accuracy_max": float(df_results["best_accuracy"].max()),
        "accuracy_median": float(df_results["best_accuracy"].median()),
        "best_overall_params": str(df_results.iloc[df_results["best_accuracy"].idxmax()].to_dict()),
        "most_common_optimizer": df_results["optimizer"].mode()[0],
        "median_lr": float(df_results["lr"].median())
    }])

    json_path = os.path.join(save_dir, "global_summary.json")
    summary_df.to_json(json_path, orient='records', indent=2, force_ascii=False)

    print(f"Saved global results in {save_dir}.")


def convert_to_serializable(obj):
    """Convert numpy and pandas objects to standard Python types for JSON serialization."""
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    return obj


def save_tuning_results(config_dir: str, results_dir: str):
    """
    Save all tuning results in specified folder.
    """
    #response = input("Would you like to save the tuning results for all folds in another folder? (y/n): ").strip().lower()
        
    #if response != 'y':
       # print("No custom directory provided. Results will not be externally saved.")
        #return
        
    custom_dir = "/mnt/data/hahlers/tuning"
    target_dir = Path(os.path.expanduser(custom_dir))
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        (Path(target_dir) / "configs").mkdir(exist_ok=True)
        (Path(target_dir) / "results").mkdir(exist_ok=True)

        print("Copying files...")

        config_path = Path(config_dir)
        if config_path.exists():
            for fold_dir in Path(config_dir).glob("fold_*"):
                (target_dir / "configs" / fold_dir.name).mkdir(parents=True, exist_ok=True)
                for config_file in fold_dir.glob("*.yaml"):
                    shutil.copy2(config_file, target_dir / "configs" / fold_dir.name / config_file.name)

        results_path = Path(results_dir)
        if results_path.exists():
            for result_file in Path(results_dir).glob("*.*"):
                if result_file.suffix in ['.csv', '.html', '.json']:
                    shutil.copy2(result_file, target_dir / result_file.name)
            
            for fold_dir in Path(results_dir).glob("[0-9]"):
                (target_dir / "results" / fold_dir.name).mkdir(parents=True, exist_ok=True)
                for result_file in fold_dir.glob("*"):
                    if result_file.is_file():
                        shutil.copy2(result_file, target_dir / "results" / fold_dir.name / result_file.name)
                ckpt_source = fold_dir / "checkpoints"
                if ckpt_source.exists():
                    (target_dir / "results" / fold_dir.name / "checkpoints").mkdir(exist_ok=True)
                    for ckpt_file in (fold_dir / "checkpoints").glob("*.pt"):
                        shutil.copy2(ckpt_file, target_dir / "results" / fold_dir.name / "checkpoints" / ckpt_file.name)
    
    except Exception as e:
        print(f"Error while copying files: {e}")
        return
        
    print(f"\nAll results saved to: {target_dir}")


def suggest_params(trial, lodo_params: Dict[str, Any]) -> Dict[str, Any]:
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


def evaluate_global_objective(trial, tuner_instance: HP_Tuner, train_data, val_data, device, save_dir):
    """ Evaluate the global objective function for hyperparameter tuning."""
    # 1. Parameter vorschlagen (suggest_params muss in tuner_instance sein)
    params = tuner_instance.suggest_params(trial)
    
    # 2. Modell erstellen
    model = tuner_instance.create_model(params)
    model.to(device)
    
    # 3. Optimierer und Scheduler einrichten
    optimizer = create_optimizer(params, model)
    scheduler = create_scheduler(params, optimizer)
    
    # 4. DataLoader vorbereiten
    train_loader, val_loader = create_dataloaders(train_data, val_data, params['batch_size'])
    
    # 5. Training durchführen
    best_accuracy = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_path=os.path.join(save_dir, f"global_trial_{trial.number}.pt"),
        scheduler_type=params['scheduler']
    )
    
    return best_accuracy

# Hilfsfunktionen für bessere Modularität
def create_optimizer(params, model):
    optimizer_name = params['optimizer']
    lr = params['lr']
    weight_decay = params['weight_decay']
    
    if optimizer_name == "Adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        return optim.AdamW(model.parameters(), lr=lr, 
                         betas=(params['beta1'], params['beta2']),
                         eps=params['eps'],
                         weight_decay=weight_decay)
    else:
        return optim.SGD(model.parameters(), lr=lr, 
                       momentum=params['momentum'],
                       weight_decay=weight_decay, 
                       nesterov=params['nesterov'])

def create_scheduler(params, optimizer):
    scheduler_type = params['scheduler']
    
    if scheduler_type == "StepLR":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params['step_size'],
            gamma=params['gamma']
        )
    elif scheduler_type == "CosineAnnealing":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=params['T_max'],
            eta_min=params['eta_min']
        )
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=params['factor'],
        patience=params['patience']
    )

def create_dataloaders(train_data, val_data, batch_size):
    def collate_fn(batch):
        imgs = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        domains = torch.tensor([item[2] for item in batch])
        return imgs, labels, domains

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader

def train_and_evaluate(model, train_loader, val_loader, optimizer, scheduler, 
                      device, checkpoint_path, scheduler_type, max_epochs=20):
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(
        patience=3, 
        delta=0.001,
        verbose=False,
        path=checkpoint_path
    )

    best_accuracy = 0
    for epoch in range(max_epochs):
        # Training
        model.train()
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Evaluation
        accuracy = evaluate_model(model, val_loader, device)
        
        # Early Stopping und Scheduler
        early_stopper(accuracy, model)
        if early_stopper.early_stop:
            break
            
        if scheduler_type == "ReduceLROnPlateau":
            scheduler.step(accuracy)
        else:
            scheduler.step()
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
    
    return best_accuracy

def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels, _ in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total