import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import optuna
from typing import Dict


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
            for config_file in config_path.glob("*.yaml"):
                shutil.copy2(config_file, target_dir / "configs" / config_file.name)
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


def save_training_results(config: Dict, target_dir: str = "/mnt/data/hahlers/training") -> None:
    """
    Save all training results (configs, models, visualizations) to a specified folder.
    
    Args:
        target_dir: Base directory where results should be saved (default: "/mnt/data/hahlers/training")
    """
    target_path = Path(os.path.expanduser(target_dir))
    target_path.mkdir(parents=True, exist_ok=True)

    try:
        # Create subdirectories
        (target_path / "saved_models").mkdir(exist_ok=True)
        (target_path / "visualizations").mkdir(exist_ok=True)
        (target_path / "logs").mkdir(exist_ok=True)

        print("Copying training results...")

        # Copy saved models
        models_dir = Path(config['save_dir'])
        if models_dir.exists():
            for model_file in models_dir.glob("*.pt"):
                shutil.copy2(model_file, target_path / "saved_models" / model_file.name)
            # Copy results JSON
            results_file = models_dir / "lodo_results.json"
            if results_file.exists():
                shutil.copy2(results_file, target_path / "lodo_results.json")

        # Copy visualizations
        vis_dir = Path(config['vis_dir'])
        if vis_dir.exists():
            for vis_file in vis_dir.glob("*"):
                if vis_file.is_file():
                    shutil.copy2(vis_file, target_path / "visualizations" / vis_file.name)
                elif vis_file.is_dir():  # For subdirectories like domain-specific visualizations
                    (target_path / "visualizations" / vis_file.name).mkdir(exist_ok=True)
                    for sub_file in vis_file.glob("*"):
                        shutil.copy2(sub_file, target_path / "visualizations" / vis_file.name / sub_file.name)

        # Copy TensorBoard logs
        log_dir = Path(config['log_dir'])
        if log_dir.exists():
            for log_file in log_dir.glob("*"):
                shutil.copy2(log_file, target_path / "logs" / log_file.name)

    except Exception as e:
        print(f"Error while copying training results: {e}")
        return
        
    print(f"\nAll training results saved to: {target_path}")