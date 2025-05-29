import os
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import optuna


def analyze_and_visualize_studies(all_studies, save_dir):
    # Sicherstellen, dass das Verzeichnis existiert
    os.makedirs(save_dir, exist_ok=True)

    # 1. Datensammlung für alle Studien
    all_results = []
    param_importance_aggregated = {}
    
    for study in all_studies:
        # Metadaten und beste Trial-Ergebnisse sammeln
        best_trial = study.best_trial
        all_results.append({
            "fold": study.user_attrs.get("fold", "N/A"),
            "test_domain": study.user_attrs.get("test_domain", "N/A"),
            "best_accuracy": best_trial.value,
            **best_trial.params
        })
        
        # Parameter-Importances aggregieren
        for param, importance in optuna.importance.get_param_importances(study).items():
            param_importance_aggregated[param] = param_importance_aggregated.get(param, 0) + importance

    # Normalisierung der Importances
    total_importance = sum(param_importance_aggregated.values())
    param_importance_normalized = {k: v / total_importance for k, v in param_importance_aggregated.items()}

    # 2. Visualisierungen (Plotly)
    # a) Optimization History (Alle Folds überlagert)
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

    # b) Parameter Importances (Aggregiert)
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

    # 3. Statistische Auswertung (CSV + JSON)
    # a) CSV mit allen Ergebnissen
    df_results = pd.DataFrame(all_results)
    csv_path = os.path.join(save_dir, "global_best_trials.csv")
    df_results.to_csv(csv_path, index=False)

    # b) JSON-Zusammenfassung
    """
    summary_stats = {
        "accuracy": {
            "mean": float(df_results["best_accuracy"].mean()),
            "std": float(df_results["best_accuracy"].std()),
            "min": float(df_results["best_accuracy"].min()),
            "max": float(df_results["best_accuracy"].max()),
            "median": float(df_results["best_accuracy"].median())
        },
        "best_overall_params": dict(df_results.iloc[df_results["best_accuracy"].idxmax()]),
        "most_common_best_params": {
            #"model_type": df_results["model_type"].mode()[0],
            "optimizer": df_results["optimizer"].mode()[0],
            "lr_median": float(df_results["lr"].median())
        }
    }
    
    json_path = os.path.join(save_dir, "global_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary_stats, f, indent=2, default=convert_to_serializable)
    """

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