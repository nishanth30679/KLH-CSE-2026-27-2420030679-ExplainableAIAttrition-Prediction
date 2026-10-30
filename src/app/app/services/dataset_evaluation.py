"""Dataset-specific training and evaluation.

This module deliberately keeps one active evaluation in memory/on disk.  Every
upload is split, trained, and scored independently; no prior dataset metrics
are read or combined.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.services.model_utils import AttritionANN, compute_hybrid_probability, predict_ann_proba


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", value)


def _encode_target(series: pd.Series) -> tuple[np.ndarray, dict[str, int]]:
    values = series.dropna()
    if values.nunique() != 2:
        raise ValueError("The attrition target must contain exactly two classes.")
    if pd.api.types.is_numeric_dtype(values):
        labels = sorted(values.unique().tolist())
        mapping = {str(labels[0]): 0, str(labels[1]): 1}
        encoded = series.map({labels[0]: 0, labels[1]: 1})
    else:
        normalized = values.astype(str).str.strip().str.lower()
        positive = next((v for v in normalized.unique() if v in {"yes", "y", "true", "1", "leave", "left", "attrition"}), None)
        if positive is None:
            positive = sorted(normalized.unique())[-1]
        classes = sorted(normalized.unique())
        negative = next(v for v in classes if v != positive)
        mapping = {negative: 0, positive: 1}
        encoded = series.astype(str).str.strip().str.lower().map(mapping)
    if encoded.isna().any():
        raise ValueError("The selected target contains missing or unsupported values.")
    return encoded.to_numpy(dtype=np.int64), mapping


def inspect_dataset(df: pd.DataFrame) -> dict[str, Any]:
    candidates = []
    for column in df.columns:
        count = int(df[column].nunique(dropna=True))
        if count == 2:
            candidates.append(column)
    return {
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "target_candidates": candidates,
        "missing_values": int(df.isna().sum().sum()),
    }


def _metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, predicted)), 4),
        "precision": round(float(precision_score(y_true, predicted, zero_division=0)), 4),
        "recall_sensitivity": round(float(recall_score(y_true, predicted, zero_division=0)), 4),
        "specificity": round(float(tn / (tn + fp)) if tn + fp else 0.0, 4),
        "f1_score": round(float(f1_score(y_true, predicted, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else 0.5, 4),
        "average_precision": round(float(average_precision_score(y_true, probabilities)), 4),
        "threshold": round(float(threshold), 4),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "labels": ["Stay", "Leave"],
        },
    }


def evaluate_dataset(
    df: pd.DataFrame,
    target_column: str,
    output_dir: str,
    dataset_name: str,
) -> dict[str, Any]:
    """Train and evaluate RF, ANN, and hybrid on exactly one uploaded dataset."""
    if df.empty:
        raise ValueError("Uploaded dataset is empty.")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")
    df = df.dropna(subset=[target_column]).reset_index(drop=True)
    y, target_mapping = _encode_target(df[target_column])
    if len(df) < 20:
        raise ValueError("At least 20 rows are required for a train/test evaluation.")

    X = df.drop(columns=[target_column]).copy()
    drop_columns = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    X = X.drop(columns=drop_columns)
    for column in X.columns:
        if X[column].isna().any():
            if pd.api.types.is_numeric_dtype(X[column]):
                X[column] = X[column].fillna(X[column].median())
            else:
                X[column] = X[column].fillna("Unknown")
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    preprocessor = ColumnTransformer(
        [("num", StandardScaler(), numeric), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)],
        remainder="drop",
    )
    train_idx, test_idx = train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=42, stratify=y
    )
    X_train = preprocessor.fit_transform(X.iloc[train_idx])
    X_test = preprocessor.transform(X.iloc[test_idx])
    y_train, y_test = y[train_idx], y[test_idx]

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=4,
        min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_probability = rf.predict_proba(X_test)[:, 1]

    torch.manual_seed(42)
    ann = AttritionANN(X_train.shape[1])
    optimizer = torch.optim.Adam(ann.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCELoss()
    train_x = torch.tensor(X_train, dtype=torch.float32)
    train_y = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    history = {"epochs": [], "train_loss": [], "validation_loss": [], "train_accuracy": [], "validation_accuracy": []}
    val_split = max(1, int(len(train_x) * 0.2))
    val_x, val_y = train_x[:val_split], train_y[:val_split]
    fit_x, fit_y = train_x[val_split:], train_y[val_split:]
    if len(fit_x) == 0:
        fit_x, fit_y = train_x, train_y
    for epoch in range(1, 41):
        ann.train()
        optimizer.zero_grad()
        loss = criterion(ann(fit_x), fit_y)
        loss.backward()
        optimizer.step()
        ann.eval()
        with torch.no_grad():
            train_pred = (ann(fit_x) >= 0.5).int().numpy().ravel()
            val_probability = ann(val_x).numpy().ravel()
            val_pred = (val_probability >= 0.5).astype(int)
        history["epochs"].append(epoch)
        history["train_loss"].append(round(float(loss.item()), 6))
        history["validation_loss"].append(round(float(criterion(torch.tensor(val_probability).unsqueeze(1), val_y).item()), 6))
        history["train_accuracy"].append(round(float(accuracy_score(fit_y.numpy().ravel(), train_pred)), 6))
        history["validation_accuracy"].append(round(float(accuracy_score(val_y.numpy().ravel(), val_pred)), 6))
    ann_probability = predict_ann_proba(ann, X_test)
    weight_rf = 0.5
    hybrid_probability = compute_hybrid_probability(rf_probability, ann_probability, weight_rf)
    models = {
        "random_forest": _metrics(y_test, rf_probability, 0.5),
        "artificial_neural_network": _metrics(y_test, ann_probability, 0.5),
        "hybrid_fusion": _metrics(y_test, hybrid_probability, 0.5),
    }

    os.makedirs(output_dir, exist_ok=True)
    figure_dir = os.path.join(os.path.dirname(output_dir), "figures")
    os.makedirs(figure_dir, exist_ok=True)
    names = {"random_forest": "Random Forest", "artificial_neural_network": "ANN", "hybrid_fusion": "Hybrid"}
    colors = {"random_forest": "#2563eb", "artificial_neural_network": "#059669", "hybrid_fusion": "#7c3aed"}
    probabilities = {"random_forest": rf_probability, "artificial_neural_network": ann_probability, "hybrid_fusion": hybrid_probability}

    metrics_figure, axis = plt.subplots(figsize=(9, 5))
    metric_names = ["accuracy", "precision", "recall_sensitivity", "f1_score", "roc_auc"]
    for index, metric_name in enumerate(metric_names):
        axis.bar(np.arange(3) + index * 0.15, [models[key][metric_name] for key in models], width=0.15, label=metric_name)
    axis.set_xticks(np.arange(3) + 0.3)
    axis.set_xticklabels(list(names.values()))
    axis.set_ylim(0, 1)
    axis.set_title("Current Dataset: Model Metrics")
    axis.legend(fontsize=8)
    metrics_figure.tight_layout()
    metrics_figure.savefig(os.path.join(figure_dir, "model_metrics.png"), dpi=160)
    plt.close(metrics_figure)

    roc_figure, roc_axis = plt.subplots(figsize=(7, 5))
    pr_figure, pr_axis = plt.subplots(figsize=(7, 5))
    for key, probability in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, probability)
        precision, recall, _ = precision_recall_curve(y_test, probability)
        roc_axis.plot(fpr, tpr, label=f"{names[key]} (AUC {models[key]['roc_auc']:.3f})", color=colors[key])
        pr_axis.plot(recall, precision, label=f"{names[key]} (AP {models[key]['average_precision']:.3f})", color=colors[key])
    roc_axis.plot([0, 1], [0, 1], "k--", alpha=0.4)
    roc_axis.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC Curves")
    pr_axis.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall Curves")
    roc_axis.legend(); pr_axis.legend()
    roc_figure.tight_layout(); pr_figure.tight_layout()
    roc_figure.savefig(os.path.join(figure_dir, "roc_curves_current.png"), dpi=160)
    pr_figure.savefig(os.path.join(figure_dir, "precision_recall_curves_current.png"), dpi=160)
    plt.close(roc_figure); plt.close(pr_figure)

    for key, model_metrics in models.items():
        figure, axis = plt.subplots(figsize=(4, 4))
        sns.heatmap(model_metrics["confusion_matrix"]["matrix"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
        axis.set_title(f"{names[key]} Confusion Matrix")
        axis.set_xlabel("Predicted"); axis.set_ylabel("Actual")
        figure.tight_layout()
        figure.savefig(os.path.join(figure_dir, f"confusion_matrix_{key}.png"), dpi=160)
        plt.close(figure)

    history_figure, history_axes = plt.subplots(1, 2, figsize=(11, 4))
    history_axes[0].plot(history["epochs"], history["train_loss"], label="Train")
    history_axes[0].plot(history["epochs"], history["validation_loss"], label="Validation")
    history_axes[0].set_title("ANN Loss"); history_axes[0].legend()
    history_axes[1].plot(history["epochs"], history["train_accuracy"], label="Train")
    history_axes[1].plot(history["epochs"], history["validation_accuracy"], label="Validation")
    history_axes[1].set_title("ANN Accuracy"); history_axes[1].legend()
    history_figure.tight_layout()
    history_figure.savefig(os.path.join(figure_dir, "ann_training_history_current.png"), dpi=160)
    plt.close(history_figure)

    thresholds = np.linspace(0.05, 0.95, 19)
    threshold_analysis = [{"threshold": round(float(t), 2), "f1_score": round(float(f1_score(y_test, hybrid_probability >= t, zero_division=0)), 4)} for t in thresholds]
    threshold_figure, threshold_axis = plt.subplots(figsize=(7, 4))
    threshold_axis.plot([item["threshold"] for item in threshold_analysis], [item["f1_score"] for item in threshold_analysis])
    threshold_axis.set(xlabel="Decision threshold", ylabel="F1-score", title="Hybrid Threshold Analysis")
    threshold_figure.tight_layout(); threshold_figure.savefig(os.path.join(figure_dir, "threshold_analysis_current.png"), dpi=160); plt.close(threshold_figure)

    feature_names = numeric + (preprocessor.named_transformers_["cat"].get_feature_names_out(categorical).tolist() if categorical else [])
    importance = sorted(
        [{"feature": feature_names[i], "importance": round(float(value), 6)} for i, value in enumerate(rf.feature_importances_)],
        key=lambda item: item["importance"], reverse=True,
    )[:20]
    feature_figure, feature_axis = plt.subplots(figsize=(8, 6))
    feature_axis.barh([item["feature"] for item in importance[::-1]], [item["importance"] for item in importance[::-1]], color="#2563eb")
    feature_axis.set_title("Random Forest Feature Importance"); feature_figure.tight_layout()
    feature_figure.savefig(os.path.join(figure_dir, "feature_importance_current.png"), dpi=160); plt.close(feature_figure)

    result = {
        "status": "success", "dataset_name": dataset_name, "target_column": target_column,
        "target_mapping": target_mapping, "row_count": len(df), "test_sample_count": len(y_test),
        "train_sample_count": len(y_train), "dropped_columns": drop_columns,
        "models": models, "ann_training_history": history, "threshold_analysis": threshold_analysis,
        "feature_importance": importance,
        "summary": {
            "total_records": len(df),
            "test_sample_count": len(y_test),
            "decision_threshold": 0.5,
            "turnover_rate_pct": f"{float(y_test.mean() * 100):.1f}%",
            "models": models,
        },
        "graphs": {key: f"/reports/figures/{value}" for key, value in {
            "model_metrics": "model_metrics.png", "roc": "roc_curves_current.png",
            "precision_recall": "precision_recall_curves_current.png",
            "ann_history": "ann_training_history_current.png", "threshold": "threshold_analysis_current.png",
            "feature_importance": "feature_importance_current.png",
            **{f"confusion_matrix_{key}": f"confusion_matrix_{key}.png" for key in models},
        }.items()},
        "records": [],
    }
    for index, actual, rf_probability_value, ann_probability_value, hybrid_probability_value in zip(
        test_idx, y_test, rf_probability, ann_probability, hybrid_probability
    ):
        probability = float(hybrid_probability_value)
        if probability >= 0.85:
            tier, color = "High Risk", "#dc2626"
        elif probability >= 0.5:
            tier, color = "Moderate Risk", "#d97706"
        else:
            tier, color = "Low Risk", "#16a34a"
        result["records"].append({
            "id": int(index), "actual": int(actual),
            "random_forest_probability": round(float(rf_probability_value), 6),
            "ann_probability": round(float(ann_probability_value), 6),
            "hybrid_probability": round(probability, 6),
            "rf_pct": f"{float(rf_probability_value) * 100:.1f}%",
            "ann_pct": f"{float(ann_probability_value) * 100:.1f}%",
            "hybrid_pct": f"{probability * 100:.1f}%",
            "risk_tier": tier, "risk_color": color,
            "job_role": str(df.iloc[index].get("JobRole", "Dataset record")),
            "department": str(df.iloc[index].get("Department", "")),
            "monthly_income": float(df.iloc[index].get("MonthlyIncome", 0) or 0),
            "overtime": str(df.iloc[index].get("OverTime", "")),
        })
    with open(os.path.join(output_dir, "current_evaluation.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result
