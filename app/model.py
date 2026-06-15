import json
import logging
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.features import (
    FEATURE_COLS,
    ROOT,
    compute_features,
    load_raw_data,
    get_selected_features,
    select_features,
)

logger = logging.getLogger(__name__)


def train_and_save() -> None:
    raw = load_raw_data()
    df = compute_features(raw)
    logger.info("Loaded %d users, churn rate: %.1f%%", len(df), df["churned"].mean() * 100)

    comparison, _ = select_features(df)
    selected = get_selected_features(comparison, min_consensus=3)
    logger.info("Selected %d features: %s", len(selected), selected)

    X = df[selected].copy()
    y = df["churned"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print()
    print("=" * 45)
    print("  Random Forest — Evaluation (test set)")
    print("-" * 45)
    print(f"  Accuracy:     {acc:.4f}")
    print(f"  Precision:    {prec:.4f}")
    print(f"  Recall:       {rec:.4f}")
    print(f"  F1-score:     {f1:.4f}")
    print()
    print("  Confusion Matrix:")
    print(f"    TN={cm[0][0]:>4}  FP={cm[0][1]:>4}")
    print(f"    FN={cm[1][0]:>4}  TP={cm[1][1]:>4}")
    print("=" * 45)

    top_idx = np.argsort(model.feature_importances_)[::-1]
    print()
    print("  Top 10 feature importances:")
    for rank, i in enumerate(top_idx, start=1):
        print(f"  {rank:>2}. {selected[i]:<40} {model.feature_importances_[i]:.4f}")

    model_path = ROOT / "model.pkl"
    scaler_path = ROOT / "scaler.pkl"
    features_path = ROOT / "selected_features.json"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    with open(features_path, "w") as f:
        json.dump(selected, f, indent=2)

    logger.info("Saved model to %s", model_path)
    logger.info("Saved scaler to %s", scaler_path)
    logger.info("Saved feature list to %s", features_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    train_and_save()
