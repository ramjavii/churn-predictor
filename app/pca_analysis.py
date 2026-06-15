import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.features import compute_features, load_raw_data, get_selected_features, select_features

logger = logging.getLogger(__name__)

OUTPUT_DIR = ROOT / "data" / "pca"


def run_pca(random_state: int = 42):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data()
    df = compute_features(raw)
    logger.info("Loaded %d users, churn rate: %.1f%%", len(df), df["churned"].mean() * 100)

    comparison, _ = select_features(df)
    selected = get_selected_features(comparison, min_consensus=3)
    logger.info("Selected %d features: %s", len(selected), selected)

    X = df[selected].copy()
    y = df["churned"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_all_scaled = scaler.transform(X)

    # ---- PCA: fit on training data ----
    pca_full = PCA()
    pca_full.fit(X_train_scaled)

    # ---- Elbow plot ----
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cumsum) + 1), cumsum, "bo-", markersize=6)
    ax.axhline(y=0.90, color="gray", linestyle="--", alpha=0.5, label="90% variance")
    ax.axhline(y=0.95, color="gray", linestyle="-.", alpha=0.5, label="95% variance")

    # Mark elbow point
    n_comps = next(i + 1 for i, v in enumerate(cumsum) if v >= 0.95)
    ax.axvline(x=n_comps, color="red", linestyle=":", alpha=0.7, label=f"95% at {n_comps} comps")
    ax.set_xlabel("Number of Components", fontsize=12)
    ax.set_ylabel("Cumulative Explained Variance", fontsize=12)
    ax.set_title("PCA Elbow Plot — 95% Variance at {} Components".format(n_comps), fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pca_elbow.png", dpi=150)
    plt.close(fig)
    logger.info("Elbow plot saved (%d components for 95%% variance)", n_comps)

    # ---- 2D scatter ----
    pca2 = PCA(n_components=2)
    X_2d = pca2.fit_transform(X_all_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    for c, lab, col in [(0, "Active", "#2ecc71"), (1, "Churned", "#e74c3c")]:
        mask = y == c
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=col, label=lab, alpha=0.5, s=30)
    ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.1%} var)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]:.1%} var)", fontsize=12)
    ax.set_title("Users in PCA Space (2D projection)", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pca_scatter.png", dpi=150)
    plt.close(fig)
    logger.info("2D scatter saved (PC1: %.1f%%, PC2: %.1f%%)",
                pca2.explained_variance_ratio_[0] * 100,
                pca2.explained_variance_ratio_[1] * 100)

    # ---- Model comparison: Original vs PCA ----
    # Original model
    rf_orig = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=random_state
    )
    rf_orig.fit(X_train_scaled, y_train)
    y_pred_orig = rf_orig.predict(X_test_scaled)
    metrics_orig = {
        "accuracy": accuracy_score(y_test, y_pred_orig),
        "precision": precision_score(y_test, y_pred_orig),
        "recall": recall_score(y_test, y_pred_orig),
        "f1": f1_score(y_test, y_pred_orig),
    }

    # PCA model — use n_comps that captures 95% variance
    pca_n = PCA(n_components=n_comps)
    X_train_pca = pca_n.fit_transform(X_train_scaled)
    X_test_pca = pca_n.transform(X_test_scaled)

    rf_pca = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=random_state
    )
    rf_pca.fit(X_train_pca, y_train)
    y_pred_pca = rf_pca.predict(X_test_pca)
    metrics_pca = {
        "accuracy": accuracy_score(y_test, y_pred_pca),
        "precision": precision_score(y_test, y_pred_pca),
        "recall": recall_score(y_test, y_pred_pca),
        "f1": f1_score(y_test, y_pred_pca),
    }

    # ---- Print comparison ----
    print()
    print("=" * 65)
    print("  PCA — Model Performance Comparison")
    print("-" * 65)
    print(f"  {'Metric':<16} {'Original (11 feat)':>18} {'PCA ({} comp)'.format(n_comps):>18}")
    print("-" * 65)
    for k in ["accuracy", "precision", "recall", "f1"]:
        print(f"  {k.capitalize():<16} {metrics_orig[k]:18.4f} {metrics_pca[k]:18.4f}")
    print("-" * 65)
    diff_acc = metrics_orig["accuracy"] - metrics_pca["accuracy"]
    direction = "better" if diff_acc > 0 else "worse"
    if abs(diff_acc) < 0.01:
        direction = "equal"
    print(f"  Accuracy diff: {diff_acc:+.4f} (PCA is {direction})")
    print(f"  Components:  {len(selected)} → {n_comps} ({n_comps/len(selected)*100:.0f}%)")
    print("=" * 65)

    # ---- Per-component variance ----
    print()
    print("  Per-component explained variance:")
    for i in range(n_comps):
        pct = pca_full.explained_variance_ratio_[i] * 100
        bar = "█" * int(pct)
        print(f"  PC{i + 1:>2}: {pct:5.1f}% {bar}")

    # ---- Save PCA model artifacts ----
    joblib.dump(rf_pca, ROOT / "model_pca.pkl")
    joblib.dump(pca_n, ROOT / "pca_model.pkl")
    joblib.dump(scaler, ROOT / "scaler_pca.pkl")
    logger.info("Saved model_pca.pkl, pca_model.pkl, scaler_pca.pkl")

    # ---- Answer the key question ----
    print()
    print("=" * 65)
    print("  ANSWER: Training on PCA components performs")
    if metrics_pca["f1"] >= metrics_orig["f1"] * 0.95:
        print(f"  COMPARABLY to original features ({metrics_pca['f1']:.3f} vs {metrics_orig['f1']:.3f} F1).")
        print(f"  The 95%% variance is captured in only {n_comps} components.")
        print(f"  PCA provides massive dimensionality reduction with minimal loss.")
    else:
        print(f"  WORSE than original features ({metrics_pca['f1']:.3f} vs {metrics_orig['f1']:.3f} F1).")
        print(f"  The information lost when compressing {len(selected)} features")
        print(f"  into {n_comps} components is significant enough to hurt performance.")
        print(f"  The original features capture non-linear relationships that PCA cannot.")
    print("=" * 65)

    return metrics_orig, metrics_pca, n_comps


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    run_pca()
