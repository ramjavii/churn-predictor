import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "follower_to_following_ratio",
    "stars_per_repository",
    "fork_to_repo_ratio",
    "open_issues_per_repo",
    "code_to_profile_age_ratio",
    "push_to_total_events_ratio",
    "repos_per_year",
    "days_since_last_api_activity",
    "days_since_last_code_push",
    "average_days_between_events",
    "profile_staleness_days",
    "total_community_validation_count",
    "total_starred_by_user",
    "total_organization_connections",
    "aggregate_codebase_footprint_kb",
    "distinct_event_types_count",
    "has_no_repos",
    "is_b2b_affiliated",
    "has_invested_profile",
    "is_actively_hireable",
    "has_external_gists",
]


def load_raw_data(path: str = "data/raw/github_users.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    date_cols = [
        "created_at",
        "updated_at",
        "most_recent_repo_push",
        "most_recent_event",
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def compute_features(
    df: pd.DataFrame, today: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    if today is None:
        today = pd.Timestamp.now(tz="utc")

    result = pd.DataFrame()
    result["username"] = df["username"]

    created = df["created_at"]
    updated = df["updated_at"]
    most_recent_event = df["most_recent_event"]
    most_recent_push = df["most_recent_repo_push"]

    profile_age_days = (today - created).dt.days.fillna(0).astype(float)
    profile_age_years = profile_age_days / 365.25
    profile_age_days_safe = profile_age_days.replace(0, 1)
    profile_age_years_safe = profile_age_years.replace(0, 0.01)

    followers = df["followers"].fillna(0).astype(float)
    following = df["following"].fillna(0).astype(float)
    public_repos = df["public_repos"].fillna(0).astype(float)
    public_repos_safe = public_repos.replace(0, 1)

    total_stars = df["total_stars_received"].fillna(0).astype(float)
    total_events = df["total_events_fetched"].fillna(0).astype(float)
    total_events_safe = total_events.replace(0, 1)
    push_events = df["push_events"].fillna(0).astype(float)
    total_open_issues = df["total_open_issues"].fillna(0).astype(float)
    total_forked = df["total_repos_forked"].fillna(0).astype(float)

    result["follower_to_following_ratio"] = followers / following.replace(0, 1)
    result["stars_per_repository"] = total_stars / public_repos_safe
    result["fork_to_repo_ratio"] = total_forked / public_repos_safe
    result["open_issues_per_repo"] = total_open_issues / public_repos_safe
    result["code_to_profile_age_ratio"] = public_repos / profile_age_days_safe
    result["push_to_total_events_ratio"] = push_events / total_events_safe
    result["repos_per_year"] = public_repos / profile_age_years_safe

    days_event = (today - most_recent_event).dt.days
    result["days_since_last_api_activity"] = days_event.fillna(9999).astype(float)
    days_push = (today - most_recent_push).dt.days
    result["days_since_last_code_push"] = days_push.fillna(9999).astype(float)
    result["average_days_between_events"] = profile_age_days_safe / total_events_safe
    result["profile_staleness_days"] = (today - updated).dt.days.fillna(0).astype(float)

    result["total_community_validation_count"] = total_stars
    result["total_starred_by_user"] = df["total_starred"].fillna(0).astype(float)
    result["total_organization_connections"] = df["total_orgs"].fillna(0).astype(float)
    result["aggregate_codebase_footprint_kb"] = (
        df["total_repo_size_kb"].fillna(0).astype(float)
    )
    result["distinct_event_types_count"] = (
        df["distinct_event_types_count"].fillna(0).astype(float)
    )

    result["has_no_repos"] = df["has_no_repos"].fillna(True).astype(int)
    result["is_b2b_affiliated"] = df["is_b2b_affiliated"].fillna(False).astype(int)
    result["has_invested_profile"] = df["has_invested_profile"].fillna(False).astype(int)
    result["is_actively_hireable"] = df["is_actively_hireable"].fillna(False).astype(int)
    result["has_external_gists"] = df["has_external_gists"].fillna(False).astype(int)

    result["churned"] = (result["days_since_last_code_push"] > 180).astype(int)

    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    result.fillna(0, inplace=True)

    return result


def select_features(
    df: pd.DataFrame, random_state: int = 42
) -> pd.DataFrame:
    X = df[FEATURE_COLS].copy()
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    n_features = len(FEATURE_COLS)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=FEATURE_COLS, index=X_train.index
    )

    selector_var = VarianceThreshold(threshold=0.01)
    selector_var.fit(X_train_scaled)
    variance_kept = selector_var.get_support()
    logger.info("Variance: %d/%d features kept", sum(variance_kept), n_features)

    corr_matrix = X_train_scaled.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop: set[str] = set()
    for col in upper.columns:
        high_corr = upper.index[upper[col] > 0.9].tolist()
        for correlated in high_corr:
            to_drop.add(correlated)
    correlation_kept = [f not in to_drop for f in FEATURE_COLS]
    logger.info("Correlation: dropped %d redundant features", len(to_drop))

    selector_anova = SelectKBest(score_func=f_classif, k="all")
    selector_anova.fit(X_train_scaled, y_train)
    anova_scores = selector_anova.scores_

    lr = LogisticRegression(max_iter=2000, random_state=random_state)
    rfe = RFE(estimator=lr, n_features_to_select=5)
    rfe.fit(X_train_scaled, y_train)
    rfe_support = rfe.support_

    dt = DecisionTreeClassifier(max_depth=5, random_state=random_state)
    dt.fit(X_train_scaled, y_train)
    dt_importances = dt.feature_importances_

    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=random_state
    )
    rf.fit(X_train_scaled, y_train)
    rf_importances = rf.feature_importances_

    dt_threshold = dt_importances.mean()
    rf_threshold = rf_importances.mean()

    records = []
    for i, name in enumerate(FEATURE_COLS):
        anova_rank = np.argsort(anova_scores)[::-1].tolist().index(i) + 1
        anova_keep = anova_rank <= 10

        consensus = sum([
            variance_kept[i],
            correlation_kept[i],
            anova_keep,
            rfe_support[i],
            dt_importances[i] > dt_threshold,
            rf_importances[i] > rf_threshold,
        ])

        records.append({
            "feature": name,
            "variance": "keep" if variance_kept[i] else "drop",
            "correlation": "keep" if correlation_kept[i] else "drop",
            "anova_rank": anova_rank,
            "anova_f_score": round(float(anova_scores[i]), 4),
            "rfe_selected": bool(rfe_support[i]),
            "dt_importance": round(float(dt_importances[i]), 4),
            "rf_importance": round(float(rf_importances[i]), 4),
            "consensus": consensus,
        })

    comparison = pd.DataFrame(records).sort_values("consensus", ascending=False)
    return comparison


def get_selected_features(
    comparison: pd.DataFrame, min_consensus: int = 3
) -> list[str]:
    return comparison[comparison["consensus"] >= min_consensus]["feature"].tolist()


def print_comparison(comparison: pd.DataFrame) -> None:
    sep = "=" * 105
    header = (
        f"{'Feature':<40} {'Var':<6} {'Corr':<6} {'ANOVA':>6} "
        f"{'RFE':<6} {'DT':>8} {'RF':>8} {'OK':>4}"
    )
    print(sep)
    print(header)
    print("-" * 105)
    for _, row in comparison.iterrows():
        rfe_mark = "\u2713" if row["rfe_selected"] else "\u2717"
        print(
            f"{row['feature']:<40} "
            f"{row['variance']:<6} "
            f"{row['correlation']:<6} "
            f"{row['anova_rank']:>5} "
            f"{rfe_mark:>4}  "
            f"{row['dt_importance']:7.4f} "
            f"{row['rf_importance']:7.4f} "
            f"{row['consensus']:>3}/6"
        )
    print(sep)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    raw = load_raw_data("data/raw/github_users.csv")
    logger.info("Loaded %d users", len(raw))

    features = compute_features(raw)
    churn_pct = features["churned"].mean() * 100
    logger.info("Computed %d features. Churn rate: %.1f%%", len(FEATURE_COLS), churn_pct)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    features.to_csv("data/processed/features.csv", index=False)

    comparison = select_features(features)
    comparison.to_csv("data/processed/selection_comparison.csv", index=False)

    print_comparison(comparison)

    selected = get_selected_features(comparison, min_consensus=3)
    logger.info("Selected (%d/%d): %s", len(selected), len(FEATURE_COLS), selected)
    logger.info("Dropped (%d): %s",
                len(FEATURE_COLS) - len(selected),
                [f for f in FEATURE_COLS if f not in selected])


if __name__ == "__main__":
    main()
