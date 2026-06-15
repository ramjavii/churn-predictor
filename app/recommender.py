import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

_lang_cols: list[str] = []
_event_cols: list[str] = []
_svd_result: dict = {}
_churn_labels: np.ndarray = np.array([])
_usernames: list[str] = []


def _parse_event_types(raw: str) -> dict:
    try:
        return json.loads(raw.replace("'", '"'))
    except (json.JSONDecodeError, ValueError):
        return {}


def build_matrix(df: pd.DataFrame):
    global _lang_cols, _event_cols

    all_langs: set[str] = set()
    user_langs: list[set[str]] = []
    for langs_str in df["repo_languages"].fillna(""):
        langs = {l.strip() for l in langs_str.split("|") if l.strip()}
        user_langs.append(langs)
        all_langs.update(langs)

    _lang_cols = sorted(all_langs)

    all_events: set[str] = set()
    user_events: list[dict] = []
    for evt_str in df["event_types_json"].fillna("{}"):
        evt_dict = _parse_event_types(evt_str)
        user_events.append(evt_dict)
        all_events.update(evt_dict.keys())

    _event_cols = sorted(all_events)

    n_users = len(df)
    n_cols = len(_lang_cols) + len(_event_cols)
    matrix = np.zeros((n_users, n_cols))

    for i in range(n_users):
        for lang in user_langs[i]:
            matrix[i, _lang_cols.index(lang)] = 1
        for evt, count in user_events[i].items():
            matrix[i, len(_lang_cols) + _event_cols.index(evt)] = count

    logger.info(
        "Built matrix: %d users x %d cols (%d languages + %d event types), sparsity: %.1f%%",
        n_users, n_cols, len(_lang_cols), len(_event_cols),
        (1 - np.count_nonzero(matrix) / matrix.size) * 100,
    )
    return matrix


def fit(matrix: np.ndarray, k: int = 10):
    global _svd_result
    k = min(k, min(matrix.shape) - 1)
    centered = matrix - matrix.mean(axis=0)
    U, sigma, Vt = svds(centered.astype(float), k=k)
    idx = np.argsort(sigma)[::-1]
    U, sigma, Vt = U[:, idx], sigma[idx], Vt[idx, :]
    sigma_diag = np.diag(sigma)
    predicted = U @ sigma_diag @ Vt + matrix.mean(axis=0)
    variance = float(np.sum(sigma**2) / np.sum(centered**2) * 100)

    _svd_result = {"U": U, "sigma": sigma, "Vt": Vt, "predicted_scores": predicted}
    logger.info("SVD fitted: k=%d, explained variance: %.1f%%", k, variance)
    return _svd_result


def init(df: pd.DataFrame, k: int = 10):
    global _churn_labels, _usernames
    _churn_labels = df["churned"].values if "churned" in df.columns else np.zeros(len(df))
    _usernames = df["username"].tolist() if "username" in df.columns else []
    matrix = build_matrix(df)
    return fit(matrix, k=k)


def recommend(user_idx: int, top_n: int = 5) -> dict:
    U = _svd_result["U"]
    predicted = _svd_result["predicted_scores"]

    user_vec = U[user_idx]
    norms = np.linalg.norm(U, axis=1) + 1e-10
    similarities = np.dot(U, user_vec) / (norms * (np.linalg.norm(user_vec) + 1e-10))

    mask = (_churn_labels == 0) & (np.arange(len(U)) != user_idx)
    similarities[~mask] = -1

    top_sim = np.argsort(similarities)[::-1][:top_n * 3]

    n_langs = len(_lang_cols)
    user_row = predicted[user_idx]

    lang_scores: dict[str, float] = {}
    for sim_idx in top_sim:
        if similarities[sim_idx] < 0:
            continue
        for j in range(n_langs):
            lang = _lang_cols[j]
            lang_scores[lang] = lang_scores.get(lang, 0) + float(predicted[sim_idx, j])

    event_scores: dict[str, float] = {}
    for sim_idx in top_sim:
        if similarities[sim_idx] < 0:
            continue
        for j in range(n_langs, len(_lang_cols) + len(_event_cols)):
            evt = _event_cols[j - n_langs]
            event_scores[evt] = event_scores.get(evt, 0) + float(predicted[sim_idx, j])

    user_lang_mask = user_row[:n_langs] > 0
    for j, lang in enumerate(_lang_cols):
        if user_lang_mask[j]:
            lang_scores.pop(lang, None)

    top_langs = sorted(lang_scores.items(), key=lambda x: x[1], reverse=True)
    top_langs = [(l, s) for l, s in top_langs if s > 0][:top_n]

    top_events = sorted(event_scores.items(), key=lambda x: x[1], reverse=True)
    top_events = [(e, s) for e, s in top_events if s > 0][:top_n]

    sim_users = [
        {"user_id": int(i), "similarity": round(float(similarities[i]), 4)}
        for i in top_sim[:3] if similarities[i] > 0
    ]

    insight_parts = []
    if top_langs:
        insight_parts.append(
            f"contribute to {', '.join(l[0] for l in top_langs[:3])} projects"
        )
    if top_events:
        insight_parts.append(
            f"engage in {'/'.join(e[0].replace('Event','') for e in top_events[:2])} activities"
        )
    insight = (
        "Similar retained users " + " and ".join(insight_parts) + "."
        if insight_parts
        else "No strong signals found — this user may need direct outreach."
    )

    username = _usernames[user_idx] if user_idx < len(_usernames) else str(user_idx)

    return {
        "user_id": user_idx,
        "username": username,
        "similar_users": sim_users,
        "language_recommendations": [
            {"language": lang, "score": round(score, 4)} for lang, score in top_langs
        ],
        "activity_recommendations": [
            {"activity": evt, "score": round(score, 4)} for evt, score in top_events
        ],
        "insight": insight,
    }


def load_and_init(raw_csv_path: str = None, k: int = 10):
    from app.features import load_raw_data, compute_features

    raw = load_raw_data(raw_csv_path)
    features = compute_features(raw)
    matrix = build_matrix(raw)
    fit_result = fit(matrix, k=k)

    global _churn_labels, _usernames
    _churn_labels = features["churned"].values
    _usernames = raw["username"].tolist() if "username" in raw.columns else []

    return fit_result
