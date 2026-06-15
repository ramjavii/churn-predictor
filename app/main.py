import json
import logging
import sys

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pathlib import Path
from pydantic import BaseModel
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("api")

FEATURE_KEYS = [
    "follower_to_following_ratio",
    "stars_per_repository",
    "fork_to_repo_ratio",
    "open_issues_per_repo",
    "code_to_profile_age_ratio",
    "push_to_total_events_ratio",
    "repos_per_year",
    "days_since_last_api_activity",
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


class ChurnInput(BaseModel):
    follower_to_following_ratio: float
    stars_per_repository: float
    fork_to_repo_ratio: float
    open_issues_per_repo: float
    code_to_profile_age_ratio: float
    push_to_total_events_ratio: float
    repos_per_year: float
    days_since_last_api_activity: float
    average_days_between_events: float
    profile_staleness_days: float
    total_community_validation_count: float
    total_starred_by_user: float
    total_organization_connections: float
    aggregate_codebase_footprint_kb: float
    distinct_event_types_count: float
    has_no_repos: float
    is_b2b_affiliated: float
    has_invested_profile: float
    is_actively_hireable: float
    has_external_gists: float


class ChurnOutput(BaseModel):
    churned: bool
    churn_probability: float


class RecommendInput(BaseModel):
    user_id: int
    top_n: int = 5


app = FastAPI(title="Customer Churn Predictor", version="2.0.0")

model = None
scaler = None
selected_features: list[str] = []
recommender_ready: bool = False
_feature_df = None


@app.on_event("startup")
def on_startup():
    global model, scaler, selected_features, recommender_ready, _feature_df

    model_path = ROOT / "model.pkl"
    scaler_path = ROOT / "scaler.pkl"
    features_path = ROOT / "selected_features.json"

    if not model_path.exists():
        logger.error("model.pkl not found at %s", model_path)
        raise RuntimeError("model.pkl not found. Run 'python app/model.py' first.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    with open(features_path) as f:
        selected_features = json.load(f)

    logger.info("Model loaded. Using %d features: %s", len(selected_features), selected_features)

    try:
        from app.recommender import load_and_init
        from app.features import load_raw_data, compute_features
        raw = load_raw_data()
        _feature_df = compute_features(raw)
        load_and_init()
        recommender_ready = True
        logger.info("Recommender initialized")
    except Exception as e:
        logger.warning("Recommender not available: %s", e)


@app.get("/")
async def root():
    return {
        "app": "Customer Churn Predictor API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict — send 20 feature values, returns churned + churn_probability",
        "recommend": "POST /recommend — user_id + top_n, returns language/activity suggestions for at-risk users",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=ChurnOutput)
async def predict(payload: ChurnInput):
    try:
        input_dict = payload.model_dump()
        row = {k: input_dict[k] for k in selected_features}
        X = pd.DataFrame([row])[selected_features]
        X_scaled = scaler.transform(X)
        prob = float(model.predict_proba(X_scaled)[0, 1])
        churned = bool(model.predict(X_scaled)[0])
        return ChurnOutput(churned=churned, churn_probability=round(prob, 4))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend")
async def recommend_endpoint(payload: RecommendInput):
    if not recommender_ready:
        raise HTTPException(status_code=503, detail="Recommender not initialized.")
    try:
        from app.recommender import recommend as get_recommendations

        if payload.user_id < 0 or payload.user_id >= len(_feature_df):
            raise HTTPException(
                status_code=404,
                detail=f"User {payload.user_id} not found (valid: 0-{len(_feature_df)-1})",
            )

        X = _feature_df[selected_features].iloc[payload.user_id:payload.user_id+1]
        X_scaled = scaler.transform(X)
        prob = float(model.predict_proba(X_scaled)[0, 1])

        result = get_recommendations(payload.user_id, top_n=payload.top_n)
        result["churn_probability"] = round(prob, 4)

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Recommendation failed")
        raise HTTPException(status_code=500, detail=str(e))
