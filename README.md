# Customer Churn Predictor

**Final Project — Intro to Data Science — Prof. Yrupe Fresco**

A Dockerized web app that predicts whether a GitHub user will churn (stop being active) and recommends personalized retention actions. Built from real GitHub API data covering 300 user profiles.

## Quick Start

```bash
git clone https://github.com/ramjavii/churn-predictor.git
cd churn-predictor
docker compose up
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

## What it does

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/predict` | POST | Accepts 23 feature values → returns `churned` (bool) + `churn_probability` (0–1) |
| `/recommend` | POST | Accepts `user_id` → returns personalized language and activity suggestions |
| `/docs` | GET | Auto-generated interactive API documentation |

## Architecture

```
GitHub API → scraper.py → raw CSV (300 users)
                ↓
          features.py → 23 features + churn label
                ↓
          features.py → 4 selection methods → 12 features selected
                ↓
          model.py → Random Forest → model.pkl + scaler.pkl
                ↓
          main.py → FastAPI (predict + recommend endpoints)
                ↓
          pca_analysis.py → PCA comparison (original vs. compressed)
          recommender.py → SVD collaborative filtering
          network_analysis.py → Co-interest graph + PageRank features
```

## API Examples

### Predict churn

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "follower_to_following_ratio": 1.5,
    "stars_per_repository": 12.3,
    "fork_to_repo_ratio": 0.2,
    "open_issues_per_repo": 3.1,
    "code_to_profile_age_ratio": 0.05,
    "push_to_total_events_ratio": 0.8,
    "repos_per_year": 4.2,
    "days_since_last_api_activity": 15.0,
    "average_days_between_events": 3.5,
    "profile_staleness_days": 30.0,
    "total_community_validation_count": 150.0,
    "total_starred_by_user": 45.0,
    "total_organization_connections": 2.0,
    "aggregate_codebase_footprint_kb": 50000.0,
    "distinct_event_types_count": 8.0,
    "has_no_repos": 0.0,
    "is_b2b_affiliated": 1.0,
    "has_invested_profile": 1.0,
    "is_actively_hireable": 1.0,
    "has_external_gists": 1.0,
    "degree_centrality": 0.85,
    "betweenness_centrality": 0.0005,
    "pagerank": 0.005
  }'
```

**Response:** `{"churned": false, "churn_probability": 0.35}`

### Get recommendations for at-risk users

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 42, "top_n": 3}'
```

**Response:**
```json
{
  "user_id": 42,
  "username": "sam",
  "churn_probability": 0.97,
  "language_recommendations": [{"language": "Rust", "score": 0.82}],
  "activity_recommendations": [{"activity": "PullRequestEvent", "score": 1.04}],
  "insight": "Similar retained users contribute to Rust projects and engage in PR/Fork activities."
}
```

## Project Structure

```
churn-predictor/
├── app/                         # Production code
│   ├── scraper.py               # GitHub API data fetcher
│   ├── features.py              # Feature generation + 4 selection methods
│   ├── model.py                 # Random Forest training + evaluation
│   ├── main.py                  # FastAPI application
│   ├── pca_analysis.py          # PCA dimensionality reduction
│   ├── recommender.py           # SVD recommendation engine
│   └── network_analysis.py      # Co-interest graph + centrality metrics
├── data/
│   ├── raw/github_users.csv     # 300 scraped user profiles
│   ├── processed/               # Features + selection comparison
│   └── pca/                     # PCA elbow plot + 2D scatter
├── notebooks/
│   └── churn_analysis.ipynb     # Full analysis notebook (63 cells, executed)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── selected_features.json       # 12 features used by the model
```

## Feature Selection — 4 Methods Compared

The project applies all four required methods:

| Method | Implementation | What it found |
|---|---|---|
| **Filter — Variance** | `VarianceThreshold(0.01)` | Dropped `code_to_profile_age_ratio` (near-constant) |
| **Filter — Correlation** | Pearson `|r| > 0.9` | Confirmed same feature is perfectly redundant |
| **Filter — ANOVA** | `SelectKBest(f_classif)` | `pagerank` #1 (F=94.4), time features dominate |
| **Wrapper — RFE** | `RFE(LogisticRegression, k=5)` | Selected `profile_staleness_days`, `pagerank`, etc. |
| **Embedded — DT** | `DecisionTreeClassifier(max_depth=5)` | `push_to_total_events_ratio` at root split |
| **Embedded — RF** | `RandomForest(n=100, balanced)` | `aggregate_codebase_footprint_kb` + `pagerank` lead |

**12 features selected** (consensus ≥ 3/6), 11 dropped.

## Model Performance

| Metric | Value |
|---|---|
| Accuracy | 68.3% |
| Precision | 58.1% |
| Recall | 75.0% |
| F1-score | 65.5% |
| Selected features | 12 / 23 |
| Churn rate | 39.7% |

## Key Extensions

### PCA — Dimensionality Reduction

Compressed 12 features into 9 components (95% variance). Original features perform slightly better (F1 67.9% vs 65.2%) due to Random Forest's natural redundancy handling.

### SVD — Recommendation Engine

Hybrid user-language-activity matrix (300 × 158). Finds similar non-churned users in latent space and recommends languages and activity types to re-engage at-risk users.

### Network Analysis — Feature Enrichment

300-node co-interest graph (38K edges). **PageRank emerged as the #1 ANOVA feature** — confirming that community prestige is the strongest churn predictor. Three centrality metrics added as features: degree, betweenness, and PageRank.

## Tech Stack

- Python 3.11 + FastAPI + Uvicorn
- scikit-learn + pandas + numpy
- networkx (graph analysis)
- joblib (model serialization)
- matplotlib + seaborn (visualization)
- Docker + Docker Compose

## Running Locally (without Docker)

```bash
pip install -r requirements.txt
python app/model.py          # Train model (reads cached CSV, no API needed)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Re-scraping Data

To fetch fresh data from the GitHub API (requires a token):

```bash
# Create .env with your token
echo "GITHUB_TOKEN=ghp_your_token_here" > .env

# Scrape 300 users
python app/scraper.py 300

# Retrain the full pipeline
python app/features.py && python app/model.py
```

## Notebook

`notebooks/churn_analysis.ipynb` contains the complete analysis pipeline with 63 cells across 9 sections:

1. Project Setup
2. Data Pipeline (raw data EDA + histograms)
3. Feature Generation (20 features, churn rate, distributions by class)
4. Feature Selection (all 6 methods + combined consensus table)
5. PCA (elbow plot, 2D scatter, original vs. PCA comparison)
6. Network Analysis (co-interest graph visualization, PageRank analysis)
7. SVD Recommender (matrix factorization, example recommendations)
8. Prediction API (endpoint demo, active vs. inactive user predictions)
9. Conclusions (top insights, method comparison, limitations)

All cells are executed with outputs included in the committed notebook.
