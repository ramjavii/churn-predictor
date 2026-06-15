import json
import logging
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


def _parse_event_types(raw: str) -> dict:
    try:
        return json.loads(raw.replace("'", '"'))
    except (json.JSONDecodeError, ValueError):
        return {}


def build_co_interest_graph(df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()

    for i, row in df.iterrows():
        G.add_node(i, username=row.get("username", str(i)))

    langs_str = df["repo_languages"].fillna("")
    events_str = df["event_types_json"].fillna("{}")

    user_langs = [
        set(l.strip() for l in s.split("|") if l.strip())
        for s in langs_str
    ]
    user_events = [
        set(_parse_event_types(s).keys()) for s in events_str
    ]

    edges_added = 0
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            shared_langs = user_langs[i] & user_langs[j]
            shared_events = user_events[i] & user_events[j]
            weight = len(shared_langs) + len(shared_events)
            if weight > 0:
                G.add_edge(i, j, weight=weight)
                edges_added += 1

    logger.info(
        "Graph: %d nodes, %d edges, avg degree: %.1f",
        G.number_of_nodes(),
        edges_added,
        np.mean([d for _, d in G.degree()]) if G.number_of_nodes() > 0 else 0,
    )
    return G


def compute_centralities(G: nx.Graph) -> pd.DataFrame:
    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, k=min(100, len(G)))
    pagerank = nx.pagerank(G, max_iter=200)

    results = pd.DataFrame(
        {
            "node": list(degree.keys()),
            "degree_centrality": list(degree.values()),
            "betweenness_centrality": list(betweenness.values()),
            "pagerank": list(pagerank.values()),
        }
    ).set_index("node")

    logger.info(
        "Centralities: degree mean=%.4f, betweenness mean=%.4f, pagerank mean=%.4f",
        results["degree_centrality"].mean(),
        results["betweenness_centrality"].mean(),
        results["pagerank"].mean(),
    )
    return results


def add_network_features(features_df: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(ROOT / "data" / "raw" / "github_users.csv")
    G = build_co_interest_graph(raw)
    centralities = compute_centralities(G)

    result = features_df.copy()
    result["degree_centrality"] = result.index.map(
        lambda i: centralities.loc[i, "degree_centrality"] if i in centralities.index else 0
    )
    result["betweenness_centrality"] = result.index.map(
        lambda i: centralities.loc[i, "betweenness_centrality"] if i in centralities.index else 0
    )
    result["pagerank"] = result.index.map(
        lambda i: centralities.loc[i, "pagerank"] if i in centralities.index else 0
    )
    return result


def get_graph_for_viz() -> nx.Graph:
    raw = pd.read_csv(ROOT / "data" / "raw" / "github_users.csv")
    return build_co_interest_graph(raw)
