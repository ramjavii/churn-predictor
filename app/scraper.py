import os
import sys
import time
import logging

import requests
import pandas as pd

from typing import Optional
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
API_BASE = "https://api.github.com"
HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"

PER_PAGE = 100
RATE_LIMIT_SLEEP = 60

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scraper")


def _api_get(url: str) -> Optional[requests.Response]:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code == 404:
        return None
    if resp.status_code in (403, 429):
        logger.warning("Rate limited, sleeping %ds...", RATE_LIMIT_SLEEP)
        time.sleep(RATE_LIMIT_SLEEP)
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code in (403, 429):
            return None
    resp.raise_for_status()
    return resp


def _get_paginated(url: str, max_pages: int = 10) -> list[dict]:
    results = []
    for page in range(1, max_pages + 1):
        full_url = f"{url}?per_page={PER_PAGE}&page={page}" if "?" not in url else f"{url}&per_page={PER_PAGE}&page={page}"
        resp = _api_get(full_url)
        if resp is None:
            break
        data = resp.json()
        if not isinstance(data, list) or not data:
            break
        results.extend(data)
        if len(data) < PER_PAGE:
            break
    return results


def fetch_user_profile(username: str) -> Optional[dict]:
    url = f"{API_BASE}/users/{username}"
    resp = _api_get(url)
    return resp.json() if resp else None


def fetch_user_repos(username: str) -> list[dict]:
    return _get_paginated(f"{API_BASE}/users/{username}/repos")


def fetch_user_events(username: str) -> list[dict]:
    return _get_paginated(f"{API_BASE}/users/{username}/events", max_pages=3)


def fetch_user_starred(username: str) -> list[dict]:
    return _get_paginated(f"{API_BASE}/users/{username}/starred", max_pages=2)


def fetch_user_orgs(username: str) -> list[dict]:
    return _get_paginated(f"{API_BASE}/users/{username}/orgs", max_pages=1)


def collect_seed_usernames(n: int = 300) -> list[str]:
    users = []
    since = 0
    while len(users) < n:
        url = f"{API_BASE}/users?per_page=100&since={since}"
        resp = _api_get(url)
        if resp is None:
            break
        batch = resp.json()
        if not isinstance(batch, list) or not batch:
            break
        users.extend([u["login"] for u in batch if isinstance(u, dict) and "login" in u])
        since = batch[-1].get("id", 0) if batch else 0
        logger.info("Collected %d/%d usernames...", len(users), n)
    return users[:n]


def collect_all_data(usernames: list[str]) -> pd.DataFrame:
    rows = []
    total = len(usernames)

    for i, username in enumerate(usernames):
        logger.info("[%d/%d] %s", i + 1, total, username)

        profile = fetch_user_profile(username)
        if profile is None:
            logger.debug("  -> profile not found, skipping")
            continue

        repos = fetch_user_repos(username)
        stars_received = sum(r.get("stargazers_count", 0) or 0 for r in repos)
        total_forks = sum(r.get("forks_count", 0) or 0 for r in repos)
        total_forked = sum(1 for r in repos if r.get("fork", False))
        total_repo_size = sum(r.get("size", 0) or 0 for r in repos)
        total_open_issues = sum(r.get("open_issues_count", 0) or 0 for r in repos)
        push_dates = [r["pushed_at"] for r in repos if r.get("pushed_at")]
        most_recent_push = max(push_dates) if push_dates else None
        languages = [r.get("language") for r in repos if r.get("language")]

        events = fetch_user_events(username)
        event_types: dict[str, int] = {}
        for e in events:
            t = e.get("type", "Unknown") or "Unknown"
            event_types[t] = event_types.get(t, 0) + 1
        push_events_count = event_types.get("PushEvent", 0)
        total_events = len(events)
        event_dates = [e["created_at"] for e in events if e.get("created_at")]
        most_recent_event = max(event_dates) if event_dates else None

        starred = fetch_user_starred(username)
        orgs = fetch_user_orgs(username)

        profile_fields = ["bio", "name", "company", "blog", "location"]
        has_invested_profile = all(
            profile.get(f) is not None and str(profile.get(f, "")).strip() != ""
            for f in profile_fields
        )

        row = {
            "username": username,
            "id": profile.get("id"),
            "followers": profile.get("followers", 0) or 0,
            "following": profile.get("following", 0) or 0,
            "public_repos": profile.get("public_repos", 0) or 0,
            "public_gists": profile.get("public_gists", 0) or 0,
            "hireable": profile.get("hireable"),
            "created_at": profile.get("created_at"),
            "updated_at": profile.get("updated_at"),
            "bio": profile.get("bio", "") or "",
            "name": profile.get("name", "") or "",
            "company": profile.get("company", "") or "",
            "blog": profile.get("blog", "") or "",
            "location": profile.get("location", "") or "",
            "email": profile.get("email", "") or "",
            "disk_usage": profile.get("disk_usage", 0) or 0,
            "type": profile.get("type", "User"),
            "site_admin": profile.get("site_admin", False),
            "total_repos_fetched": len(repos),
            "total_stars_received": stars_received,
            "total_forks": total_forks,
            "total_repos_forked": total_forked,
            "total_repo_size_kb": total_repo_size,
            "total_open_issues": total_open_issues,
            "most_recent_repo_push": most_recent_push,
            "repo_languages": "|".join(sorted(set(languages) - {None})) if languages else "",
            "total_events_fetched": total_events,
            "push_events": push_events_count,
            "most_recent_event": most_recent_event,
            "distinct_event_types_count": len(event_types),
            "event_types_json": str(event_types),
            "total_starred": len(starred),
            "total_orgs": len(orgs),
            "has_invested_profile": has_invested_profile,
            "has_no_repos": (profile.get("public_repos", 0) or 0) == 0,
            "is_b2b_affiliated": len(orgs) > 0,
            "is_actively_hireable": profile.get("hireable") is True,
            "has_external_gists": (profile.get("public_gists", 0) or 0) > 0,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    if not GITHUB_TOKEN:
        logger.warning("No GITHUB_TOKEN set. Rate limit: 60 requests/hour.")
        logger.warning("Set it in .env file: GITHUB_TOKEN=ghp_...")

    logger.info("Collecting %d seed usernames...", n)
    usernames = collect_seed_usernames(n)
    logger.info("Got %d usernames. Fetching full data...", len(usernames))

    df = collect_all_data(usernames)
    logger.info("Collected data for %d users.", len(df))

    os.makedirs("data/raw", exist_ok=True)
    output_path = "data/raw/github_users.csv"
    df.to_csv(output_path, index=False)
    logger.info("Saved to %s (%d rows, %d columns)", output_path, len(df), len(df.columns))
    return df


if __name__ == "__main__":
    main()
