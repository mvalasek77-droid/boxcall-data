"""Wikipedia pageview velocity — free attention proxy, no key.

Pageviews on a film's article are one of the better publicly available
leading indicators for opening weekend: they spike on trailer drops,
casting news, and review embargo lifts, and they are not gameable the
way engagement counts are.

The Wikimedia REST API is free and needs no key. Anonymous callers are
rate-limited per IP, which is fine — one run touches a few dozen
articles, well inside the allowance, and it runs a handful of times
a day.
"""
from __future__ import annotations

import datetime as dt
import urllib.parse

import httpx

REST_HOST = "https://wikimedia.org/api/rest_v1"
SEARCH_HOST = "https://en.wikipedia.org/w/api.php"

# Wikimedia asks every client to identify itself. An anonymous or
# spoofed agent is the fastest way to get an IP blocked.
USER_AGENT = "BoxCall/1.0 (https://github.com/mvalasek77-droid/Mike_claw) httpx"


def daily_views_path(article: str, start: dt.date, end: dt.date) -> str:
    """Build the per-article daily pageviews path.

    ``article`` must be the exact Wikipedia title with underscores, and
    it has to be percent-encoded *twice* for titles containing slashes —
    the API takes the title as a single path segment.
    """
    encoded = urllib.parse.quote(article.replace(" ", "_"), safe="")
    return (
        f"{REST_HOST}/metrics/pageviews/per-article/en.wikipedia"
        f"/all-access/user/{encoded}/daily"
        f"/{start.strftime('%Y%m%d')}/{end.strftime('%Y%m%d')}"
    )


def parse_daily_views(payload: dict) -> list[tuple[str, int]]:
    """Extract ``(yyyymmdd, views)`` pairs, oldest first."""
    items = payload.get("items") or []
    out: list[tuple[str, int]] = []
    for item in items:
        timestamp = item.get("timestamp")
        views = item.get("views")
        if timestamp is None or views is None:
            continue
        # Timestamps arrive as YYYYMMDDHH.
        out.append((str(timestamp)[:8], int(views)))
    out.sort(key=lambda pair: pair[0])
    return out


def velocity(series: list[tuple[str, int]]) -> tuple[int, float]:
    """Trailing-week views and the ratio against the week before.

    Returns ``(views_last_7, velocity)`` where velocity is 1.0 for flat
    interest, above 1 for building, below 1 for fading. A movie with no
    prior week to compare against reports 1.0 rather than infinity.
    """
    if not series:
        return 0, 1.0
    counts = [views for _, views in series]
    last7 = sum(counts[-7:])
    prior7 = sum(counts[-14:-7])
    if prior7 <= 0:
        return last7, 1.0
    return last7, round(last7 / prior7, 4)


def resolve_article(client: httpx.Client, title: str, year: int | None) -> str | None:
    """Find the Wikipedia article title for a film.

    Searching rather than guessing matters: the article for a film is
    usually "Title (2027 film)" and sometimes just "Title", and guessing
    wrong silently yields zero views — which the crowd read would then
    have to treat as real.
    """
    query = f"{title} film"
    if year:
        query = f"{title} {year} film"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 1,
        "format": "json",
    }
    try:
        response = client.get(
            SEARCH_HOST, params=params, timeout=15,
            headers={"User-Agent": USER_AGENT},
        )
        if response.status_code != 200:
            return None
        results = response.json().get("query", {}).get("search") or []
        return results[0]["title"] if results else None
    except (httpx.HTTPError, ValueError, KeyError, IndexError):
        return None


def fetch_velocity(
    client: httpx.Client,
    article: str,
    *,
    today: dt.date | None = None,
) -> tuple[int, float]:
    """Trailing-week pageviews and velocity for an article."""
    today = today or dt.date.today()
    # Wikimedia publishes with roughly a day's lag; end yesterday.
    end = today - dt.timedelta(days=1)
    start = end - dt.timedelta(days=14)
    try:
        response = client.get(
            daily_views_path(article, start, end),
            timeout=15,
            headers={"User-Agent": USER_AGENT},
        )
        if response.status_code != 200:
            return 0, 1.0
        return velocity(parse_daily_views(response.json()))
    except (httpx.HTTPError, ValueError):
        return 0, 1.0
