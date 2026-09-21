"""YouTube trailer statistics — free tier, quota-aware.

Mirrors the logic the iOS client uses so both sides agree on what a
trailer number means: trailing-7-day views estimated from the lifetime
count and the publish date, and likes-per-view engagement rather than a
like ratio the API stopped being able to provide when public dislike
counts were removed in 2021.

Quota is the binding constraint. `search.list` costs 100 units against a
10,000/day default; `videos.list` costs 1 regardless of how many ids it
prices. So we resolve each trailer id at most once, persist the map, and
batch every lookup afterwards.
"""
from __future__ import annotations

import datetime as dt
import math
import re

import httpx

API_HOST = "https://www.googleapis.com/youtube/v3"

# Cumulative trailer views modelled as a saturating exponential.
VIEW_CURVE_TAU_DAYS = 21.0

_STOP_WORDS = {"the", "a", "an", "of", "and", "part", "chapter", "movie", "film"}
_DISQUALIFIERS = (
    "reaction", "review", "breakdown", "explained", "concept",
    "fan made", "fan-made", "recut", "parody", "how to", "ranking",
)


def tokens(text: str) -> list[str]:
    return [
        word
        for word in re.split(r"[^a-z0-9]+", text.lower())
        if word and word not in _STOP_WORDS
    ]


def is_plausible_trailer(video_title: str, channel_title: str, movie_title: str) -> bool:
    """Guard against pricing the chain off a reaction video."""
    lowered = video_title.lower()
    if "trailer" not in lowered and "teaser" not in lowered:
        return False
    if any(bad in lowered for bad in _DISQUALIFIERS):
        return False
    wanted = tokens(movie_title)
    if not wanted:
        return False
    found = set(tokens(video_title)) | set(tokens(channel_title))
    hits = sum(1 for word in wanted if word in found)
    return hits / len(wanted) >= 0.6


def trailing_week_fraction(age_days: float, tau: float = VIEW_CURVE_TAU_DAYS) -> float:
    """Share of lifetime views that landed in the trailing 7 days."""
    if age_days <= 7:
        return 1.0
    cumulative = lambda t: 1 - math.exp(-max(0.0, t) / tau)  # noqa: E731
    total = cumulative(age_days)
    if total <= 1e-9:
        return 1.0
    return max(0.0, min(1.0, (total - cumulative(age_days - 7)) / total))


def trailing_week_views(
    lifetime: int, published_at: dt.datetime | None, now: dt.datetime | None = None
) -> int:
    """Estimate trailing-7-day views from a lifetime count."""
    if published_at is None:
        # No publish date: assume the median marketing-trailer age
        # rather than passing a lifetime count off as a weekly one.
        return int(lifetime * trailing_week_fraction(30))
    now = now or dt.datetime.now(dt.timezone.utc)
    age = (now - published_at).total_seconds() / 86400
    return int(lifetime * trailing_week_fraction(age))


def parse_published_at(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_video_stats(payload: dict) -> dict[str, dict]:
    """Map video id -> {views, likes, published_at} from videos.list."""
    out: dict[str, dict] = {}
    for item in payload.get("items") or []:
        video_id = item.get("id")
        if not video_id:
            continue
        stats = item.get("statistics") or {}
        snippet = item.get("snippet") or {}
        # Counts arrive as strings, and a missing count means the
        # uploader hid it — which is not the same as zero.
        views = stats.get("viewCount")
        likes = stats.get("likeCount")
        out[video_id] = {
            "views": int(views) if views is not None else None,
            "likes": int(likes) if likes is not None else None,
            "published_at": parse_published_at(snippet.get("publishedAt")),
        }
    return out


def pick_trailer(payload: dict, movie_title: str) -> str | None:
    """Choose the first search hit that actually looks like the trailer."""
    for item in payload.get("items") or []:
        video_id = (item.get("id") or {}).get("videoId")
        snippet = item.get("snippet") or {}
        if not video_id:
            continue
        if is_plausible_trailer(
            snippet.get("title") or "", snippet.get("channelTitle") or "", movie_title
        ):
            return video_id
    return None


def resolve_trailer_id(client: httpx.Client, api_key: str, movie_title: str) -> str | None:
    """One `search.list` call — 100 quota units. Callers must cache."""
    try:
        response = client.get(
            f"{API_HOST}/search",
            params={
                "key": api_key,
                "part": "snippet",
                "type": "video",
                "maxResults": 5,
                "q": f"{movie_title} official trailer",
            },
            timeout=15,
        )
        if response.status_code != 200:
            return None
        return pick_trailer(response.json(), movie_title)
    except (httpx.HTTPError, ValueError):
        return None


def fetch_stats(client: httpx.Client, api_key: str, video_ids: list[str]) -> dict[str, dict]:
    """Batch up to 50 ids per call — 1 quota unit each."""
    out: dict[str, dict] = {}
    for start in range(0, len(video_ids), 50):
        chunk = video_ids[start : start + 50]
        try:
            response = client.get(
                f"{API_HOST}/videos",
                params={
                    "key": api_key,
                    "part": "statistics,snippet",
                    "id": ",".join(chunk),
                },
                timeout=15,
            )
            if response.status_code != 200:
                break
            out.update(parse_video_stats(response.json()))
        except (httpx.HTTPError, ValueError):
            break
    return out
