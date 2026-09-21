"""TMDB upcoming releases — the catalog the whole app hangs off.

TMDB's API is free for non-commercial and small-scale use with an
instantly-issued key. It is the only source in this pipeline that needs
any signup at all, and the app degrades to its bundled seed slate
without it.
"""
from __future__ import annotations

import datetime as dt

import httpx

API_HOST = "https://api.themoviedb.org/3"
IMAGE_HOST = "https://image.tmdb.org/t/p/w500"

# Films opening beyond this horizon have no tradeable signal yet.
DEFAULT_WINDOW_DAYS = 90

GENRE_NAMES = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
    80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
    14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
    9648: "Mystery", 10749: "Romance", 878: "Sci-Fi", 53: "Thriller",
    10752: "War", 37: "Western",
}


def parse_release_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def parse_movie(raw: dict, *, today: dt.date) -> dict | None:
    """Normalize one TMDB result into our own shape.

    Returns None for anything unreleased-but-undated or already open —
    a film we cannot date has no expiry, and one that already opened has
    nothing left to predict.
    """
    release = parse_release_date(raw.get("release_date"))
    if release is None or release < today:
        return None

    title = (raw.get("title") or "").strip()
    if not title:
        return None

    poster = raw.get("poster_path")
    genre_ids = raw.get("genre_ids") or []

    return {
        "id": f"tmdb_{raw['id']}",
        "tmdbId": raw["id"],
        "title": title,
        "releaseDate": release.isoformat(),
        "posterURL": f"{IMAGE_HOST}{poster}" if poster else None,
        "overview": (raw.get("overview") or "").strip() or None,
        "genre": GENRE_NAMES.get(genre_ids[0], "Drama") if genre_ids else "Drama",
        "popularity": float(raw.get("popularity") or 0.0),
        "voteAverage": float(raw.get("vote_average") or 0.0),
        "originalLanguage": raw.get("original_language"),
    }


def parse_upcoming(payload: dict, *, today: dt.date, window_days: int) -> list[dict]:
    """Filter and normalize a TMDB /movie/upcoming page."""
    horizon = today + dt.timedelta(days=window_days)
    out: list[dict] = []
    for raw in payload.get("results") or []:
        movie = parse_movie(raw, today=today)
        if movie is None:
            continue
        if dt.date.fromisoformat(movie["releaseDate"]) > horizon:
            continue
        out.append(movie)
    return out


def fetch_upcoming(
    client: httpx.Client,
    api_key: str,
    *,
    pages: int = 3,
    window_days: int = DEFAULT_WINDOW_DAYS,
    today: dt.date | None = None,
) -> list[dict]:
    """Pull the upcoming slate. Returns [] when unconfigured."""
    if not api_key:
        return []
    today = today or dt.date.today()
    seen: dict[str, dict] = {}
    for page in range(1, max(1, pages) + 1):
        try:
            response = client.get(
                f"{API_HOST}/movie/upcoming",
                params={"api_key": api_key, "page": page, "region": "US"},
                timeout=15,
            )
            if response.status_code != 200:
                break
            batch = parse_upcoming(
                response.json(), today=today, window_days=window_days
            )
        except (httpx.HTTPError, ValueError):
            break
        if not batch:
            break
        for movie in batch:
            seen.setdefault(movie["id"], movie)

    return sorted(seen.values(), key=lambda m: m["releaseDate"])
