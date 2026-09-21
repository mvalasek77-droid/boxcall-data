"""Fetch actual domestic opening-weekend grosses from free public sources.

Three independent reads, cross-checked. Publishing a wrong settlement
number is the worst failure this pipeline can produce — a user trades
against a consensus of $45M, the studio reports $60M, and the app pays
out on a typo. So no single source is ever trusted alone:

  1. Box Office Mojo weekend chart (latest weekend, all titles).
  2. The Numbers weekend chart (latest weekend, all titles).
  3. Box Office Mojo per-release pages (the "Opening" figure for a
     specific film — a different BOM page, fetched on demand).

When two sources agree within tolerance the primary's number is
published. When they disagree, nothing is published for that title —
the settlement holds until the next pipeline run re-checks.

Holdover guard: a film's SECOND weekend gross must never be recorded
as its opening. BOM weekend rows carry a weeks-in-release column and
TN weekend rows carry a "(new)" marker; both parsers use them.

URL note: both sites' un-dated "current" pages are CALENDAR INDEXES,
not charts. The real charts are dated by weekend:
  BOM:   /weekend/{isoYear}W{isoWeek:02d}/
  TN:    /box-office-chart/weekend/{YYYY}/{MM}/{DD-of-Friday}
so the fetchers compute the most recent weekend's Friday and build
the dated URL from it.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

import httpx

# Tolerance for cross-source agreement, as a fraction of the larger
# number. Weekend estimates vs. finals legitimately differ by ~1-2%,
# so anything within 2% is the same figure.
AGREEMENT_TOLERANCE = 0.02

USER_AGENT = (
    "BoxCallBot/1.0 (https://github.com/mvalasek77-droid/Mike_claw; "
    "movie box-office data for a play-money trading game)"
)


@dataclass
class OpeningResult:
    title: str
    gross_millions: float
    is_opening_weekend: bool
    source: str


def _normalize(title: str) -> str:
    """Case/punctuation-insensitive title key for cross-source matching."""
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def latest_weekend_friday(today: dt.date | None = None) -> dt.date:
    """The Friday of the most recent weekend (possibly still running).

    On Sun Sep 20, 2026 → Sep 18. On Mon Sep 21 → still Sep 18: the
    weekend's chart holds the finals through Monday. Friday itself →
    that Friday: the chart exists with partial-day estimates.
    """
    today = today or dt.datetime.now(dt.timezone.utc).date()
    return today - dt.timedelta(days=(today.weekday() - 4) % 7)


def _bom_weekend_url(friday: dt.date) -> str:
    iso = friday.isocalendar()
    return f"https://www.boxofficemojo.com/weekend/{iso[0]}W{iso[1]:02d}/"


def _tn_weekend_url(friday: dt.date) -> str:
    return (
        f"https://www.the-numbers.com/box-office-chart/weekend/"
        f"{friday.year}/{friday.month:02d}/{friday.day:02d}"
    )


def _get(client: httpx.Client, url: str) -> str | None:
    try:
        resp = client.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        if resp.status_code == 200:
            return resp.text
    except Exception:
        pass
    return None


def fetch_actuals(client: httpx.Client) -> list[OpeningResult]:
    """BOM weekend chart first; The Numbers as fallback.

    Kept for backward compatibility and for callers that only need a
    single-source read (e.g. status reporting). Settlement data goes
    through `fetch_cross_checked` instead.
    """
    results = _fetch_bom_weekend(client)
    if results:
        return results
    return _fetch_tn_weekend(client)


def fetch_cross_checked(client: httpx.Client, titles: list[str]) -> list[OpeningResult]:
    """Cross-check every wanted title against both weekend charts.

    Returns the subset of `titles` whose numbers agree across sources,
    labelled with the primary (BOM) figure and a combined source tag.
    Titles the charts don't carry, or on which the sources disagree,
    are simply absent — the settlement layer treats absence as
    "keep waiting", never as zero.
    """
    bom = _fetch_bom_weekend(client)
    tn = _fetch_tn_weekend(client)

    bom_by_title = {_normalize(r.title): r for r in bom}
    tn_by_title = {_normalize(r.title): r for r in tn}

    out: list[OpeningResult] = []
    for wanted in titles:
        key = _normalize(wanted)
        b = bom_by_title.get(key)
        t = tn_by_title.get(key)
        if b is None or t is None:
            continue
        if not (b.is_opening_weekend and t.is_opening_weekend):
            continue
        scale = max(b.gross_millions, t.gross_millions)
        if scale <= 0:
            continue
        if abs(b.gross_millions - t.gross_millions) / scale <= AGREEMENT_TOLERANCE:
            out.append(OpeningResult(
                title=b.title,
                gross_millions=round(b.gross_millions, 2),
                is_opening_weekend=True,
                source="boxofficemojo+the-numbers",
            ))
    return out


def fetch_release_opening(client: httpx.Client, title: str, release_id: str | None = None) -> OpeningResult | None:
    """Read a film's Opening figure from its BOM release page.

    A different BOM page from the weekend chart, so agreement between
    the two is real corroboration. Without a known `release_id` the
    title's /release/ link is found on the latest weekend chart.
    """
    if release_id:
        html = _get(client, f"https://www.boxofficemojo.com/release/{release_id}/")
        if html:
            got = _parse_release_opening(html, title)
            if got:
                return got
    chart = _get(client, _bom_weekend_url(latest_weekend_friday()))
    if not chart:
        return None
    # Chart release links carry query strings: /release/rl…/?ref_=…
    for m in re.finditer(r'href="/release/(rl\d+)[/?][^"]*"[^>]*>\s*([^<]+?)\s*<', chart):
        if _normalize(m.group(2)) == _normalize(title):
            got = fetch_release_opening(client, title, release_id=m.group(1))
            if got:
                return got
    return None


# ----------------------------------------------------------------------
# BOM weekend chart
# ----------------------------------------------------------------------

def _fetch_bom_weekend(client: httpx.Client) -> list[OpeningResult]:
    html = _get(client, _bom_weekend_url(latest_weekend_friday()))
    if not html:
        return []
    return _parse_bom(html)


def _parse_bom(html: str) -> list[OpeningResult]:
    """Parse BOM's weekend chart.

    BOM's layout: Rank | Last-week rank | Release | Gross | Change ...
    The columns BEFORE the title vary (weekend charts have a
    "previous rank" cell), so fixed indices silently read the wrong
    cell. Instead: find the release link cell, then the FIRST
    money-formatted cell after it. Weeks-in-release is a small integer
    cell near the row's end; opening weekend ⇔ weeks == 1.
    """
    results: list[OpeningResult] = []

    rows = re.findall(
        r'<tr[^>]*>(.*?)</tr>',
        html,
        re.DOTALL | re.IGNORECASE,
    )

    for row in rows:
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL | re.IGNORECASE)
        if len(cells) < 4:
            continue

        # Title = the cell whose link points at /release/ (the movie).
        title: str | None = None
        title_idx = -1
        for i, cell in enumerate(cells):
            m = re.search(r'href="/release/[^"]*"[^>]*>\s*([^<]+?)\s*<', cell)
            if m and m.group(1).strip():
                title = m.group(1).strip()
                title_idx = i
                break
        if not title or title.lower() in ("release", "title"):
            continue

        # Gross = first money cell AFTER the title cell.
        gross = None
        for cell in cells[title_idx + 1:]:
            gross = _parse_gross(re.sub(r'<[^>]+>', '', cell).strip())
            if gross is not None:
                break
        if gross is None or gross <= 0:
            continue

        weeks = _parse_weeks(cells)

        results.append(OpeningResult(
            title=title,
            gross_millions=round(gross / 1_000_000, 2),
            is_opening_weekend=weeks == 1,
            source="boxofficemojo",
        ))

    return results


def _parse_weeks(cells: list[str]) -> int | None:
    """Weeks-in-release from a BOM row's trailing cells.

    The column sits near the row's end (after the theater counts and
    averages, before the studio/distributor cells); scanning backwards
    for the last plain small integer is stable across BOM's layout
    variations. Returns None when no such cell exists — the caller
    treats that as "not proven to be an opening weekend".
    """
    for cell in reversed(cells):
        text = re.sub(r'<[^>]+>', '', cell).strip()
        if re.fullmatch(r'[1-9]\d{0,2}', text):
            return int(text)
    return None


def _parse_release_opening(html: str, title: str) -> OpeningResult | None:
    """Extract the Opening figure from a BOM /release/ page.

    Layout (Sep 2026): a summary table with label/value rows, e.g.
    'Opening' -> '$60,000000' style money, 'Release Date' -> date,
    'MPAA', 'Running Time', ... The Opening row's money cell sits
    immediately after the 'Opening' label cell.
    """
    m = re.search(
        r'>Opening<.*?>(\$[\d,]+)<',
        html,
        re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return None
    gross = _parse_gross(m.group(1))
    if gross is None or gross <= 0:
        return None
    return OpeningResult(
        title=title,
        gross_millions=round(gross / 1_000_000, 2),
        is_opening_weekend=True,
        source="boxofficemojo-release",
    )


# ----------------------------------------------------------------------
# The Numbers weekend chart
# ----------------------------------------------------------------------

def _fetch_tn_weekend(client: httpx.Client) -> list[OpeningResult]:
    html = _get(client, _tn_weekend_url(latest_weekend_friday()))
    if not html:
        return []
    return _parse_the_numbers(html)


def _parse_the_numbers(html: str) -> list[OpeningResult]:
    """Parse The Numbers' weekend chart.

    The /box-office page carries MORE than one movie table: the
    weekend chart itself, followed by 'Top 5' daily/total tables that
    reuse /movie/ links with TOTAL grosses. Picking the wrong table
    publishes a cumulative gross as an opening weekend. The weekend
    chart is the first table under the 'Weekend Domestic Box Office'
    heading — anchor to it and parse only that table.

    Observed row shape (Sep 2026):
      <td class="data">1</td>            rank
      <td class="data">(new)</td>        prev rank — '(new)' = opening
      <td><b><a href="/movie/...">Resident Evil</a></b></td>
      <td class="data chart_estimate">$60,000,000</td>   weekend gross
      ...
    Title = the cell whose link points at /movie/; gross = the first
    money cell after it; opening = the prev-rank '(new)' marker.
    """
    # Slice the page from the weekend heading onward, stopping at the
    # next chart heading ('Top Movies' etc.) so secondary tables never
    # leak in.
    anchor = re.search(
        r'Weekend\s+Domestic\s+Box\s+Office',
        html,
        re.IGNORECASE,
    )
    if not anchor:
        return []
    segment = html[anchor.end():]
    # The weekend chart is the FIRST <table> after the heading.
    tbl = re.search(r'<table[^>]*>(.*?)</table>', segment, re.DOTALL | re.IGNORECASE)
    if not tbl:
        return []
    table_html = tbl.group(1)

    results: list[OpeningResult] = []
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, re.DOTALL | re.IGNORECASE)

    for row in rows:
        if "/movie/" not in row:
            continue
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL | re.IGNORECASE)
        if len(cells) < 3:
            continue

        title: str | None = None
        title_idx = -1
        for i, cell in enumerate(cells):
            m = re.search(r'href="/movie/[^"]*"[^>]*>\s*([^<]+?)\s*<', cell)
            if m and m.group(1).strip():
                title = m.group(1).strip()
                title_idx = i
                break
        if not title:
            continue

        gross = None
        for cell in cells[title_idx + 1:]:
            gross = _parse_gross(re.sub(r'<[^>]+>', '', cell).strip())
            if gross is not None:
                break
        if gross is None or gross <= 0:
            continue

        is_opening = "(new)" in row

        results.append(OpeningResult(
            title=title,
            gross_millions=round(gross / 1_000_000, 2),
            is_opening_weekend=is_opening,
            source="the-numbers",
        ))

    return results


def _parse_gross(text: str) -> float | None:
    """'$60,000,000' -> 60000000.0. Anything else -> None.

    A leading '$' is REQUIRED. This is deliberate: the weekend charts
    are full of bare numbers that are NOT grosses (theater counts,
    weeks-in-release, ranks), and accepting bare integers let a
    theater count parse as a $0.000037M "gross". Money cells on BOM
    and TN are always '$'-prefixed.
    """
    text = text.strip()
    if not text.startswith("$"):
        return None
    cleaned = re.sub(r"[^0-9.]", "", text)
    if not cleaned or ("." not in text and cleaned.count(".") > 1):
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value if value > 0 else None