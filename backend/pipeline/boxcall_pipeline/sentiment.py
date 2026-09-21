"""Sentiment scoring for social posts about movies.

Uses VADER, which is a rule-based analyser tuned for exactly this kind
of text — short, slangy, emoji-heavy social posts. It needs no model
download, no API call, and no GPU, which is what makes the whole
pipeline free to run.

Everything here is a pure function of its inputs so it can be tested
without touching the network.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from vaderSentiment.vaderSentiment import BOOSTER_DICT, SentimentIntensityAnalyzer

# VADER does not know film-marketing vocabulary. These are the terms that
# actually move a box-office read, with valences on VADER's -4..+4 scale.
DOMAIN_LEXICON: dict[str, float] = {
    # Bullish
    "masterpiece": 3.4,
    "cinema": 1.2,
    "goated": 3.0,
    "peak": 2.2,
    "banger": 2.8,
    "hyped": 2.2,
    "presales": 1.0,
    "sold": 1.4,
    "soldout": 3.0,
    "imax": 1.2,
    "rewatch": 2.0,
    "chills": 2.4,
    "opening": 0.4,
    "blockbuster": 2.4,
    "certified": 1.6,
    # Bearish
    "flop": -3.2,
    "bomb": -3.0,
    "bombed": -3.2,
    "mid": -1.8,
    "meh": -1.6,
    "snoozefest": -3.0,
    "cashgrab": -3.0,
    "reshoots": -1.8,
    "delayed": -2.0,
    "recast": -1.2,
    "woke": -0.8,
    "unwatchable": -3.4,
    "overrated": -2.0,
    "cgi": -0.6,
    "derivative": -2.0,
    # Words people reach for constantly when reviewing a trailer that
    # VADER's general-purpose lexicon does not score at all. Without
    # these, "this looks incredible" and "totally forgettable" both
    # register as perfectly neutral.
    "incredible": 3.0,
    "phenomenal": 3.2,
    "spectacular": 2.9,
    "riveting": 2.4,
    "gripping": 2.3,
    "immersive": 2.0,
    "touching": 2.0,
    "moving": 1.8,
    "must-see": 3.0,
    "unmissable": 3.0,
    "wild": 1.2,
    # VADER lists "insane" at -1.7. In film talk it is unambiguous
    # praise — "this looks insane" is what people say about a trailer
    # they loved.
    "insane": 2.4,
    "generic": -2.0,
    "forgettable": -2.4,
    "bloated": -2.0,
    "soulless": -3.0,
    "lifeless": -2.8,
    "predictable": -1.8,
    "clunky": -2.0,
    "convoluted": -1.8,
    "tedious": -2.4,
    "cringe": -2.6,
    "cringey": -2.6,
    "pandering": -2.2,
    "rushed": -1.6,
    "uninspired": -2.6,
    "hollow": -1.8,
    "underwhelming": -2.4,
}

# VADER scores these as sentiment-bearing, but in conversational social
# text they are discourse hedges, epistemic markers, or bare degree
# modifiers that carry no opinion of their own. Left alone they inject a
# systematic positive bias, because a hedge prefaces criticism at least
# as often as praise: "honestly it looked mid" scores *positive* on
# stock VADER purely because "honestly" is listed at +2.0, and "pretty
# bad" gets +2.2 from "pretty".
NEUTRALIZE: tuple[str, ...] = (
    "honestly", "truly", "certainly", "definitely", "clearly", "surely",
    "arguably", "seriously", "hopefully", "pretty", "kind", "share",
    "shared", "hell", "damn",
)

def _release_boosters(lexicon: dict[str, float]) -> list[str]:
    """Let our own words carry valence instead of only amplifying.

    `polarity_scores` short-circuits any token found in VADER's
    BOOSTER_DICT, appending a valence of zero *before* it ever consults
    the lexicon. "incredible" ships as a booster, so however we score it,
    "this looks incredible" reads as perfectly neutral.

    BOOSTER_DICT is a module-level global that the analyser reads
    directly — there is no per-instance copy to override — so the only
    way to give these words an opinion is to drop them from it. This
    pipeline is a standalone script and the sole consumer of the library
    in its process, so mutating it here is contained.

    Returns the words actually removed, so the behaviour is testable.
    """
    removed = [word for word in lexicon if word in BOOSTER_DICT]
    for word in removed:
        del BOOSTER_DICT[word]
    return removed


RELEASED_BOOSTERS = _release_boosters(DOMAIN_LEXICON)

_analyzer = SentimentIntensityAnalyzer()
_analyzer.lexicon.update(DOMAIN_LEXICON)
_analyzer.lexicon.update({word: 0.0 for word in NEUTRALIZE})

_URL_RE = re.compile(r"https?://\S+")
_HANDLE_RE = re.compile(r"@[\w.\-]+")
_HASHTAG_RE = re.compile(r"#(\w+)")


def clean(text: str) -> str:
    """Strip the parts of a post that carry no sentiment.

    Links and handles are noise. Hashtags keep their word, because
    ``#masterpiece`` is a real opinion.
    """
    text = _URL_RE.sub(" ", text)
    text = _HANDLE_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(r"\1", text)
    return " ".join(text.split())


def score_text(text: str) -> float:
    """Compound sentiment for one post, in [-1, 1]."""
    cleaned = clean(text)
    if not cleaned:
        return 0.0
    return float(_analyzer.polarity_scores(cleaned)["compound"])


@dataclass(frozen=True)
class SentimentSummary:
    """What the crowd said, condensed to what the market maker needs."""

    score: float  # -1..1, engagement-weighted mean
    dispersion: float  # 0..1, how split the room is
    sample_size: int  # posts actually scored
    positive: int
    negative: int
    neutral: int

    @property
    def is_empty(self) -> bool:
        return self.sample_size == 0


EMPTY = SentimentSummary(0.0, 0.0, 0, 0, 0, 0)

# Posts scoring inside this band are treated as having no opinion.
NEUTRAL_BAND = 0.05


def summarize(posts: list[dict]) -> SentimentSummary:
    """Condense scored posts into one reading.

    Each post may carry ``text`` plus optional ``likes`` and ``reposts``.
    Engagement weighting matters: a post with 900 likes represents far
    more people than one with none, and treating them equally lets a
    handful of low-reach accounts move the market.

    Weights are logarithmic so a single viral post cannot own the score.
    """
    scored: list[tuple[float, float]] = []  # (score, weight)
    positive = negative = neutral = 0

    for post in posts:
        text = post.get("text") or ""
        s = score_text(text)
        engagement = int(post.get("likes", 0)) + 2 * int(post.get("reposts", 0))
        weight = 1.0 + math.log1p(max(0, engagement))
        scored.append((s, weight))
        if s > NEUTRAL_BAND:
            positive += 1
        elif s < -NEUTRAL_BAND:
            negative += 1
        else:
            neutral += 1

    if not scored:
        return EMPTY

    total_weight = sum(w for _, w in scored)
    mean = sum(s * w for s, w in scored) / total_weight

    # Weighted standard deviation, rescaled so that a room split evenly
    # between strong positives and strong negatives reads near 1.0.
    variance = sum(w * (s - mean) ** 2 for s, w in scored) / total_weight
    dispersion = min(1.0, math.sqrt(variance) / 0.6)

    return SentimentSummary(
        score=max(-1.0, min(1.0, mean)),
        dispersion=dispersion,
        sample_size=len(scored),
        positive=positive,
        negative=negative,
        neutral=neutral,
    )
