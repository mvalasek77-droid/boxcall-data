"""Tests for the free data pipeline.

Every parser is exercised against a recorded response shape, so the
whole suite runs with no network and no API keys.
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from boxcall_pipeline import sentiment  # noqa: E402
from boxcall_pipeline.sources import bluesky, boxoffice, tmdb, wikipedia, youtube  # noqa: E402

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 9, 8, 12, 0, tzinfo=UTC)


# --------------------------------------------------------------------
# Sentiment
# --------------------------------------------------------------------

class TestSentiment:
    def test_clean_strips_links_and_handles_but_keeps_hashtag_words(self):
        got = sentiment.clean("@someone this is https://x.co/abc #masterpiece truly")
        assert "someone" not in got
        assert "https" not in got
        assert "masterpiece" in got

    def test_domain_lexicon_reads_film_slang(self):
        # Plain VADER has no idea these are opinions about a movie.
        assert sentiment.score_text("absolute banger, peak cinema") > 0.3
        assert sentiment.score_text("this is going to flop, total cashgrab") < -0.3

    def test_mid_is_negative_in_film_context(self):
        assert sentiment.score_text("honestly it looked mid") < 0

    def test_discourse_hedges_do_not_read_as_praise(self):
        """Stock VADER lists 'honestly' at +2.0 and 'pretty' at +2.2, so
        a hedged criticism scored positive. That is a systematic upward
        bias, because hedges preface complaints at least as often as
        compliments."""
        for phrase in [
            "honestly it looked mid",
            "pretty bad honestly",
            "seriously this looks like a flop",
            "arguably the worst trailer this year",
        ]:
            assert sentiment.score_text(phrase) < 0, phrase

    def test_neutralised_words_carry_no_opinion_alone(self):
        for word in ["honestly", "pretty", "clearly", "definitely"]:
            assert sentiment.score_text(word) == 0.0, word

    def test_genuine_praise_still_registers(self):
        """Neutralising hedges must not flatten real sentiment."""
        assert sentiment.score_text("honestly this looks incredible") > 0.3
        assert sentiment.score_text("pretty amazing trailer") > 0.3

    def test_trailer_vocabulary_vader_does_not_know(self):
        """Stock VADER scores none of these — they are absent from its
        lexicon entirely, so a whole register of trailer criticism read
        as perfectly neutral."""
        for phrase in ["completely forgettable", "soulless and bloated",
                       "totally uninspired", "so predictable", "pure cringe"]:
            assert sentiment.score_text(phrase) < 0, phrase
        for phrase in ["absolutely phenomenal", "gripping and immersive",
                       "this is a must-see", "spectacular stuff"]:
            assert sentiment.score_text(phrase) > 0, phrase

    def test_booster_words_were_released_so_they_can_carry_valence(self):
        """VADER zeroes any token in BOOSTER_DICT before reading the
        lexicon, so a word shipped as a booster can never hold an
        opinion no matter what we score it."""
        from vaderSentiment.vaderSentiment import BOOSTER_DICT

        assert "incredible" in sentiment.RELEASED_BOOSTERS
        assert "incredible" not in BOOSTER_DICT
        assert sentiment.score_text("this looks incredible") > 0.3

    def test_insane_is_praise_in_film_talk(self):
        """VADER lists it at -1.7; "this looks insane" is a rave."""
        assert sentiment.score_text("that trailer looks insane") > 0

    def test_empty_input_is_neutral_not_an_error(self):
        assert sentiment.score_text("") == 0.0
        assert sentiment.summarize([]) is sentiment.EMPTY
        assert sentiment.summarize([]).is_empty

    def test_summary_direction(self):
        positive = sentiment.summarize(
            [{"text": "masterpiece, best thing all year"}, {"text": "absolutely loved it"}]
        )
        negative = sentiment.summarize(
            [{"text": "total flop incoming"}, {"text": "unwatchable garbage"}]
        )
        assert positive.score > 0.2
        assert negative.score < -0.2
        assert positive.positive == 2
        assert negative.negative == 2

    def test_split_room_reads_as_dispersed(self):
        united = sentiment.summarize([{"text": "masterpiece"}] * 6)
        split = sentiment.summarize(
            [{"text": "masterpiece"}, {"text": "unwatchable"}] * 3
        )
        assert split.dispersion > united.dispersion
        assert abs(split.score) < abs(united.score)

    def test_engagement_weighting_favours_reach(self):
        """A post nobody saw should not outvote one thousands liked."""
        loud_positive = sentiment.summarize(
            [
                {"text": "masterpiece, incredible", "likes": 5000},
                {"text": "unwatchable garbage", "likes": 0},
            ]
        )
        assert loud_positive.score > 0

    def test_viral_post_cannot_fully_own_the_score(self):
        """Log weighting caps how much one post can dominate."""
        with_viral = sentiment.summarize(
            [
                {"text": "masterpiece", "likes": 10_000_000},
                {"text": "unwatchable"},
                {"text": "unwatchable"},
                {"text": "unwatchable"},
            ]
        )
        only_viral = sentiment.summarize(
            [{"text": "masterpiece", "likes": 10_000_000}]
        )
        assert with_viral.score < only_viral.score

    def test_scores_stay_in_range(self):
        summary = sentiment.summarize(
            [{"text": "masterpiece " * 50, "likes": 10**9}] * 20
        )
        assert -1.0 <= summary.score <= 1.0
        assert 0.0 <= summary.dispersion <= 1.0


# --------------------------------------------------------------------
# Bluesky
# --------------------------------------------------------------------

SEARCH_FIXTURE = {
    "posts": [
        {
            "uri": "at://did:plc:aaa/app.bsky.feed.post/1",
            "likeCount": 120,
            "repostCount": 14,
            "replyCount": 3,
            "record": {"text": "Dune Part Three trailer is peak cinema",
                       "createdAt": "2026-09-08T09:00:00.000Z"},
        },
        {
            "uri": "at://did:plc:bbb/app.bsky.feed.post/2",
            "likeCount": 2,
            "repostCount": 0,
            "record": {"text": "not sure about this one honestly",
                       "createdAt": "2026-09-08T11:30:00Z"},
        },
        {   # Outside the 24h window — must be dropped.
            "uri": "at://did:plc:ccc/app.bsky.feed.post/3",
            "likeCount": 900,
            "record": {"text": "old chatter from last month",
                       "createdAt": "2026-08-01T09:00:00Z"},
        },
        {   # No text — must be skipped, not crash.
            "uri": "at://did:plc:ddd/app.bsky.feed.post/4",
            "record": {"createdAt": "2026-09-08T10:00:00Z"},
        },
    ]
}


class TestBluesky:
    def test_query_is_quoted_so_words_stay_adjacent(self):
        assert bluesky.build_query("Dune: Part Three") == '"Dune Part Three"'

    def test_query_survives_punctuation_and_empties(self):
        assert bluesky.build_query("Wicked: For Good") == '"Wicked For Good"'
        assert bluesky.build_query("   ") == ""
        assert bluesky.build_query("!!!") == ""

    def test_parse_keeps_recent_posts_only(self):
        since = NOW - dt.timedelta(hours=24)
        posts = bluesky.parse_search_response(SEARCH_FIXTURE, since=since)
        texts = [p["text"] for p in posts]
        assert len(posts) == 2
        assert "old chatter from last month" not in texts

    def test_parse_extracts_engagement(self):
        since = NOW - dt.timedelta(hours=24)
        posts = bluesky.parse_search_response(SEARCH_FIXTURE, since=since)
        assert posts[0]["likes"] == 120
        assert posts[0]["reposts"] == 14

    def test_parse_tolerates_missing_and_malformed_fields(self):
        since = NOW - dt.timedelta(hours=24)
        assert bluesky.parse_search_response({}, since=since) == []
        assert bluesky.parse_search_response({"posts": []}, since=since) == []
        weird = {"posts": [{"record": {"text": "hi", "createdAt": "not-a-date"}}]}
        # Unparseable date means we cannot prove it is old, so keep it.
        assert len(bluesky.parse_search_response(weird, since=since)) == 1

    def test_search_returns_empty_on_transport_failure(self):
        class Boom:
            def get(self, *a, **k):
                raise __import__("httpx").HTTPError("down")

        assert bluesky.search_posts(Boom(), "Dune", now=NOW) == []

    def test_search_returns_empty_on_error_status(self):
        class Resp:
            status_code = 503

            def json(self):
                raise AssertionError("must not parse a failed response")

        class Client:
            def get(self, *a, **k):
                return Resp()

        assert bluesky.search_posts(Client(), "Dune", now=NOW) == []


# --------------------------------------------------------------------
# Wikipedia
# --------------------------------------------------------------------

PAGEVIEWS_FIXTURE = {
    "items": [
        {"timestamp": f"202608{day:02d}00", "views": 1000} for day in range(18, 25)
    ] + [
        {"timestamp": f"202609{day:02d}00", "views": 3000} for day in range(1, 8)
    ]
}


class TestWikipedia:
    def test_path_encodes_the_title_as_one_segment(self):
        path = wikipedia.daily_views_path(
            "Dune: Part Three", dt.date(2026, 8, 1), dt.date(2026, 8, 8)
        )
        assert "Dune%3A_Part_Three" in path
        assert path.endswith("/20260801/20260808")

    def test_parse_sorts_oldest_first(self):
        series = wikipedia.parse_daily_views(PAGEVIEWS_FIXTURE)
        assert series[0][0] < series[-1][0]
        assert len(series) == 14

    def test_velocity_detects_building_interest(self):
        views, vel = wikipedia.velocity(wikipedia.parse_daily_views(PAGEVIEWS_FIXTURE))
        assert views == 21000
        assert vel == pytest.approx(3.0)

    def test_velocity_of_flat_interest_is_one(self):
        flat = [(f"2026090{d}", 500) for d in range(1, 9)][:14]
        series = [(f"20260{d:03d}", 500) for d in range(801, 815)]
        _, vel = wikipedia.velocity(series)
        assert vel == pytest.approx(1.0)

    def test_no_prior_week_reports_flat_not_infinity(self):
        series = [("20260901", 100), ("20260902", 200)]
        views, vel = wikipedia.velocity(series)
        assert views == 300
        assert vel == 1.0

    def test_empty_series_is_safe(self):
        assert wikipedia.velocity([]) == (0, 1.0)
        assert wikipedia.parse_daily_views({}) == []

    def test_fetch_velocity_degrades_on_failure(self):
        class Client:
            def get(self, *a, **k):
                raise __import__("httpx").HTTPError("nope")

        assert wikipedia.fetch_velocity(Client(), "Dune") == (0, 1.0)


# --------------------------------------------------------------------
# YouTube
# --------------------------------------------------------------------

class TestYouTube:
    def test_trailer_matching_accepts_the_real_thing(self):
        assert youtube.is_plausible_trailer(
            "Dune: Part Three | Official Trailer", "Warner Bros. Pictures",
            "Dune: Part Three")

    def test_trailer_matching_rejects_reactions_and_fan_edits(self):
        assert not youtube.is_plausible_trailer(
            "Dune Part Three Trailer REACTION", "Fan Channel", "Dune: Part Three")
        assert not youtube.is_plausible_trailer(
            "Superman Concept Trailer (Fan Made)", "Edits", "Superman")

    def test_trailer_matching_rejects_a_different_film(self):
        assert not youtube.is_plausible_trailer(
            "Wicked: For Good | Official Trailer", "Universal", "Dune: Part Three")

    def test_matcher_agrees_with_the_swift_client(self):
        """Both sides must accept and reject the same videos, or the
        server and the app will price different trailers."""
        assert youtube.is_plausible_trailer(
            "THE LEGEND OF ZELDA — Official Trailer (2027)", "Sony Pictures",
            "The Legend of Zelda")
        assert not youtube.is_plausible_trailer(
            "Dune: Part Three — Behind The Scenes", "Warner Bros.", "Dune: Part Three")

    def test_view_curve_matches_the_swift_implementation(self):
        assert youtube.trailing_week_fraction(1) == 1.0
        assert youtube.trailing_week_fraction(7) == 1.0
        assert youtube.trailing_week_fraction(30) < youtube.trailing_week_fraction(7)
        assert youtube.trailing_week_fraction(180) < 0.01

    def test_view_curve_is_monotonic(self):
        previous = 1.1
        for age in range(1, 366, 3):
            fraction = youtube.trailing_week_fraction(age)
            assert 0.0 <= fraction <= 1.0
            assert fraction <= previous + 1e-9
            previous = fraction

    def test_trailing_views_separate_fresh_from_stale(self):
        fresh = youtube.trailing_week_views(
            50_000_000, NOW - dt.timedelta(days=5), now=NOW)
        stale = youtube.trailing_week_views(
            50_000_000, NOW - dt.timedelta(days=180), now=NOW)
        assert fresh == 50_000_000
        assert stale < 1_000_000

    def test_hidden_counts_parse_as_none_not_zero(self):
        payload = {"items": [{"id": "abc", "statistics": {"viewCount": "500"},
                              "snippet": {"publishedAt": "2026-09-01T00:00:00Z"}}]}
        stats = youtube.parse_video_stats(payload)
        assert stats["abc"]["views"] == 500
        assert stats["abc"]["likes"] is None

    def test_parse_stats_reads_publish_date(self):
        payload = {"items": [{"id": "xyz",
                              "statistics": {"viewCount": "10", "likeCount": "2"},
                              "snippet": {"publishedAt": "2026-09-01T10:30:00Z"}}]}
        stats = youtube.parse_video_stats(payload)
        assert stats["xyz"]["published_at"].year == 2026
        assert stats["xyz"]["likes"] == 2

    def test_pick_trailer_skips_bad_hits_and_takes_the_good_one(self):
        payload = {"items": [
            {"id": {"videoId": "bad"},
             "snippet": {"title": "Dune Part Three Trailer Reaction",
                         "channelTitle": "Fans"}},
            {"id": {"videoId": "good"},
             "snippet": {"title": "Dune: Part Three | Official Trailer",
                         "channelTitle": "Warner Bros. Pictures"}},
        ]}
        assert youtube.pick_trailer(payload, "Dune: Part Three") == "good"

    def test_pick_trailer_returns_none_when_nothing_matches(self):
        payload = {"items": [{"id": {"videoId": "x"},
                              "snippet": {"title": "unrelated vlog",
                                          "channelTitle": "someone"}}]}
        assert youtube.pick_trailer(payload, "Dune") is None


# --------------------------------------------------------------------
# Box Office
# --------------------------------------------------------------------

# Live shapes captured Sep 2026 (weekend 2026W38).
BOM_FIXTURE = """
<table>
<tr><td>1</td><td>-</td>
<td><a href="/release/rl1/">Resident Evil</a></td>
<td>$60,000,000</td><td>-</td><td>3,684</td><td>-</td>
<td>$16,286</td><td>$60,000,000</td><td>1</td>
<td>Columbia Pictures</td><td>true</td><td>true</td></tr>
<tr><td>2</td><td>1</td>
<td><a href="/release/rl2/">Practical Magic 2</a></td>
<td>$12,155,000</td><td>-59.5%</td><td>4,163</td><td>+17</td>
<td>$2,919</td><td>$50,031,587</td><td>2</td>
<td>Warner Bros.</td><td>false</td><td>true</td></tr>
</table>
"""

TN_FIXTURE = """
<h1>Weekend Domestic Box Office September 18, 2026</h1>
<table>
<tr><td class="data">1</td><td class="data">(new)</td>
<td><b><a href="/movie/Resident-Evil-(2026)">Resident Evil</a></b></td>
<td class="data chart_estimate">$60,000,000</td>
<td class="data Weekend"> </td><td class="data">3,684</td>
<td class="data">$16,287</td><td class="data">$60,000,000</td>
<td class="data">3</td></tr>
<tr><td class="data">2</td><td class="data">(1)</td>
<td><b><a href="/movie/Practical-Magic-2">Practical Magic 2</a></b></td>
<td class="data">$12,155,000</td>
<td class="data">-59%</td><td class="data">4,163</td>
<td class="data">$2,920</td><td class="data">$50,032,000</td>
<td class="data">10</td></tr>
</table>
<table>
<!-- Secondary 'Top 5' daily table with TOTAL grosses: must never parse. -->
<tr><td>1</td><td><a href="/movie/Resident-Evil-(2026)">Resident Evil</a></td>
<td>$60,000,000</td><td>3,684</td><td>$60,000,000</td></tr>
</table>
"""

BOM_RELEASE_PAGE_FIXTURE = """
<div>Opening</div><span>$60,000,000</span>
<div>3,684 theaters</div>
<div>Release Date</div><span>Sep 18, 2026</span>
"""


class TestBoxOffice:
    def test_parse_bom_extracts_titles_and_grosses(self):
        results = boxoffice._parse_bom(BOM_FIXTURE)
        assert len(results) == 2
        assert results[0].title == "Resident Evil"
        assert results[0].gross_millions == 60.0
        assert results[0].source == "boxofficemojo"

    def test_parse_bom_uses_weeks_in_release_not_rank(self):
        """A week-2 holdover ranked #2 must NOT be an opening, and a
        #1-ranked opener must be — the weeks column decides, not the
        rank or a 'new' text marker."""
        results = boxoffice._parse_bom(BOM_FIXTURE)
        assert results[0].is_opening_weekend is True   # weeks == 1
        assert results[1].is_opening_weekend is False  # weeks == 2

    def test_parse_the_numbers_anchors_to_the_weekend_table(self):
        results = boxoffice._parse_the_numbers(TN_FIXTURE)
        assert len(results) == 2  # the Top-5 total table must not parse
        assert results[0].title == "Resident Evil"
        assert results[0].gross_millions == 60.0
        assert results[0].source == "the-numbers"

    def test_parse_the_numbers_detects_new_releases(self):
        results = boxoffice._parse_the_numbers(TN_FIXTURE)
        assert results[0].is_opening_weekend is True    # (new)
        assert results[1].is_opening_weekend is False  # (1)

    def test_cross_check_publishes_only_agreements(self):
        class Client:
            def get(self, url, **k):
                class Resp:
                    status_code = 200

                    def __init__(self):
                        if "boxofficemojo" in str(url):
                            self.text = BOM_FIXTURE
                        else:
                            self.text = TN_FIXTURE

                return Resp()

        got = boxoffice.fetch_cross_checked(
            Client(), ["Resident Evil", "Practical Magic 2"]
        )
        titles = [r.title for r in got]
        assert "Resident Evil" in titles          # agrees, opening
        assert "Practical Magic 2" not in titles  # holdover: excluded
        re_row = next(r for r in got if r.title == "Resident Evil")
        assert re_row.source == "boxofficemojo+the-numbers"

    def test_cross_check_drops_disagreements(self):
        disagree_bom = BOM_FIXTURE.replace("$60,000,000", "$45,000,000")

        class Client:
            def get(self, url, **k):
                class Resp:
                    status_code = 200

                    def __init__(self):
                        self.text = disagree_bom if "boxofficemojo" in str(url) else TN_FIXTURE

                return Resp()

        assert boxoffice.fetch_cross_checked(Client(), ["Resident Evil"]) == []

    def test_release_page_opening_is_read(self):
        got = boxoffice._parse_release_opening(
            BOM_RELEASE_PAGE_FIXTURE, "Resident Evil"
        )
        assert got is not None
        assert got.gross_millions == 60.0
        assert got.source == "boxofficemojo-release"

    def test_parse_gross_handles_formats(self):
        assert boxoffice._parse_gross("$75,200,000") == 75200000.0
        assert boxoffice._parse_gross("$60,000,000") == 60000000.0
        assert boxoffice._parse_gross("") is None
        assert boxoffice._parse_gross("N/A") is None
        assert boxoffice._parse_gross("-59.5%") is None
        # Bare integers must NOT parse: theater counts, ranks, and
        # weeks-in-release cells are bare numbers everywhere on these
        # charts. (A theater count once parsed as a $0.000037M gross.)
        assert boxoffice._parse_gross("3,684") is None
        assert boxoffice._parse_gross("37") is None

    def test_fetch_degrades_on_failure(self):
        class Boom:
            def get(self, *a, **k):
                raise __import__("httpx").HTTPError("down")

        assert boxoffice.fetch_actuals(Boom()) == []


# --------------------------------------------------------------------
# TMDB
# --------------------------------------------------------------------

class TestTMDB:
    def test_parse_drops_already_released_films(self):
        today = dt.date(2026, 9, 8)
        raw = {"id": 1, "title": "Old One", "release_date": "2026-01-01"}
        assert tmdb.parse_movie(raw, today=today) is None

    def test_parse_drops_undated_films(self):
        today = dt.date(2026, 9, 8)
        assert tmdb.parse_movie({"id": 1, "title": "TBD"}, today=today) is None
        assert tmdb.parse_movie(
            {"id": 1, "title": "Bad", "release_date": "not-a-date"}, today=today
        ) is None

    def test_parse_builds_our_shape(self):
        today = dt.date(2026, 9, 8)
        movie = tmdb.parse_movie(
            {"id": 42, "title": "Future Film", "release_date": "2026-11-20",
             "poster_path": "/abc.jpg", "genre_ids": [878], "popularity": 91.2,
             "overview": "Something happens."},
            today=today,
        )
        assert movie["id"] == "tmdb_42"
        assert movie["genre"] == "Sci-Fi"
        assert movie["posterURL"].endswith("/abc.jpg")
        assert movie["overview"] == "Something happens."

    def test_parse_upcoming_respects_the_window(self):
        today = dt.date(2026, 9, 8)
        payload = {"results": [
            {"id": 1, "title": "Soon", "release_date": "2026-09-25"},
            {"id": 2, "title": "Way Out", "release_date": "2028-01-01"},
        ]}
        titles = [m["title"] for m in
                  tmdb.parse_upcoming(payload, today=today, window_days=90)]
        assert titles == ["Soon"]

    def test_fetch_without_a_key_is_empty_not_an_error(self):
        assert tmdb.fetch_upcoming(None, "") == []


# --------------------------------------------------------------------
# End-to-end build with every network call stubbed
# --------------------------------------------------------------------

class StubClient:
    """Serves recorded payloads by URL substring."""

    def __init__(self):
        self.calls: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, **kwargs):
        url = str(url)
        params = kwargs.get("params") or {}
        self.calls.append(url)

        class Resp:
            status_code = 200

            def __init__(self, payload):
                self._payload = payload
                self.text = json.dumps(payload) if payload else ""

            def json(self):
                return self._payload

        if "searchPosts" in url:
            return Resp(SEARCH_FIXTURE)
        if "api.php" in url:
            return Resp({"query": {"search": [{"title": "Dune: Part Three"}]}})
        if "pageviews" in url:
            return Resp(PAGEVIEWS_FIXTURE)
        if "youtube/v3/search" in url:
            return Resp({"items": [{"id": {"videoId": "vid1"},
                                    "snippet": {"title": "Dune: Part Three | Official Trailer",
                                                "channelTitle": "Warner Bros."}}]})
        if "youtube/v3/videos" in url:
            return Resp({"items": [{"id": "vid1",
                                    "statistics": {"viewCount": "20000000",
                                                   "likeCount": "600000"},
                                    "snippet": {"publishedAt": "2026-09-05T00:00:00Z"}}]})
        if "movie/upcoming" in url:
            if params.get("page", 1) > 1:
                return Resp({"results": []})
            return Resp({"results": [
                {"id": 42, "title": "Dune: Part Three", "release_date": "2026-11-20",
                 "genre_ids": [878], "poster_path": "/d.jpg", "popularity": 88.0},
            ]})
        return Resp({})


@pytest.fixture()
def stub_httpx(monkeypatch):
    from boxcall_pipeline import build as build_module

    client = StubClient()
    monkeypatch.setattr(build_module.httpx, "Client", lambda **kw: client)
    return client


class TestBuild:
    def _run(self, tmp_path, stub_httpx, **overrides):
        from boxcall_pipeline import build as build_module

        kwargs = dict(
            seed_path=tmp_path / "seed.json",
            cache_path=tmp_path / "cache.json",
            tmdb_key="fake",
            youtube_key="fake",
            max_movies=5,
            youtube_search_budget=8,
            now=NOW,
        )
        kwargs.update(overrides)
        return build_module.build(tmp_path / "api", **kwargs)

    def test_build_writes_all_four_documents(self, tmp_path, stub_httpx):
        manifest = self._run(tmp_path, stub_httpx)
        out = tmp_path / "api"
        assert (out / "index.json").exists()
        assert (out / "upcoming.json").exists()
        assert (out / "signals.json").exists()
        assert (out / "actuals.json").exists()
        assert manifest["version"] == 1
        assert manifest["movieCount"] == 1

    def test_signals_carry_every_source(self, tmp_path, stub_httpx):
        self._run(tmp_path, stub_httpx)
        signals = json.loads((tmp_path / "api" / "signals.json").read_text())
        entry = signals["signals"]["tmdb_42"]
        assert entry["socialMentions24h"] == 2
        assert entry["socialSentiment"] is not None
        assert entry["wikipediaViews7d"] == 21000
        assert entry["wikipediaVelocity"] == pytest.approx(3.0)
        assert entry["youtubeViews7d"] == 20_000_000
        assert entry["youtubeEngagementRate"] == pytest.approx(0.03)

    def test_internal_fields_do_not_leak_into_the_published_json(self, tmp_path, stub_httpx):
        self._run(tmp_path, stub_httpx)
        raw = (tmp_path / "api" / "signals.json").read_text()
        assert "_trailerVideoId" not in raw

    def test_trailer_ids_are_cached_so_search_runs_once(self, tmp_path, stub_httpx):
        self._run(tmp_path, stub_httpx)
        cache = json.loads((tmp_path / "cache.json").read_text())
        assert cache["tmdb_42"] == "vid1"

        second = StubClient()
        from boxcall_pipeline import build as build_module
        build_module.httpx.Client = lambda **kw: second  # type: ignore[assignment]
        self._run(tmp_path, second)
        assert not any("youtube/v3/search" in c for c in second.calls), \
            "a cached trailer id must not trigger another 100-unit search"

    def test_without_a_youtube_key_fields_are_null_not_zero(self, tmp_path, stub_httpx):
        self._run(tmp_path, stub_httpx, youtube_key="")
        signals = json.loads((tmp_path / "api" / "signals.json").read_text())
        entry = signals["signals"]["tmdb_42"]
        assert entry["youtubeViews7d"] is None
        assert entry["youtubeEngagementRate"] is None

    def test_falls_back_to_the_seed_when_tmdb_is_unconfigured(self, tmp_path, stub_httpx):
        seed = tmp_path / "seed.json"
        seed.write_text(json.dumps([
            {"id": "seed_a", "title": "Seeded Film", "releaseDate": "2026-12-01"}
        ]))
        manifest = self._run(tmp_path, stub_httpx, tmdb_key="")
        assert manifest["movieCount"] == 1
        assert "seed" in manifest["sources"]["tmdb"]
        signals = json.loads((tmp_path / "api" / "signals.json").read_text())
        assert "seed_a" in signals["signals"]

    def test_actuals_json_is_written(self, tmp_path, stub_httpx):
        self._run(tmp_path, stub_httpx)
        actuals = json.loads((tmp_path / "api" / "actuals.json").read_text())
        assert actuals["version"] == 1
        assert "actuals" in actuals

    def test_manifest_reports_per_source_coverage(self, tmp_path, stub_httpx):
        manifest = self._run(tmp_path, stub_httpx)
        assert "bluesky" in manifest["sources"]
        assert "wikipedia" in manifest["sources"]
        assert "youtube" in manifest["sources"]
        assert "boxoffice" in manifest["sources"]
        assert manifest["generatedAt"].endswith("Z")
