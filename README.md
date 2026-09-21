# BoxCall Data

The static BoxCall data API — built from free public sources and served
from this repo's GitHub Pages. **Dedicated repo**: no other project
deploys to this Pages site, so nothing can clobber it.

- **Live API:** `https://mvalasek77-droid.github.io/boxcall-data/api/v1/`
- **App consumer:** `Config.dataAPIBaseURL` in the BoxCall iOS app
- **Source of truth for pipeline code:** `backend/pipeline/` in the
  [Mike_claw monorepo](https://github.com/mvalasek77-droid/Mike_claw)
  (`pr-30` branch). This repo is the deployment target, not the dev
  home — pipeline work happens there and is mirrored here.

## What it publishes

| File | Contents |
|---|---|
| `index.json` | Manifest: schema version, build time, per-source coverage |
| `upcoming.json` | The catalog — titles, dates, posters, genres |
| `signals.json` | Per-movie crowd signals, keyed by movie id |
| `actuals.json` | Domestic opening-weekend actuals for settlement |

`actuals.json` is cross-checked: a film publishes only when **both**
Box Office Mojo's weekend chart **and** The Numbers' weekend chart
report it as a new release with figures agreeing within 2%. A film's
second-weekend gross can never be recorded as its opening (holdover
guard: BOM weeks-in-release column, TN `(new)` marker, 10-day window).

## Schedule

Every 3 hours via GitHub Actions cron (this repo's default branch is
`main`, so the schedule actually fires — unlike a workflow stranded on
a feature branch).

## Secrets (optional)

| Secret | Purpose |
|---|---|
| `TMDB_API_KEY` | Catalog enrichment; without it the bundled seed slate is used |
| `YOUTUBE_API_KEY` | Trailer stats; without it those fields publish `null` |

With no secrets at all the pipeline still publishes sentiment, Wikipedia
velocity, and box-office actuals — everything the settlement path needs.