# LoL Patch Impact

How do balance changes move champion priority in pro play, and can we predict
the next patch from the patch notes alone?

Covers the **LCK, LPL, LCS, LEC and CBLOL, 2026 season** (patches 26.1 onward).

What changed in each version: see [CHANGELOG.md](CHANGELOG.md).

```
python -m lolpatch all         # scrape + analyse + backtest + predict + events
python -m lolpatch events      # refresh Worlds after each round (also reruns regions)
python -m lolpatch assets      # champion portraits + team logos (once; re-run for new teams)
python -m lolpatch dashboard   # interactive dashboard in the browser
```

## Worlds 2026: the out-of-sample test

The model trains **only on domestic league games** (LCK/LPL/LCS/LEC/CBLOL). International
events are pure tests: for each one we forecast presence on the event's patch from
the last domestic patch (26.17) plus the patch notes in between, and score it
against what actually happened.

| Event | Patch | Forecast from | Role |
|---|---|---|---|
| First Stand 2026 | 26.5 | domestic 26.4 | decides any international adjustment |
| MSI 2026 | 26.13 | domestic 26.11 | decides |
| Esports World Cup 2026 | 26.13 | domestic 26.11 | decides |
| Demacia Cup Global Invitational 2026 | 26.18 | domestic 26.17 | test only |
| Worlds 2026 (Play-In + Swiss, in progress) | 26.20 | domestic 26.17 | test only |

Each forecast only uses domestic games from patches *before* the event.

**Patches at Worlds:** every Worlds 2026 game so far is on **26.20** (gol.gg
reports the game version as 16.20). The domestic leagues finished on 26.17
(last game 2026-10-10), so no league games exist on 26.18-26.20. Worlds is
the only 26.20 pro data. The Worlds forecast starts from domestic 26.17 and
applies the 26.18, 26.19 and 26.20 balance changes.

**International adjustment, tested and rejected.** After Worlds the theory was
that top domestic picks hold at internationals, so the model's drift should be
dropped. On First Stand/MSI/EWC that made things worse; top picks actually
fall *more* than the model expects at most events. Results are in
`event_adjustment_test.csv` and the dashboard's *Event tests* tab.

- **Frozen forecasts** (`data/forecasts/*_frozen.csv`) are written once and never
  overwritten, so the blind test can't be tuned after the fact. Delete the
  file only if you deliberately want a new blind test.
- **Current forecasts** are rebuilt each run with the latest classifier/model;
  they are useful but are *not* blind tests.
- **Live Worlds estimate**: the forecast is treated as a prior worth 20 games of
  evidence and updated with every Worlds game played (Beta-binomial), with an
  80% range. Re-run `python -m lolpatch events` after each Swiss round. The
  last two days of series pages are re-downloaded each run so a series cached
  mid-way gets completed.

Event games live in `data/processed/event_games.csv` and never enter training.

## Regions & teams (`python -m lolpatch regions`)

gol.gg lists picks and bans in **draft order** (verified: average presence
falls steadily from pick 1 to pick 5 and ban 1 to ban 5), so *early priority*
= a team's first two picks or first three bans.

- `region_champions.csv`: each region's presence and early priority per
  champion vs the all-league average on the same patches.
- `team_champions.csv`: each team's pick share vs its league, early picks, own
  bans, bans *against* them, win rate.
- `event_region_champions.csv`: what each region picked/banned at each event.
  Teams outside the scraped leagues (the LCP teams) are mapped in `config.EXTRA_TEAM_REGIONS`.

Champion portraits come from Riot's Data Dragon CDN (`data/assets/champion_icons.json`);
team logos are downloaded once from gol.gg into `data/assets/logos/`.

**Does priority win?** (`priority_effects.json`)

| Test | How it avoids fooling itself | Result |
|---|---|---|
| Drafting more-contested champions | contested = league presence on the *previous* patch; logistic regression controlling for team strength; both sides of every game | no effect (~50%) |
| Comfort picks (played 3+ times *before* this game) | comfort uses only earlier games; compared within team | no edge (−1.7 pp, CI −4.0 to +0.4) |
| Regional favorites at internationals | within region, cross-region games only | inconclusive (small n) |

## Ripple effects (`python -m lolpatch meta`)

How one champion's changes move *other* champions, by name.

- **Champion groups** (dashboard): pick any group (e.g. Nocturne + Qiyana). Group
  presence = share of games where at least one member was picked or banned. Over
  the group's rise (low point → peak, adjustable), every other champion's move
  is listed, with **In step?**: whether its patch-to-patch moves oppose the
  group's (likely pushed out) or just declined over the same stretch for other
  reasons.
- **Substitutions** (`substitutions.csv`, `substitute_pairs.csv`): a changed
  champion moves ≥10pp and untouched same-role champions move ≥5pp the other
  way in the same window (e.g. Poppy's 26.14 jungle buff vs Trundle/Sejuani/
  Nasus; Varus nerfs → Ezreal, twice).
- **Pro roles** come from gol.gg game pages (exact) where fetched, otherwise
  inferred from team compositions (each team fields one top/jungle/mid/bot/
  support). The inference scores 98.4% on held-out games with known roles.

## Draft answers (`python -m lolpatch players`, then `draft`)

Which picks teams make *in response* to an opponent's pick.

- **Draft order**: gol.gg groups each team's picks by draft turn: the first-pick
  team's picks read `1 | 2 2` (global picks 1 | 4-5 | 8-9), the other team's
  `2 2 | 1` (2-3 | 6-7 | 10). Known for ~94% of games. (gol.gg's blue/red labels
  don't line up with who picked first, so the dashboard says "first-pick team".)
- **Lane answers** (`lane_answers.csv`): team A fills a role with X, team B fills
  the same role later with Y. A pattern needs ≥4 occurrences, ≥1.5× the rate at
  which Y is usually the later pick in that role **on the same patches** (so
  champions that were merely popular together don't count), and a binomial test
  surviving a 10% false-discovery-rate correction.
- **First-pick answers** (`first_pick_answers.csv`): global pick 1 → picks 2-3.
- **Where it comes from** (`answer_breakdown.csv`): each pattern is broken down by
  event/league, region, team and player (players from `players.csv`, fetched
  from every game page) and tagged Worlds-, region-, team- or player-specific
  when one group accounts for most of it at ≥2× everyone else's rate.

## Worlds games, ratings and expected results (`python -m lolpatch matches`)

- **Game details** (`match_games/teams/players/timeline/gold.csv`): for every
  international game, from gol.gg's game and "All stats" pages: objectives,
  objective timeline, gold lead by minute, plates/voidgrubs, ~60 stats per player.
- **Expected result** (`match_expected.csv`, `expected_eval.json`): pre-game win
  probability from team + region strength over all 2026 games before that day
  (recent games weighted more). Settings, including whether draft/comfort
  features are used, are tuned on First Stand, MSI, EWC and Demacia; Worlds is
  only scored. The draft features didn't help on the tuning events, so they're off.
- **Player rating** (`player_ratings.csv`): stats include damage/gold/kill
  participation as shares of the team and of the whole game, damage and gold
  vs the lane opponent, and damage per gold; role-weighted z-scores vs the same
  role across all international games, plus 0.4 × (won − expected win chance)
  in the same units, turned into a 1.0–5.0 percentile. Best player on a winning
  team: median ~4.8; best on a losing team: ~3.3.
- **Scope:** ratings cover every 2026 pro game (domestic + international,
  2,397 games) on one scale, so `python -m lolpatch matches` fetches each
  game's stats page once (~1 hour first run, cached after).
- **Expected player rating** (`expected_rating_eval.json`): before each game,
  from the player's earlier games (form, shrunk toward 3.0), their record on
  that champion relative to their form (shrunk toward 0), the team's win
  chance, and for international games their region's cross-region strength.
  Fit on games before Worlds, scored on Worlds: correlation with actual
  ratings 0.24 (0.20 without champion history).

## Upcoming matches (`python -m lolpatch upcoming`)

- **Schedule:** Leaguepedia's MatchSchedule when it responds (it often rate-limits
  anonymous requests); otherwise `data/overrides/upcoming.csv`
  (`date,stage,team_a,team_b,best_of`), which also keeps the last good fetch.
  Leaguepedia names are mapped to gol.gg's (`TEAM_ALIASES`).
- **Win chance:** team + region strength fit on every 2026 game (same model as
  the expected results); series odds from the per-game chance.
- **Likely bans:** half the team's own recent bans, half what opponents ban
  against the other team. **Likely picks:** 80% the team's recent pool for that
  role, 20% the Worlds meta, minus likely bans (game 1, since Worlds is Fearless).
- **Chance to reach the knockouts:** the remaining Swiss rounds simulated 20,000
  times (same-record pairings, rematches avoided, Bo3).

## Pipeline

| Step | Module | Output (`data/processed/`) |
|---|---|---|
| Pro games: picks, bans, winner, game version per game, scraped from [gol.gg](https://gol.gg) | `scrape_matches.py` | `games.csv` |
| Champion changes from Riot patch notes, auto-scored buff/nerf | `scrape_patches.py` | `patch_changes.csv` (per line), `champion_changes.csv` (per champion) |
| Per-patch stats + patch-to-patch transitions | `analyze.py` | `champion_patch_stats.csv`, `transitions.csv`, `verdict_effects.csv` |
| Walk-forward backtest + next-patch prediction | `model.py` | `backtest_metrics.csv`, `backtest_predictions.csv`, `predictions.csv` |

All downloaded pages are cached in `data/raw/`, so re-runs are offline and
instant. Use `--refresh` to pick up new games or patch notes.

## Key definitions

- **Presence** = (picks + bans) ÷ games on that patch. This is the main
  "priority" metric. Pick rate, ban rate and win rate are kept too. Win rate
  is shrunk toward 50% for small samples.
- **Patch**: pro games are bucketed by the *game version they were actually
  played on*. gol.gg reports it as `16.N`; Riot's branding is `26.N`.
- **Transition**: a pair of consecutive patches that have pro games (≥20 games).
  Riot patches skipped by pro play (e.g. a patch released during a break) are
  folded into the next transition, so their changes still count.
- **Buff/nerf score** (−1…+1): Riot doesn't label changes, so each `X ⇒ Y`
  line is scored by direction, with "lower is better" stats such as cooldowns
  and costs flipped. That is blended with cue phrases in the designer context
  ("overperforming", "struggling", …). Scores ≥ 0.25 are *buff*, ≤ −0.25 are
  *nerf*, and anything in between is *adjust*.

### Correcting the classifier

Auto-classification is a heuristic and big reworks are often "adjust". Fix any
row in `data/overrides/champion_overrides.csv`; overrides always win:

```csv
patch,champion,verdict,score,note
26.15,Bel'Veth,nerf,-0.5,rework was a net nerf in pro
```

Then run `python -m lolpatch fetch && python -m lolpatch predict`. Fetch is
cached, so it only re-scores.

## Predicting a future patch

The model learns `Δ presence` from: buff/nerf score, number and size of
changes, whether Riot mentions pro play, a base-stat flag, the champion's
current presence/pick/ban/win rates, its momentum, and the previous patch's
change (lagged effect). It is validated **walk-forward**: for each patch it
trains only on earlier patches, then compares against "nothing changes" and
"mean-reversion only" baselines.

```bash
# Predict the newest patch in Riot's notes (default)
python -m lolpatch predict

# Predict a patch whose notes are at a non-standard URL
python -m lolpatch predict --target 26.22 --notes-url https://www.leagueoflegends.com/en-us/news/game-updates/...

# Predict from a preview list (e.g. Riot's patch preview, before notes ship)
python -m lolpatch predict --target 26.22 --changes-csv my_preview.csv   # columns: champion,verdict[,score]
```

## Limitations

- Item, rune and system changes aren't modelled. They are often what really
  moves the meta (e.g. a champion rising because their core item was buffed).
- Fearless draft inflates presence for the whole pool and shifts the
  pick-vs-ban mix. It is consistent across 2026, so deltas stay comparable.
- Mid-patch hotfixes and B-patches are folded into the patch they shipped on.
- One season is a small sample (on the order of a few hundred changed-champion
  transitions), so treat predictions as informed priors, not certainties.
