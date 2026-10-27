"""Expected results and player ratings for international games.

Expected result: a pre-game win probability for each team, using only games played
*before* that day:
  * team strength: logistic (Bradley-Terry style) team + region effects over all 2026
    games, recent games weighted more; international games tie the leagues together
  * draft: each pick's recent win rate (decayed, shrunk toward 50%)
  * comfort: how much the team has played its picks before
  * first pick: whether the team had first pick in the draft
Settings, including whether the draft/comfort features are used at all, are tuned on the
earlier events (First Stand, MSI, EWC, Demacia Cup); Worlds is only ever scored. (On those
events the draft features didn't improve predictions, so the tuned model leaves them out;
each game's draft edge is still reported.)

Player rating (1.0-5.0): role-relative stats (z-scores vs the same role across every
international game) weighted per role, plus the result relative to expectation
(won - expected win probability). The combined score is turned into a percentile, so 3.0
is a typical international performance and ~4.8 is the top 5%.

Outputs (data/processed/): match_expected.csv, expected_eval.json, player_ratings.csv
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

from .analyze import load_games
from .config import PROCESSED
from .regions import team_regions

TEST_EVENT = "worlds"
CHAMP_HALF_LIFE, COMFORT_HALF_LIFE, GAME_HALF_LIFE = 30, 120, 90  # days
CHAMP_PRIOR = 10  # pseudo-games at 50% for champion win rates
# Weight of (won - expected) next to the standardised stat score. Stats already track winning
# (r ~ 0.46), so 0.4 keeps the best winner around 4.8 while a standout loser still reaches ~3.3+.
RESULT_WEIGHT = 0.4

# Per-role stat weights (sum to 1). Negative-is-good stats are flipped before weighting.
ROLE_WEIGHTS = {
    "top": {"laning": .22, "dmg_share": .16, "kp": .10, "deaths": .14, "tanking": .12, "turrets": .10,
            "csm": .08, "ka_pm": .08},
    "jungle": {"kp": .18, "deaths": .14, "vision": .14, "laning": .14, "dmg_share": .12, "tanking": .08,
               "ka_pm": .12, "cc": .08},
    "mid": {"dmg_share": .20, "laning": .18, "kp": .14, "deaths": .14, "csm": .10, "ka_pm": .12, "dpm": .12},
    "bot": {"dmg_share": .20, "dpm": .14, "laning": .14, "csm": .14, "deaths": .16, "kp": .10, "turrets": .12},
    "support": {"vision": .24, "kp": .18, "deaths": .14, "cc": .12, "laning": .08, "ka_pm": .12, "protect": .12},
}
STAT_LABELS = {"laning": "Lane lead at 15", "dmg_share": "Damage share", "kp": "Kill participation",
               "deaths": "Deaths (fewer is better)", "tanking": "Damage absorbed", "turrets": "Turret damage",
               "csm": "CS per minute", "ka_pm": "Kills + assists per minute", "vision": "Vision",
               "cc": "Crowd control", "dpm": "Damage per minute", "protect": "Heals & shields on allies"}


# --- expected result ------------------------------------------------------------------

def all_games() -> pd.DataFrame:
    dom = load_games()
    ev = pd.read_csv(PROCESSED / "event_games.csv", dtype={"patch": str, "series_id": str})
    games = pd.concat([dom.assign(event=None), ev.assign(event=ev.league)], ignore_index=True)
    games["date"] = pd.to_datetime(games.date)
    regions = team_regions(dom)
    games["region"] = games.team.map(regions).fillna("Other")
    return games


def _decay(age_days, half_life):
    return 0.5 ** (np.asarray(age_days, float) / half_life)


def game_features(games: pd.DataFrame) -> pd.DataFrame:
    """Per team-game: draft strength and comfort from strictly earlier days (decayed counts)."""
    picks = games.assign(champion=games.picks.str.split("|")).explode("champion")
    rows = []
    for day in sorted(games.date.unique()):
        past = picks[picks.date < day]
        today = games[games.date == day]
        if len(past):
            age = (day - past.date).dt.days.to_numpy()
            w_ch = _decay(age, CHAMP_HALF_LIFE)
            ch = past.assign(w=w_ch, ww=w_ch * past.result).groupby("champion")[["w", "ww"]].sum()
            wr = (ch.ww + CHAMP_PRIOR * 0.5) / (ch.w + CHAMP_PRIOR)
            w_cf = _decay(age, COMFORT_HALF_LIFE)
            comfort = past.assign(w=w_cf).groupby(["team", "champion"]).w.sum()
        else:
            wr, comfort = pd.Series(dtype=float), pd.Series(dtype=float)
        for r in today.itertuples():
            champs = r.picks.split("|")
            rows.append(dict(game_uid=r.game_uid, team=r.team,
                             draft=float(np.mean([wr.get(c, 0.5) - 0.5 for c in champs])),
                             comfort=float(np.mean([np.log1p(comfort.get((r.team, c), 0.0)) for c in champs]))))
    return games.merge(pd.DataFrame(rows), on=["game_uid", "team"])


def _design(pairs: pd.DataFrame, teams: list[str], regions: list[str], region_scale: float,
            use_draft: bool) -> np.ndarray:
    t_idx = {t: i for i, t in enumerate(teams)}
    r_idx = {r: i for i, r in enumerate(regions)}
    X = np.zeros((len(pairs), len(teams) + len(regions) + 3))
    for k, p in enumerate(pairs.itertuples()):
        if p.team_a in t_idx:
            X[k, t_idx[p.team_a]] += 1
        if p.team_b in t_idx:
            X[k, t_idx[p.team_b]] -= 1
        # Scaling region columns up = weaker penalty on region effects than on team effects.
        X[k, len(teams) + r_idx[p.region_a]] += region_scale
        X[k, len(teams) + r_idx[p.region_b]] -= region_scale
        if use_draft:
            X[k, -3] = (p.draft_a - p.draft_b) * 10
            X[k, -2] = p.comfort_a - p.comfort_b
        X[k, -1] = p.fp_a
    return X


def to_pairs(g: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """One row per game, randomly oriented (team A vs team B)."""
    rng = np.random.default_rng(seed)
    a = g.merge(g, on="game_uid", suffixes=("_a", "_b"))
    a = a[a.team_a != a.team_b]
    a = a.assign(_r=rng.random(len(a))).sort_values("_r").drop_duplicates("game_uid")
    fp = a.first_pick_a.fillna(0.5)
    return pd.DataFrame({
        "game_uid": a.game_uid, "date": a.date_a, "event": a.event_a, "team_a": a.team_a, "team_b": a.team_b,
        "region_a": a.region_a, "region_b": a.region_b, "draft_a": a.draft_a, "draft_b": a.draft_b,
        "comfort_a": a.comfort_a, "comfort_b": a.comfort_b, "fp_a": (fp - 0.5) * 2, "y": a.result_a,
    })


def predict_events(pairs: pd.DataFrame, C: float, region_scale: float, use_draft: bool = True) -> pd.Series:
    """Win probability for team A in every international game, fit on earlier days only."""
    regions = sorted(set(pairs.region_a) | set(pairs.region_b))
    out = pd.Series(np.nan, index=pairs.index)
    for day in sorted(pairs[pairs.event.notna()].date.unique()):
        train = pairs[pairs.date < day]
        test = pairs[(pairs.date == day) & pairs.event.notna()]
        teams = sorted(set(train.team_a) | set(train.team_b))
        w = _decay((day - train.date).dt.days, GAME_HALF_LIFE)
        model = LogisticRegression(C=C, fit_intercept=False, max_iter=2000)
        model.fit(_design(train, teams, regions, region_scale, use_draft), train.y, sample_weight=w)
        out[test.index] = model.predict_proba(_design(test, teams, regions, region_scale, use_draft))[:, 1]
    return out


def _metrics(y, p) -> dict:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return dict(n=int(len(y)), log_loss=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                accuracy=float(np.mean((p > 0.5) == (y == 1))), brier=float(np.mean((p - y) ** 2)))


def expected_results(log=print) -> pd.DataFrame:
    games = game_features(all_games())
    pairs = to_pairs(games).reset_index(drop=True)
    tune = pairs.event.notna() & (pairs.event != TEST_EVENT)
    # Tuning never sees Worlds games, not even as training data (the Play-In overlapped the
    # end of the Demacia Cup), so Worlds stays a clean test.
    no_worlds = pairs[(pairs.event != TEST_EVENT) | pairs.event.isna()]
    best = None
    for C in (0.3, 1.0, 3.0):
        for region_scale in (1.0, 3.0):
            for use_draft in (True, False):
                p = predict_events(no_worlds, C, region_scale, use_draft)
                ll = _metrics(no_worlds.y[tune], p[tune[no_worlds.index]])["log_loss"]
                if best is None or ll < best[0]:
                    best = (ll, C, region_scale, use_draft)
    _, C, region_scale, use_draft = best
    p_chosen = predict_events(pairs, C, region_scale, use_draft)
    p_other = predict_events(pairs, C, region_scale, not use_draft)
    test = pairs.event == TEST_EVENT
    coin = _metrics(pairs.y[test], np.full(test.sum(), 0.5)) | {"accuracy": 0.5}
    evaluation = dict(
        settings=dict(C=C, region_scale=region_scale, uses_draft=use_draft),
        tuned_on=_metrics(pairs.y[tune], p_chosen[tune]),
        worlds=dict(model=_metrics(pairs.y[test], p_chosen[test]),
                    with_draft=_metrics(pairs.y[test], p_chosen[test] if use_draft else p_other[test]),
                    strength_only=_metrics(pairs.y[test], p_other[test] if use_draft else p_chosen[test]),
                    coin_flip=coin),
    )
    p_full = p_chosen
    (PROCESSED / "expected_eval.json").write_text(json.dumps(evaluation, indent=2))
    pairs["p_a"] = p_full
    ev = pairs[pairs.event.notna()]
    both = pd.concat([
        pd.DataFrame({"game_uid": ev.game_uid, "team": ev.team_a, "opponent": ev.team_b, "p_win": ev.p_a,
                      "draft_edge": ev.draft_a - ev.draft_b, "won": ev.y}),
        pd.DataFrame({"game_uid": ev.game_uid, "team": ev.team_b, "opponent": ev.team_a, "p_win": 1 - ev.p_a,
                      "draft_edge": ev.draft_b - ev.draft_a, "won": 1 - ev.y}),
    ], ignore_index=True)
    both.to_csv(PROCESSED / "match_expected.csv", index=False)
    w = evaluation["worlds"]
    log(f"  expected results: tuned C={C}, region_scale={region_scale}, draft features {'on' if use_draft else 'off'}")
    log(f"  Worlds ({w['model']['n']} games): accuracy {w['model']['accuracy']:.0%}, log loss {w['model']['log_loss']:.3f} "
        f"(strength only {w['strength_only']['log_loss']:.3f}, coin flip {w['coin_flip']['log_loss']:.3f})")
    return both


# --- player ratings ------------------------------------------------------------------------

def player_metrics(players: pd.DataFrame) -> pd.DataFrame:
    p = players.copy()
    dur = p.duration.clip(lower=10)
    team_tot = lambda col: p.groupby(["game_uid", "team"])[col].transform("sum").replace(0, np.nan)
    p["laning"] = (p["GD@15"].fillna(0) / 1000 + p["CSD@15"].fillna(0) / 10 + p["XPD@15"].fillna(0) / 1000) / 3
    p["dmg_share"] = p["DMG%"]
    p["kp"] = p["KP%"]
    p["deaths"] = -(p["Deaths"] / dur)
    p["tanking"] = p["Total damage taken"] / team_tot("Total damage taken") + \
        p["Damage self mitigated"] / team_tot("Damage self mitigated")
    p["turrets"] = p["Damage dealt to turrets"] / team_tot("Damage dealt to turrets")
    p["csm"] = p["CSM"]
    p["ka_pm"] = p["K+A Per Minute"]
    p["vision"] = p["VSPM"]
    p["cc"] = p["Time ccing others"] / dur
    p["dpm"] = p["DPM"]
    p["protect"] = (p["Total Heals On Teammates"].fillna(0) + p["Total Damage Shielded On Teammates"].fillna(0)) / dur
    return p


def player_ratings(expected: pd.DataFrame, log=print) -> pd.DataFrame:
    players = pd.read_csv(PROCESSED / "match_players.csv")
    games = pd.read_csv(PROCESSED / "match_games.csv")
    p = player_metrics(players)
    stats = sorted({s for w in ROLE_WEIGHTS.values() for s in w})
    # z-scores within role across every international game (the reference population).
    for s in stats:
        grp = p.groupby("role")[s]
        p[f"z_{s}"] = ((p[s] - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)).clip(-3, 3).fillna(0)
    p["perf"] = 0.0
    for role, weights in ROLE_WEIGHTS.items():
        m = p.role == role
        p.loc[m, "perf"] = sum(w * p.loc[m, f"z_{s}"] for s, w in weights.items())
    p["perf_std"] = p.groupby("role").perf.transform(lambda s: (s - s.mean()) / s.std())
    p = p.merge(expected[["game_uid", "team", "p_win", "opponent"]], on=["game_uid", "team"], how="left")
    p["p_win"] = p.p_win.fillna(0.5)
    p["vs_expected"] = p.result - p.p_win
    p["score"] = p.perf_std + RESULT_WEIGHT * p.vs_expected / 0.5
    p["rating"] = (1 + 4 * (rankdata(p.score) - 0.5) / len(p)).round(1)
    p["stat_rating"] = (1 + 4 * (rankdata(p.perf_std) - 0.5) / len(p)).round(1)
    p["mvp"] = p.rating == p.groupby("game_uid").rating.transform("max")
    p = p.merge(games[["game_uid", "event", "stage", "date"]], on="game_uid", how="left")
    keep = ["game_uid", "event", "stage", "date", "team", "opponent", "side", "role", "player", "champion", "result",
            "p_win", "vs_expected", "rating", "stat_rating", "mvp", "Kills", "Deaths", "Assists", "CSM", "GOLD%",
            "DMG%", "DPM", "VSPM", "KP%", "GD@15", "CSD@15", "XPD@15", "Golds", "Total damage to Champion",
            "Vision Score", "Solo kills", "duration"] + [f"z_{s}" for s in stats]
    out = p[keep]
    out.to_csv(PROCESSED / "player_ratings.csv", index=False)
    log(f"  rated {len(out)} player-games across {out.game_uid.nunique()} international games")
    return out


def run(log=print):
    expected = expected_results(log=log)
    return player_ratings(expected, log=log)
