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

Expected player rating: before each game, from the player's earlier games (form), their
record on that champion relative to their form, their team's win chance, and for
international games how strong their region has been against other regions.

Outputs (data/processed/): match_expected.csv, expected_eval.json, player_ratings.csv,
expected_rating_eval.json
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
# Gold lead that counts as a clear win when results are blended with how decisive games were:
# an average lead of 2k over the game reads as a 73% "deserved" win.
MARGIN_SCALE = 2000
MARGIN_WEIGHTS = (0.25, 0.5)  # tried on top of the best results-only settings
# Weight of (won - expected) next to the standardised stat score. Stats already track winning
# (r ~ 0.46), so 0.4 keeps the best winner around 4.8 while a standout loser still reaches ~3.3+.
RESULT_WEIGHT = 0.4

# Per-role stat weights (sum to 1). Negative-is-good stats are flipped before weighting.
# "_share" = share of the player's own team; "_game" = share of everything in the game (both teams);
# "_vs_opp" = against the lane opponent (same role, other team).
ROLE_WEIGHTS = {
    "top": {"laning": .16, "dmg_share": .10, "dmg_game": .04, "dmg_vs_opp": .08, "gold_vs_opp": .08,
            "gold_share": .04, "kp": .10, "deaths": .12, "tanking": .10, "turrets": .08, "csm": .04, "ka_pm": .06},
    "jungle": {"kp": .16, "deaths": .12, "vision": .12, "laning": .10, "dmg_share": .08, "dmg_game": .04,
               "gold_vs_opp": .08, "dmg_vs_opp": .04, "gold_share": .04, "tanking": .06, "ka_pm": .10, "cc": .06},
    "mid": {"dmg_share": .14, "dmg_game": .06, "dmg_vs_opp": .08, "gold_vs_opp": .08, "dmg_per_gold": .06,
            "gold_share": .04, "laning": .12, "kp": .12, "deaths": .12, "csm": .06, "ka_pm": .06, "dpm": .06},
    "bot": {"dmg_share": .14, "dmg_game": .06, "dmg_vs_opp": .08, "gold_vs_opp": .08, "dmg_per_gold": .06,
            "gold_share": .04, "dpm": .08, "laning": .10, "csm": .10, "deaths": .12, "kp": .08, "turrets": .06},
    "support": {"vision": .20, "kp": .16, "deaths": .12, "cc": .10, "laning": .06, "ka_pm": .10, "protect": .10,
                "gold_vs_opp": .04, "dmg_game": .04, "tanking": .04, "dmg_vs_opp": .04},
}
assert all(abs(sum(w.values()) - 1) < 1e-9 for w in ROLE_WEIGHTS.values())
STAT_LABELS = {"laning": "Lead over lane opponent at 15 min", "dmg_share": "Share of team's damage",
               "dmg_game": "Share of all damage in the game", "dmg_vs_opp": "Damage vs lane opponent",
               "gold_vs_opp": "Gold vs lane opponent", "gold_share": "Share of team's gold",
               "dmg_per_gold": "Damage for the gold earned", "kp": "Kill participation",
               "deaths": "Staying alive (fewer deaths)", "tanking": "Damage absorbed for the team",
               "turrets": "Damage to towers", "csm": "Farming (CS per minute)", "ka_pm": "Kills + assists per minute",
               "vision": "Vision (wards)", "cc": "Crowd control", "dpm": "Damage per minute",
               "protect": "Heals & shields on teammates"}


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


def with_margins(pairs: pd.DataFrame) -> pd.DataFrame:
    """Add `margin`: team A's average gold lead over the game (NaN where there's no gold timeline)."""
    teams = pd.read_csv(PROCESSED / "match_teams.csv", usecols=["game_uid", "side", "team"])
    blue = teams[teams.side == "blue"].set_index("game_uid").team
    lead = pd.read_csv(PROCESSED / "match_gold.csv").groupby("game_uid").blue_lead.mean()
    sign = np.where(pairs.game_uid.map(blue) == pairs.team_a, 1.0, -1.0)
    return pairs.assign(margin=pairs.game_uid.map(lead) * sign)


def fit_strength(train: pd.DataFrame, teams: list[str], regions: list[str], C: float, region_scale: float,
                 use_draft: bool, margin_weight: float, weights) -> LogisticRegression:
    """Team + region logistic model. With margin_weight > 0 the target blends the result with how
    decisively the game went (average gold lead, squashed to 0–1): a scrappy win counts as less
    than a stomp, so a team winning narrowly, or losing badly, moves its strength accordingly."""
    X = _design(train, teams, regions, region_scale, use_draft)
    y = train.y.to_numpy(float)
    if margin_weight and "margin" in train:
        soft = 1 / (1 + np.exp(-train.margin.to_numpy(float) / MARGIN_SCALE))
        y = np.where(np.isnan(soft), y, (1 - margin_weight) * y + margin_weight * soft)
    w = np.asarray(weights, float)
    # Fractional targets: each game enters once as a win and once as a loss, weighted by y and 1 - y.
    model = LogisticRegression(C=C, fit_intercept=False, max_iter=3000)
    model.fit(np.vstack([X, X]), np.r_[np.ones(len(y)), np.zeros(len(y))], sample_weight=np.r_[w * y, w * (1 - y)])
    return model


def predict_events(pairs: pd.DataFrame, C: float, region_scale: float, use_draft: bool = True,
                   all_days: bool = False, margin_weight: float = 0.0) -> pd.Series:
    """Win probability for team A, fit on earlier days only: international games, or every game."""
    regions = sorted(set(pairs.region_a) | set(pairs.region_b))
    out = pd.Series(np.nan, index=pairs.index)
    target = pairs if all_days else pairs[pairs.event.notna()]
    for day in sorted(target.date.unique()):
        train = pairs[pairs.date < day]
        test = target[target.date == day]
        if train.y.nunique() < 2:
            continue  # the season's first days: nothing to learn from yet
        teams = sorted(set(train.team_a) | set(train.team_b))
        w = _decay((day - train.date).dt.days, GAME_HALF_LIFE)
        model = fit_strength(train, teams, regions, C, region_scale, use_draft, margin_weight, w)
        out[test.index] = model.predict_proba(_design(test, teams, regions, region_scale, use_draft))[:, 1]
    return out


def _metrics(y, p) -> dict:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return dict(n=int(len(y)), log_loss=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                accuracy=float(np.mean((p > 0.5) == (y == 1))), brier=float(np.mean((p - y) ** 2)))


def expected_results(log=print) -> pd.DataFrame:
    games = game_features(all_games())
    pairs = with_margins(to_pairs(games).reset_index(drop=True))
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
    results_only_ll, C, region_scale, use_draft = best
    # Then: does blending in how decisive games were (gold leads) beat results alone?
    margin_weight = 0.0
    for mw in MARGIN_WEIGHTS:
        p = predict_events(no_worlds, C, region_scale, use_draft, margin_weight=mw)
        ll = _metrics(no_worlds.y[tune], p[tune[no_worlds.index]])["log_loss"]
        if ll < best[0]:
            best, margin_weight = (ll, C, region_scale, use_draft), mw
    p_chosen = predict_events(pairs, C, region_scale, use_draft, margin_weight=margin_weight)
    p_other = predict_events(pairs, C, region_scale, not use_draft, margin_weight=margin_weight)
    p_results = predict_events(pairs, C, region_scale, use_draft) if margin_weight else p_chosen
    test = pairs.event == TEST_EVENT
    coin = _metrics(pairs.y[test], np.full(test.sum(), 0.5)) | {"accuracy": 0.5}
    evaluation = dict(
        settings=dict(C=C, region_scale=region_scale, uses_draft=use_draft, margin_weight=margin_weight),
        tuned_on=_metrics(pairs.y[tune], p_chosen[tune]),
        tuned_on_results_only=dict(log_loss=results_only_ll),
        worlds=dict(model=_metrics(pairs.y[test], p_chosen[test]),
                    results_only=_metrics(pairs.y[test], p_results[test]),
                    with_draft=_metrics(pairs.y[test], p_chosen[test] if use_draft else p_other[test]),
                    strength_only=_metrics(pairs.y[test], p_other[test] if use_draft else p_chosen[test]),
                    coin_flip=coin),
    )
    (PROCESSED / "expected_eval.json").write_text(json.dumps(evaluation, indent=2))
    # Every game gets a prior-only expectation (player ratings use it for domestic games too).
    pairs["p_a"] = predict_events(pairs, C, region_scale, use_draft, all_days=True,
                                  margin_weight=margin_weight).fillna(0.5)
    ev = pairs
    both = pd.concat([
        pd.DataFrame({"game_uid": ev.game_uid, "team": ev.team_a, "opponent": ev.team_b, "p_win": ev.p_a,
                      "draft_edge": ev.draft_a - ev.draft_b, "won": ev.y}),
        pd.DataFrame({"game_uid": ev.game_uid, "team": ev.team_b, "opponent": ev.team_a, "p_win": 1 - ev.p_a,
                      "draft_edge": ev.draft_b - ev.draft_a, "won": 1 - ev.y}),
    ], ignore_index=True)
    both.to_csv(PROCESSED / "match_expected.csv", index=False)
    w = evaluation["worlds"]
    log(f"  expected results: tuned C={C}, region_scale={region_scale}, draft features {'on' if use_draft else 'off'}, "
        f"margin weight {margin_weight}")
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
    # Relative to the whole game and to the lane opponent (same role, other team).
    dmg, gold = "Total damage to Champion", "Golds"
    p["gold_share"] = p["GOLD%"]
    p["dmg_game"] = 100 * p[dmg] / p.groupby("game_uid")[dmg].transform("sum").replace(0, np.nan)
    p["gold_game"] = 100 * p[gold] / p.groupby("game_uid")[gold].transform("sum").replace(0, np.nan)
    opp = p[["game_uid", "role", "team", dmg, gold]].rename(columns={"team": "opp_team", dmg: "opp_dmg", gold: "opp_gold"})
    p = p.merge(opp, on=["game_uid", "role"], how="left")
    p = p[p.team != p.opp_team].drop(columns="opp_team")
    p["dmg_vs_opp"] = np.log2((p[dmg] + 500) / (p.opp_dmg + 500)).clip(-3, 3)
    p["gold_vs_opp"] = (p[gold] - p.opp_gold) / dur
    p["dmg_per_gold"] = p[dmg] / p[gold].replace(0, np.nan)
    return p


def player_ratings(expected: pd.DataFrame, log=print) -> pd.DataFrame:
    players = pd.read_csv(PROCESSED / "match_players.csv")
    games = pd.read_csv(PROCESSED / "match_games.csv")
    p = player_metrics(players)
    stats = sorted({s for w in ROLE_WEIGHTS.values() for s in w})
    # z-scores within role across every 2026 game (domestic + international), one common scale.
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
    p = p.merge(games[["game_uid", "event", "league", "series_id", "stage", "date"]], on="game_uid", how="left")
    p = expected_player_ratings(p, log=log)
    keep = ["game_uid", "event", "league", "series_id", "stage", "date", "team", "opponent", "side", "role", "player", "champion", "result",
            "p_win", "vs_expected", "rating", "stat_rating", "mvp", "Kills", "Deaths", "Assists", "CSM", "GOLD%",
            "DMG%", "DPM", "VSPM", "KP%", "GD@15", "CSD@15", "XPD@15", "Golds", "Total damage to Champion",
            "Vision Score", "Solo kills", "duration", "gold_share", "dmg_game", "gold_game", "opp_dmg", "opp_gold",
            "dmg_per_gold", "expected_rating", "rating_vs_expected", "player_prior",
            "player_games", "champ_games", "champ_prior"] + [f"z_{s}" for s in stats]
    out = p[keep]
    out.to_csv(PROCESSED / "player_ratings.csv", index=False)
    log(f"  rated {len(out)} player-games across {out.game_uid.nunique()} games")
    return out


# --- expected player rating ------------------------------------------------------------------

PLAYER_PRIOR_GAMES, CHAMP_PRIOR_GAMES, REGION_PRIOR_GAMES = 5, 3, 20


def _prior_by(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Sum and count of rating for each key from strictly earlier days (same-day games excluded)."""
    daily = df.groupby(keys + ["date"]).rating.agg(["sum", "count"]).reset_index().sort_values("date")
    daily[["cum_sum", "cum_n"]] = daily.groupby(keys)[["sum", "count"]].cumsum()
    daily["prior_sum"] = daily.cum_sum - daily["sum"]
    daily["prior_n"] = daily.cum_n - daily["count"]
    return daily[keys + ["date", "prior_sum", "prior_n"]]


def region_strength_prior(games: pd.DataFrame) -> pd.DataFrame:
    """Each region's win rate in cross-region games before each date (shrunk toward 50%)."""
    g = all_games()[["game_uid", "date", "team", "region", "result"]]
    pair = g.merge(g, on="game_uid", suffixes=("", "_opp"))
    pair = pair[(pair.team != pair.team_opp) & (pair.region != pair.region_opp)]
    rows = []
    for day in sorted(pd.to_datetime(games.date).unique()):
        past = pair[pair.date < day].groupby("region").result.agg(["sum", "count"])
        for region, r in past.iterrows():
            rows.append(dict(date=day, region=region,
                             region_strength=(r["sum"] + REGION_PRIOR_GAMES * 0.5) / (r["count"] + REGION_PRIOR_GAMES)))
    return pd.DataFrame(rows)


def expected_player_ratings(p: pd.DataFrame, log=print) -> pd.DataFrame:
    """Pre-game expected rating from the player's earlier games, their record on this champion,
    the team's win chance, and (for international games) how strong their region has been.

    Fit on every game before Worlds; Worlds is scored against simpler baselines."""
    from sklearn.linear_model import LinearRegression
    p = p.copy()
    p["date"] = pd.to_datetime(p.date)
    pl = _prior_by(p, ["player"])
    ch = _prior_by(p, ["player", "champion"]).rename(columns={"prior_sum": "c_sum", "prior_n": "c_n"})
    p = p.merge(pl, on=["player", "date"], how="left").merge(ch, on=["player", "champion", "date"], how="left")
    p["player_games"] = p.prior_n.fillna(0).astype(int)
    p["champ_games"] = p.c_n.fillna(0).astype(int)
    p["player_prior"] = (p.prior_sum.fillna(0) + PLAYER_PRIOR_GAMES * 3.0) / (p.player_games + PLAYER_PRIOR_GAMES)
    p["champ_prior"] = (p.c_sum.fillna(0) + CHAMP_PRIOR_GAMES * p.player_prior) / (p.champ_games + CHAMP_PRIOR_GAMES)
    regions = team_regions(load_games())
    p["region"] = p.team.map(regions).fillna("Other")
    rs = region_strength_prior(p)
    p = p.merge(rs, on=["date", "region"], how="left")
    p["region_strength"] = p.region_strength.fillna(0.5)
    p["intl"] = p.event.notna().astype(float)

    def features(df, champion=True):
        X = pd.DataFrame({
            "form": df.player_prior - 3, "win_chance": df.p_win - 0.5, "intl": df.intl,
            "intl_form": df.intl * (df.player_prior - 3), "intl_region": df.intl * (df.region_strength - 0.5),
        })
        if champion:
            X["champ_edge"] = df.champ_prior - df.player_prior
            X["champ_experience"] = np.log1p(df.champ_games)
        return X

    train = p[p.event != TEST_EVENT]
    test = p[p.event == TEST_EVENT]
    full = LinearRegression().fit(features(train), train.rating)
    no_champ = LinearRegression().fit(features(train, False), train.rating)
    mae = lambda pred: float(np.mean(np.abs(np.clip(pred, 1, 5) - test.rating)))
    corr = lambda pred: float(np.corrcoef(pred, test.rating)[0, 1])
    pred_full = full.predict(features(test))
    evaluation = dict(
        n=int(len(test)),
        coefficients=dict(zip(features(train).columns, map(float, full.coef_))),
        mae=dict(model=mae(pred_full), without_champion_history=mae(no_champ.predict(features(test, False))),
                 everyone_average=mae(np.full(len(test), train.rating.mean()))),
        correlation=dict(model=corr(pred_full), without_champion_history=corr(no_champ.predict(features(test, False)))),
    )
    (PROCESSED / "expected_rating_eval.json").write_text(json.dumps(evaluation, indent=2))
    p["expected_rating"] = np.clip(full.predict(features(p)), 1, 5).round(1)
    p["rating_vs_expected"] = (p.rating - p.expected_rating).round(1)
    p["date"] = p.date.dt.strftime("%Y-%m-%d")
    log(f"  expected rating on Worlds ({len(test)} player-games): error {evaluation['mae']['model']:.2f} "
        f"(without champion history {evaluation['mae']['without_champion_history']:.2f}, "
        f"everyone average {evaluation['mae']['everyone_average']:.2f})")
    return p


def run(log=print):
    expected = expected_results(log=log)
    return player_ratings(expected, log=log)
