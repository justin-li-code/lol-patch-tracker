"""What regions and teams prioritise, and whether that priority wins games.

gol.gg lists each team's picks and bans in draft order (verified: average
champion presence falls monotonically from pick 1 to pick 5 and from ban 1
to ban 5), so "early" priority means picked in a team's first two pick
slots or banned in its first three (phase-one) bans.

Outputs (data/processed/):
    region_champions.csv   region x champion: presence/early priority vs the 4-league average
    team_champions.csv     team x champion: pick share, early picks, bans by/against, win rate
    event_region_champions.csv  what each region played at international events
    priority_effects.json  the three "does priority matter?" tests
"""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .analyze import load_games
from .config import EXTRA_TEAM_REGIONS, MIN_GAMES_PER_PATCH, PROCESSED, patch_key

EARLY_PICKS, EARLY_BANS = 2, 3
COMFORT_MIN_PRIOR_PICKS = 3
REGION_FAVORITE_LIFT = 0.05
N_BOOT = 2000


def long_draft(games: pd.DataFrame) -> pd.DataFrame:
    """One row per pick or ban: game, team, champion, action, slot (1-based), result."""
    rows = []
    for r in games.itertuples(index=False):
        for action, field in (("pick", r.picks), ("ban", r.bans)):
            if not isinstance(field, str) or not field:
                continue
            for slot, champ in enumerate(field.split("|"), start=1):
                rows.append((r.game_uid, r.patch, r.date, r.league, r.team, champ, action, slot, r.result))
    d = pd.DataFrame(rows, columns=["game_uid", "patch", "date", "league", "team", "champion", "action", "slot", "result"])
    d["early"] = np.where(d.action == "pick", d.slot <= EARLY_PICKS, d.slot <= EARLY_BANS)
    return d


def team_regions(domestic: pd.DataFrame) -> dict:
    regions = domestic.groupby("team").league.agg(lambda s: s.mode().iloc[0]).to_dict()
    return {**EXTRA_TEAM_REGIONS, **regions}


def _per_game_rates(d: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Presence / early priority per game for each `by` + champion group."""
    games = d.groupby(by).game_uid.nunique().rename("games")
    g = d.groupby(by + ["champion"]).agg(
        presence_n=("game_uid", "nunique"),
        picks=("action", lambda s: (s == "pick").sum()),
        bans=("action", lambda s: (s == "ban").sum()),
        early_n=("early", "sum"),
    ).reset_index().join(games, on=by)
    g["presence"] = g.presence_n / g.games
    g["early_priority"] = g.early_n / g.games
    return g


def region_champions(d: pd.DataFrame) -> pd.DataFrame:
    """Region presence vs what the 4-league average would predict on the same patches."""
    reg = _per_game_rates(d, ["league", "patch"])
    glob = _per_game_rates(d.assign(league="ALL"), ["league", "patch"]).set_index(["patch", "champion"])
    reg = reg.join(glob[["presence", "early_priority"]].add_prefix("global_"), on=["patch", "champion"])
    # Champions a region never touched on a patch still have an expected rate there.
    region_games = reg.drop_duplicates(["league", "patch"])[["league", "patch", "games"]]
    expected = region_games.merge(glob.reset_index()[["patch", "champion", "presence", "early_priority"]], on="patch")
    expected = expected.assign(exp_p=expected.presence * expected.games, exp_e=expected.early_priority * expected.games)
    exp = expected.groupby(["league", "champion"])[["exp_p", "exp_e"]].sum()
    obs = reg.groupby(["league", "champion"]).agg(presence_n=("presence_n", "sum"), early_n=("early_n", "sum"),
                                                  picks=("picks", "sum"), bans=("bans", "sum"))
    tot = region_games.groupby("league").games.sum()
    out = exp.join(obs, how="outer").fillna(0).reset_index()
    out["games"] = out.league.map(tot)
    out["presence"] = out.presence_n / out.games
    out["expected_presence"] = out.exp_p / out.games
    out["early_priority"] = out.early_n / out.games
    out["expected_early"] = out.exp_e / out.games
    out["presence_lift"] = out.presence - out.expected_presence
    out["early_lift"] = out.early_priority - out.expected_early
    return out.drop(columns=["exp_p", "exp_e"]).sort_values(["league", "presence"], ascending=[True, False])


def team_champions(d: pd.DataFrame) -> pd.DataFrame:
    team_games = d.groupby("team").game_uid.nunique().rename("team_games")
    own = d.groupby(["team", "champion"]).agg(
        league=("league", "first"),
        picks=("action", lambda s: (s == "pick").sum()),
        early_picks=("early", lambda s: 0),  # filled below
        bans_by=("action", lambda s: (s == "ban").sum()),
    )
    picks = d[d.action == "pick"]
    own["early_picks"] = picks[picks.early].groupby(["team", "champion"]).size().reindex(own.index).fillna(0)
    own["wins"] = picks.groupby(["team", "champion"]).result.sum().reindex(own.index).fillna(0)
    # Bans *against* a team: the opponent's bans in the same game.
    opp = d[d.action == "ban"].merge(d[["game_uid", "team"]].drop_duplicates(), on="game_uid", suffixes=("_banner", ""))
    opp = opp[opp.team_banner != opp.team]
    against = opp.groupby(["team", "champion"]).size().rename("bans_against")
    out = own.join(against, how="outer").fillna({"picks": 0, "early_picks": 0, "bans_by": 0, "wins": 0, "bans_against": 0})
    counts = ["picks", "early_picks", "bans_by", "wins", "bans_against"]
    out[counts] = out[counts].astype(int)
    out = out.reset_index().join(team_games, on="team")
    out["league"] = out.groupby("team").league.transform(lambda s: s.dropna().iloc[0] if s.notna().any() else None)
    out["pick_share"] = out.picks / out.team_games
    # League-average pick share per team-game for the same champion (patch-weighted via the league's own games).
    lg = d[d.action == "pick"].groupby(["league", "champion"]).size() / (d.groupby("league").game_uid.nunique() * 2)
    out["league_pick_share"] = [lg.get((l, c), 0.0) for l, c in zip(out.league, out.champion)]
    out["pick_lift"] = out.pick_share - out.league_pick_share
    out["win_rate"] = np.where(out.picks > 0, out.wins / out.picks.replace(0, np.nan), np.nan)
    out["against_rate"] = out.bans_against / out.team_games
    return out.sort_values(["team", "picks"], ascending=[True, False])


def _bootstrap_mean(values: np.ndarray, weights: np.ndarray | None = None, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float)
    w = np.ones_like(values) if weights is None else np.asarray(weights, float)
    est = np.average(values, weights=w)
    idx = rng.integers(0, len(values), (N_BOOT, len(values)))
    boots = (values[idx] * w[idx]).sum(1) / w[idx].sum(1)
    return est, *np.percentile(boots, [2.5, 97.5])


def meta_draft_effect(games: pd.DataFrame) -> dict:
    """Does drafting the more contested champions win, controlling for team strength?

    "Contested" is measured on the league's *previous* patch: same-patch presence
    would leak results, because teams copy whatever wins early in a patch.
    """
    d = long_draft(games)
    rates = _per_game_rates(d, ["league", "patch"])
    rates = rates[rates.games >= MIN_GAMES_PER_PATCH]
    prev = {}
    for league, g in rates.groupby("league"):
        ps = sorted(g.patch.unique(), key=patch_key)
        prev.update({(league, p): q for q, p in zip(ps, ps[1:])})
    pres = rates.set_index(["league", "patch", "champion"]).presence
    picks = d[(d.action == "pick")].copy()
    picks["prev_patch"] = [prev.get(k) for k in zip(picks.league, picks.patch)]
    picks = picks[picks.prev_patch.notna()]
    picks["meta"] = [pres.get(k, 0.0) for k in zip(picks.league, picks.prev_patch, picks.champion)]
    tg = picks.groupby(["game_uid", "team"]).agg(meta=("meta", "mean"), result=("result", "first")).reset_index()
    # Leave-one-out season win rate as team strength.
    tot = tg.groupby("team").result.agg(["sum", "count"])
    tg["strength"] = (tg.team.map(tot["sum"]) - tg.result) / (tg.team.map(tot["count"]) - 1)
    # Both orientations of every game, so summaries are symmetric. (gol.gg lists
    # the series winner first on summary pages, so "first row" would be biased.)
    pair = tg.merge(tg, on="game_uid", suffixes=("", "_opp"))
    pair = pair[pair.team != pair.team_opp].copy()
    pair["meta_diff"] = pair.meta - pair.meta_opp
    pair["strength_diff"] = pair.strength - pair.strength_opp
    raw = (pair.result[pair.meta_diff > 0].mean(), int((pair.meta_diff > 0).sum()))
    # The regression needs independent rows: one randomly oriented row per game.
    one = pair.sample(frac=1, random_state=0).drop_duplicates("game_uid")
    X = sm.add_constant(one[["meta_diff", "strength_diff"]].assign(meta_diff=lambda x: x.meta_diff * 10))
    fit = sm.Logit(one.result, X).fit(disp=0)
    lo, hi = fit.conf_int().loc["meta_diff"]
    # Effect of a +10pp average-presence edge for evenly matched teams, as a win probability.
    to_p = lambda b: 1 / (1 + np.exp(-b))
    buckets = pd.cut(pair.meta_diff, [-1, -0.1, -0.03, 0.03, 0.1, 1],
                     labels=["much less meta", "less meta", "even", "more meta", "much more meta"])
    return dict(
        games=len(one),
        win_rate_more_meta=raw[0],
        n_more_meta=raw[1],
        adjusted_win_prob_plus10pp=to_p(fit.params["meta_diff"]),
        adjusted_ci=[to_p(lo), to_p(hi)],
        p_value=float(fit.pvalues["meta_diff"]),
        strength_coef=float(fit.params["strength_diff"]),
        by_bucket=pair.groupby(buckets, observed=True).result.agg(["mean", "size"]).reset_index()
                     .rename(columns={"meta_diff": "bucket", "mean": "win_rate", "size": "n"}).to_dict("records"),
    )


def comfort_effect(games: pd.DataFrame) -> dict:
    """Win rate on champions a team had already played >= N times before this game, vs other picks.

    Comfort is defined only from *earlier* games, and compared within each team
    so team strength cancels out.
    """
    picks = long_draft(games).query("action == 'pick'").sort_values(["date", "game_uid"])
    picks["prior"] = picks.groupby(["team", "champion"]).cumcount()
    # cumcount counts earlier rows; games on the same date are rare enough to ignore ordering within a day.
    picks["comfort"] = picks.prior >= COMFORT_MIN_PRIOR_PICKS
    per_team = picks.groupby(["team", "comfort"]).result.agg(["mean", "size"]).unstack()
    per_team = per_team.dropna()
    per_team = per_team[(per_team[("size", True)] >= 10) & (per_team[("size", False)] >= 10)]
    diff = per_team[("mean", True)] - per_team[("mean", False)]
    weight = 1 / (1 / per_team[("size", True)] + 1 / per_team[("size", False)])
    est, lo, hi = _bootstrap_mean(diff.values, weight.values)
    return dict(
        teams=len(per_team), comfort_picks=int(picks.comfort.sum()), other_picks=int((~picks.comfort).sum()),
        win_rate_comfort=picks.result[picks.comfort].mean(), win_rate_other=picks.result[~picks.comfort].mean(),
        within_team_diff=est, ci=[lo, hi],
        per_team=diff.rename("diff").reset_index().assign(n_comfort=per_team[("size", True)].values,
                                                           n_other=per_team[("size", False)].values).to_dict("records"),
    )


def region_favorite_effect(event_games: pd.DataFrame, region_champs: pd.DataFrame, regions: dict) -> dict:
    """At international events: does a region win more on champions it prioritised at home?

    Compared within region (its favourites vs its other picks at the events),
    so regional strength cancels out.
    """
    picks = long_draft(event_games.rename(columns={"league": "event"}).assign(league=lambda x: x.event))
    picks = picks[picks.action == "pick"].copy()
    picks["region"] = picks.team.map(regions)
    # Home priority only means something against a team from another region.
    n_regions = picks.groupby("game_uid").region.transform("nunique")
    picks = picks[n_regions == 2]
    lift = region_champs.set_index(["league", "champion"]).presence_lift
    picks["home_lift"] = [lift.get((r, c), np.nan) for r, c in zip(picks.region, picks.champion)]
    picks = picks[picks.home_lift.notna()]
    picks["favorite"] = picks.home_lift >= REGION_FAVORITE_LIFT
    rows = []
    for region, g in picks.groupby("region"):
        fav, other = g[g.favorite], g[~g.favorite]
        if len(fav) >= 5 and len(other) >= 5:
            rows.append(dict(region=region, n_favorite=len(fav), n_other=len(other),
                             win_rate_favorite=fav.result.mean(), win_rate_other=other.result.mean()))
    t = pd.DataFrame(rows)
    if t.empty:
        return dict(regions=[], within_region_diff=np.nan, ci=[np.nan, np.nan])
    t["diff"] = t.win_rate_favorite - t.win_rate_other
    # Pick-level bootstrap within region keeps the comparison within region.
    rng = np.random.default_rng(0)
    boots = []
    groups = {r: (g.result[g.favorite].values, g.result[~g.favorite].values)
              for r, g in picks.groupby("region") if r in set(t.region)}
    w = t.set_index("region").pipe(lambda x: 1 / (1 / x.n_favorite + 1 / x.n_other))
    for _ in range(N_BOOT):
        diffs = [rng.choice(f, len(f)).mean() - rng.choice(o, len(o)).mean() for f, o in groups.values()]
        boots.append(np.average(diffs, weights=w.reindex(list(groups)).values))
    return dict(regions=t.to_dict("records"), within_region_diff=float(np.average(t["diff"], weights=w.reindex(t.region).values)),
                ci=list(np.percentile(boots, [2.5, 97.5])), favorite_lift=REGION_FAVORITE_LIFT,
                favorite_picks=int(picks.favorite.sum()), total_picks=len(picks))


def event_region_champions(event_games: pd.DataFrame, regions: dict) -> pd.DataFrame:
    """Per event and region: presence of each champion in that region's games (picks+bans by the region's teams)."""
    d = long_draft(event_games)
    d["region"] = d.team.map(regions).fillna("Other")
    games = d.groupby(["league", "region"]).game_uid.nunique().rename("region_games")
    g = d.groupby(["league", "region", "champion"]).agg(
        picks=("action", lambda s: (s == "pick").sum()), bans=("action", lambda s: (s == "ban").sum()),
        early=("early", "sum"), wins=("result", lambda s: s[d.loc[s.index, "action"] == "pick"].sum()),
    ).reset_index().join(games, on=["league", "region"])
    # Per region-game rates: a team picks/bans a champion at most once per game.
    g["pick_ban_rate"] = (g.picks + g.bans) / g.region_games
    g["early_rate"] = g.early / g.region_games
    g["win_rate"] = np.where(g.picks > 0, g.wins / g.picks.replace(0, np.nan), np.nan)
    return g.rename(columns={"league": "event"}).sort_values(["event", "region", "pick_ban_rate"], ascending=[True, True, False])


def run(log=print) -> dict:
    domestic = load_games()
    events = pd.read_csv(PROCESSED / "event_games.csv", dtype={"patch": str, "series_id": str})
    regions = team_regions(domestic)

    d = long_draft(domestic)
    rc = region_champions(d)
    rc.to_csv(PROCESSED / "region_champions.csv", index=False)
    tc = team_champions(d)
    tc.to_csv(PROCESSED / "team_champions.csv", index=False)
    erc = event_region_champions(events, regions)
    erc.to_csv(PROCESSED / "event_region_champions.csv", index=False)

    effects = dict(
        meta_draft=meta_draft_effect(domestic),
        comfort=comfort_effect(domestic),
        region_favorites=region_favorite_effect(events, rc, regions),
    )
    (PROCESSED / "priority_effects.json").write_text(json.dumps(effects, indent=2, default=float))
    (PROCESSED / "team_regions.json").write_text(json.dumps(regions, indent=2))

    m, c, r = effects["meta_draft"], effects["comfort"], effects["region_favorites"]
    log(f"Meta drafts: team with the more contested picks won {m['win_rate_more_meta']:.1%} of {m['n_more_meta']} games; "
        f"controlling for team strength, a +10pp edge -> {m['adjusted_win_prob_plus10pp']:.1%} "
        f"(95% CI {m['adjusted_ci'][0]:.1%}-{m['adjusted_ci'][1]:.1%}, p={m['p_value']:.3f})")
    log(f"Comfort picks (played >= {COMFORT_MIN_PRIOR_PICKS}x before): {c['win_rate_comfort']:.1%} vs {c['win_rate_other']:.1%}; "
        f"within-team difference {c['within_team_diff']:+.1%} (95% CI {c['ci'][0]:+.1%} to {c['ci'][1]:+.1%}, {c['teams']} teams)")
    if r["regions"]:
        log(f"Regional favourites at internationals: within-region difference {r['within_region_diff']:+.1%} "
            f"(95% CI {r['ci'][0]:+.1%} to {r['ci'][1]:+.1%}; {r['favorite_picks']} favourite picks of {r['total_picks']})")
        log(pd.DataFrame(r["regions"]).round(3).to_string(index=False))
    return effects
