"""Upcoming Worlds matches and matchup previews.

Precomputes everything a preview needs, so the dashboard can preview any pairing instantly:
  team_strength.csv   current strength per team (same team + region model as the expected
                      results, fit on every 2026 game, recent games weighted more)
  team_pools.csv      each team's recent champion picks by role (decayed weights)
  team_bans.csv       what each team bans, and what opponents ban against it (decayed)
  worlds_role_meta.csv  each champion's share of picks per role at Worlds
  worlds_standings.csv  Swiss record and status per team
  upcoming.csv        scheduled matches: from Leaguepedia when it answers, otherwise from
                      data/overrides/upcoming.csv (team_a, team_b, date, best_of, stage)
"""
import json

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression

from .config import OVERRIDES, PROCESSED
from .ratings import GAME_HALF_LIFE, _decay, _design, all_games, game_features, to_pairs
from .regions import long_draft

POOL_HALF_LIFE = 30  # days: recent picks matter most
BAN_HALF_LIFE = 30
LEAGUEPEDIA = "https://lol.fandom.com/api.php"
UPCOMING_TEMPLATE = OVERRIDES / "upcoming.csv"


def team_strengths(log=print) -> pd.DataFrame:
    settings = json.loads((PROCESSED / "expected_eval.json").read_text())["settings"]
    games = game_features(all_games())
    pairs = to_pairs(games)
    today = pairs.date.max()
    teams = sorted(set(pairs.team_a) | set(pairs.team_b))
    regions = sorted(set(pairs.region_a) | set(pairs.region_b))
    model = LogisticRegression(C=settings["C"], fit_intercept=False, max_iter=3000)
    model.fit(_design(pairs, teams, regions, settings["region_scale"], False), pairs.y,
              sample_weight=_decay((today - pairs.date).dt.days, GAME_HALF_LIFE))
    coef = model.coef_[0]
    team_region = games.drop_duplicates("team").set_index("team").region
    rs = settings["region_scale"]
    out = pd.DataFrame({"team": teams, "region": [team_region.get(t, "Other") for t in teams]})
    out["strength"] = [coef[i] + rs * coef[len(teams) + regions.index(r)] for i, r in enumerate(out.region)]
    out.to_csv(PROCESSED / "team_strength.csv", index=False)
    log(f"  team strengths for {len(out)} teams")
    return out


def win_prob(strength: pd.Series, a: str, b: str) -> float:
    return float(1 / (1 + np.exp(-(strength.get(a, 0.0) - strength.get(b, 0.0)))))


def series_prob(p: float, best_of: int) -> float:
    """Chance to win a best-of-N series when each game is won with probability p."""
    need = best_of // 2 + 1
    from math import comb
    return float(sum(comb(need - 1 + k, k) * p ** need * (1 - p) ** k for k in range(need)))


def pools_and_bans(log=print):
    games = all_games()
    players = pd.read_csv(PROCESSED / "players.csv")
    today = games.date.max()
    dates = games.drop_duplicates("game_uid").set_index("game_uid").date
    pl = players.assign(date=players.game_uid.map(dates)).dropna(subset=["date"])
    pl["w"] = _decay((today - pl.date).dt.days, POOL_HALF_LIFE)
    pools = pl.groupby(["team", "role", "champion"]).agg(weight=("w", "sum"), games=("game_uid", "nunique"),
                                                         last_played=("date", "max")).reset_index()
    pools["share"] = pools.weight / pools.groupby(["team", "role"]).weight.transform("sum")
    pools.to_csv(PROCESSED / "team_pools.csv", index=False)

    d = long_draft(games.assign(date=games.date.dt.strftime("%Y-%m-%d")))
    d["date"] = pd.to_datetime(d.date)
    d["w"] = _decay((today - d.date).dt.days, BAN_HALF_LIFE)
    team_games = d.drop_duplicates(["game_uid", "team"]).groupby("team").w.sum()
    bans = d[d.action == "ban"]
    own = bans.groupby(["team", "champion"]).w.sum().reset_index().rename(columns={"w": "w_own"})
    own["own_rate"] = own.w_own / own.team.map(team_games)
    vs = bans.merge(d[["game_uid", "team"]].drop_duplicates(), on="game_uid", suffixes=("_banner", ""))
    vs = vs[vs.team_banner != vs.team].groupby(["team", "champion"]).w.sum().reset_index()
    vs["against_rate"] = vs.w / vs.team.map(team_games)
    out = own[["team", "champion", "own_rate"]].merge(vs[["team", "champion", "against_rate"]],
                                                      on=["team", "champion"], how="outer").fillna(0)
    out.to_csv(PROCESSED / "team_bans.csv", index=False)

    w = players[players.tournament.str.startswith("Worlds 2026")]
    meta = (w.groupby(["role", "champion"]).size() / w.groupby("role").size()).rename("share").reset_index()
    meta.to_csv(PROCESSED / "worlds_role_meta.csv", index=False)
    log(f"  champion pools for {pools.team.nunique()} teams, ban tendencies, Worlds meta by role")


def standings(log=print) -> pd.DataFrame:
    g = pd.read_csv(PROCESSED / "match_games.csv", dtype={"event": str, "series_id": str})
    sw = g[(g.event == "worlds") & g.tournament.str.contains("Main Event")]
    rows = []
    for sid, s in sw.groupby("series_id"):
        wins = s.winner.value_counts()
        teams = set(s.blue) | set(s.red)
        winner = wins.idxmax()
        for t in teams:
            rows.append(dict(team=t, series_id=sid, won=int(t == winner), date=s.date.min()))
    r = pd.DataFrame(rows)
    st = r.groupby("team").agg(wins=("won", "sum"), played=("won", "size"), last=("date", "max")).reset_index()
    st["losses"] = st.played - st.wins
    st["status"] = np.where(st.wins >= 3, "Advanced", np.where(st.losses >= 3, "Eliminated", "Still playing"))
    st = st.sort_values(["wins", "losses"], ascending=[False, True])
    st.to_csv(PROCESSED / "worlds_standings.csv", index=False)
    log(f"  Swiss standings: {(st.status == 'Still playing').sum()} teams still playing")
    return st


def swiss_odds(st: pd.DataFrame, strength: pd.Series, n_sims: int = 20000, seed: int = 0) -> pd.Series:
    """Chance each team still playing reaches 3 wins, simulating the remaining Swiss rounds.

    Each round pairs teams with the same record at random (avoiding rematches when possible);
    remaining matches are best-of-three, each game won with the model's win chance.
    """
    g = pd.read_csv(PROCESSED / "match_games.csv", dtype={"event": str, "series_id": str})
    sw = g[(g.event == "worlds") & g.tournament.str.contains("Main Event")]
    played = {frozenset((s.blue.iloc[0], s.red.iloc[0])) for _, s in sw.groupby("series_id")}
    rng = np.random.default_rng(seed)
    alive0 = st[st.status == "Still playing"]
    advanced = pd.Series(0.0, index=st.team)
    advanced[st[st.status == "Advanced"].team] = n_sims
    p_series = {}
    for _ in range(n_sims):
        rec = {t: [w, l] for t, w, l in zip(alive0.team, alive0.wins, alive0.losses)}
        while rec:
            groups = {}
            for t, (w, l) in rec.items():
                groups.setdefault((w, l), []).append(t)
            for group in groups.values():
                rng.shuffle(group)
                # Re-shuffle a few times to avoid rematches, then accept whatever pairing we have.
                for _try in range(10):
                    pairs = [(group[i], group[i + 1]) for i in range(0, len(group) - 1, 2)]
                    if all(frozenset(p) not in played for p in pairs):
                        break
                    rng.shuffle(group)
                for a, b in pairs:
                    key = (a, b)
                    if key not in p_series:
                        p_series[key] = series_prob(win_prob(strength, a, b), 3)
                    winner, loser = (a, b) if rng.random() < p_series[key] else (b, a)
                    rec[winner][0] += 1
                    rec[loser][1] += 1
            for t in list(rec):
                if rec[t][0] >= 3:
                    advanced[t] += 1
                    del rec[t]
                elif rec[t][1] >= 3:
                    del rec[t]
    return (advanced / n_sims).rename("p_advance")


# Leaguepedia names that differ from gol.gg's (bracketed suffixes are stripped separately).
TEAM_ALIASES = {"LØS": "Los Grandes"}


def _team_name(name: str) -> str:
    import re
    name = re.sub(r"\s*\(.*\)$", "", str(name)).strip()
    return TEAM_ALIASES.get(name, name)


def schedule(log=print) -> pd.DataFrame:
    """Upcoming Worlds matches: Leaguepedia when it answers, else the hand-maintained file."""
    rows = []
    try:
        params = dict(action="cargoquery", format="json", tables="MatchSchedule=MS", limit=100,
                      fields="MS.Team1,MS.Team2,MS.DateTime_UTC,MS.BestOf,MS.Tab",
                      where="MS.OverviewPage LIKE '2026 Season World Championship%' AND MS.Winner IS NULL",
                      order_by="MS.DateTime_UTC")
        resp = requests.get(LEAGUEPEDIA, params=params, timeout=20,
                            headers={"User-Agent": "lol-patch-tracker (fan research project)"}).json()
        for item in resp.get("cargoquery", []):
            t = item["title"]
            if t.get("Team1") and t.get("Team2") and t["Team1"] != "TBD":
                rows.append(dict(team_a=_team_name(t["Team1"]), team_b=_team_name(t["Team2"]), date=t["DateTime UTC"][:16],
                                 best_of=int(t.get("BestOf") or 1), stage=t.get("Tab", "")))
        if "error" in resp:
            log(f"  schedule: Leaguepedia unavailable ({resp['error'].get('code')}), using {UPCOMING_TEMPLATE.name}")
    except (requests.RequestException, ValueError) as e:
        log(f"  schedule: Leaguepedia unreachable ({e.__class__.__name__}), using {UPCOMING_TEMPLATE.name}")
    if not rows:
        if not UPCOMING_TEMPLATE.exists():
            UPCOMING_TEMPLATE.parent.mkdir(parents=True, exist_ok=True)
            UPCOMING_TEMPLATE.write_text("date,stage,team_a,team_b,best_of\n")
        manual = pd.read_csv(UPCOMING_TEMPLATE)
        rows = manual.to_dict("records")
    up = pd.DataFrame(rows, columns=["date", "stage", "team_a", "team_b", "best_of"])
    known = set(pd.read_csv(PROCESSED / "team_strength.csv").team)
    unknown = sorted((set(up.team_a) | set(up.team_b)) - known)
    if unknown:
        log(f"  schedule: unrecognised team names {unknown}; add them to TEAM_ALIASES")
    up.to_csv(PROCESSED / "upcoming.csv", index=False)
    if len(up):  # keep a copy, so the page still works if Leaguepedia is rate-limiting next time
        up.to_csv(UPCOMING_TEMPLATE, index=False)
    log(f"  {len(up)} upcoming matches")
    return up


def run(log=print):
    strength = team_strengths(log=log).set_index("team").strength
    pools_and_bans(log=log)
    st = standings(log=log)
    odds = swiss_odds(st, strength)
    st = st.merge(odds, left_on="team", right_index=True, how="left")
    st.to_csv(PROCESSED / "worlds_standings.csv", index=False)
    log("  chance to reach the knockouts: " + ", ".join(
        f"{t} {p:.0%}" for t, p in st[st.status == "Still playing"].set_index("team").p_advance.items()))
    schedule(log=log)
