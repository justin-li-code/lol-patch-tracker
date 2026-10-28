"""Upcoming Worlds matches and matchup previews.

Precomputes everything a preview needs, so the dashboard can preview any pairing instantly:
  team_strength.csv   current strength per team (same team + region model as the expected
                      results, fit on every 2026 game, recent games weighted more)
  team_pools.csv      each team's recent champion picks by role (decayed weights)
  team_bans.csv       what each team bans, and what opponents ban against it (decayed)
  worlds_role_meta.csv  each champion's share of picks per role at Worlds
  worlds_standings.csv  Swiss record, status, and chances to reach each round / win Worlds
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


def swiss_bracket(games: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """Every Swiss match (played, then scheduled) with its round and the record both teams brought in.

    Rounds aren't labelled on gol.gg, so they're rebuilt by playing the series back in order:
    a match is round n+1 for teams that had played n series, in the pool of their shared record.
    """
    sw = games[(games.event == "worlds") & games.tournament.str.contains("Main Event")]
    record: dict[str, list[int]] = {}
    rows = []
    for sid, s in sorted(sw.groupby("series_id"), key=lambda x: (x[1].date.min(), x[0])):
        a, b = s.blue.iloc[0], s.red.iloc[0]
        wins = s.winner.value_counts()
        ra, rb = record.setdefault(a, [0, 0]), record.setdefault(b, [0, 0])
        rows.append(dict(round=sum(ra) + 1, pool=f"{ra[0]}–{ra[1]}", team_a=a, team_b=b,
                         score_a=int(wins.get(a, 0)), score_b=int(wins.get(b, 0)), winner=wins.idxmax(),
                         date=s.date.min(), pending=False))
        ra[0 if wins.idxmax() == a else 1] += 1
        rb[0 if wins.idxmax() == b else 1] += 1
    for m in sched.itertuples():
        ra = record.get(m.team_a, [0, 0])
        rows.append(dict(round=sum(ra) + 1, pool=f"{ra[0]}–{ra[1]}", team_a=m.team_a, team_b=m.team_b,
                         score_a=None, score_b=None, winner=None, date=str(m.date), pending=True))
    out = pd.DataFrame(rows)
    out.attrs["records"] = {t: tuple(r) for t, r in record.items()}
    return out


def tournament_odds(st: pd.DataFrame, strength: pd.Series, regions: pd.Series, n_sims: int = 20000,
                    seed: int = 0) -> pd.DataFrame:
    """Play out the rest of Worlds many times: remaining Swiss rounds, then a best-of-five bracket.

    Swiss: teams with the same record meet (rematches avoided when possible), best of three.
    Quarterfinal draw (the real one isn't known until the Swiss ends), following recent Worlds
    rules: the best records face the worst (3-0 vs 3-2, 3-1 vs 3-1), same-region quarterfinals
    avoided when possible; the bracket halves are random. Every knockout match is best of five.
    Returns, per team, the chance to reach the quarterfinals, semifinals, final, and to win.
    """
    g = pd.read_csv(PROCESSED / "match_games.csv", dtype={"event": str, "series_id": str})
    sw = g[(g.event == "worlds") & g.tournament.str.contains("Main Event")]
    played = {frozenset((s.blue.iloc[0], s.red.iloc[0])) for _, s in sw.groupby("series_id")}
    rng = np.random.default_rng(seed)
    alive0 = st[st.status == "Still playing"]
    done0 = st[st.status == "Advanced"]
    counts = {k: pd.Series(0.0, index=st.team) for k in ("p_advance", "p_semis", "p_final", "p_title")}
    cache = {}

    def series_winner(a, b, best_of):
        key = (a, b, best_of)
        if key not in cache:
            cache[key] = series_prob(win_prob(strength, a, b), best_of)
        return (a, b) if rng.random() < cache[key] else (b, a)

    def pair_up(teams, avoid):
        """Pair a list in order (first with last, ...) after shuffling to dodge `avoid` pairs."""
        best = None
        for _try in range(20):
            pairs = [(teams[i], teams[len(teams) - 1 - i]) for i in range(len(teams) // 2)]
            bad = sum(avoid(a, b) for a, b in pairs)
            if best is None or bad < best[0]:
                best = (bad, pairs)
            if bad == 0:
                break
            # Shuffle within each record tier only, keeping "best vs worst" intact.
            tiers = {}
            for t in teams:
                tiers.setdefault(record[t], []).append(t)
            teams = [t for r in sorted(tiers, key=lambda r: (-r[0], r[1])) for t in rng.permutation(tiers[r])]
        return best[1]

    for _ in range(n_sims):
        record = {t: (w, l) for t, w, l in zip(st.team, st.wins, st.losses)}
        live = {t: [w, l] for t, w, l in zip(alive0.team, alive0.wins, alive0.losses)}
        qualified = list(done0.team)
        while live:
            groups = {}
            for t, (w, l) in live.items():
                groups.setdefault((w, l), []).append(t)
            for group in groups.values():
                rng.shuffle(group)
                for _try in range(10):
                    pairs = [(group[i], group[i + 1]) for i in range(0, len(group) - 1, 2)]
                    if all(frozenset(p) not in played for p in pairs):
                        break
                    rng.shuffle(group)
                for a, b in pairs:
                    w_, l_ = series_winner(a, b, 3)
                    live[w_][0] += 1
                    live[l_][1] += 1
            for t in list(live):
                record[t] = tuple(live[t])
                if live[t][0] >= 3:
                    qualified.append(t)
                    del live[t]
                elif live[t][1] >= 3:
                    del live[t]
        for t in qualified:
            counts["p_advance"][t] += 1
        # Quarterfinals: best records vs worst, avoiding same-region pairs where possible.
        seeded = sorted(qualified, key=lambda t: (-record[t][0], record[t][1], rng.random()))
        qf = pair_up(seeded, lambda a, b: regions.get(a) == regions.get(b))
        rng.shuffle(qf)
        semis = [series_winner(a, b, 5)[0] for a, b in qf]
        for t in semis:
            counts["p_semis"][t] += 1
        finalists = [series_winner(semis[i], semis[i + 1], 5)[0] for i in (0, 2)]
        for t in finalists:
            counts["p_final"][t] += 1
        counts["p_title"][series_winner(*finalists, 5)[0]] += 1
    return pd.DataFrame({k: v / n_sims for k, v in counts.items()})


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
    ts = team_strengths(log=log).set_index("team")
    pools_and_bans(log=log)
    st = standings(log=log)
    odds = tournament_odds(st, ts.strength, ts.region)
    st = st.merge(odds, left_on="team", right_index=True, how="left")
    st.to_csv(PROCESSED / "worlds_standings.csv", index=False)
    log("  chance to win Worlds: " + ", ".join(
        f"{t} {p:.0%}" for t, p in st.set_index("team").p_title.sort_values(ascending=False).head(8).items()))
    schedule(log=log)
