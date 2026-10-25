"""Ripple effects: how one champion's balance changes move *other* champions.

Pieces:
  * Champion metadata from Meraki Analytics (solo-queue positions, which seed
    the role inference; Riot class tags kept for reference), with Data Dragon
    tags for champions Meraki doesn't have yet.
  * Pro roles inferred from team compositions: every team fields exactly one
    top/jungle/mid/bot/support, so we assign each team's five champions to the
    five roles (best permutation under each champion's role distribution) and
    re-estimate the distributions, EM-style. Seeded with solo-queue positions,
    anchored with the single-game gol.gg pages that list champions in role
    order, and validated on those pages held out.
  * Substitutions: after a big move by a changed champion, which untouched
    same-role champion moved the other way.
  * Champion groups: as a chosen group rose, which named champions fell, and
    whether their patch-to-patch moves actually oppose the group's.

Outputs (data/processed/): champion_meta.csv, pick_roles.csv,
role_validation.json, substitutions.csv, substitute_pairs.csv
"""
import itertools
import json
import re

import numpy as np
import pandas as pd
import requests

from . import http
from .analyze import build_transitions, champion_stats, load_changes, load_games
from .assets import ASSETS, DDRAGON
from .champions import champ_key
from .config import MIN_GAMES_PER_PATCH, PROCESSED, patch_key
from .regions import long_draft

ROLES = ["top", "jungle", "mid", "bot", "support"]
MERAKI = "https://cdn.merakianalytics.com/riot/lol/resources/latest/en-US/champions.json"
POSITION_MAP = {"TOP": "top", "JUNGLE": "jungle", "MIDDLE": "mid", "BOTTOM": "bot", "SUPPORT": "support"}
TOP_LEVEL = {"TANK": "Tank", "FIGHTER": "Fighter", "ASSASSIN": "Slayer", "SLAYER": "Slayer", "MAGE": "Mage",
             "MARKSMAN": "Marksman", "SUPPORT": "Controller", "CONTROLLER": "Controller", "SPECIALIST": "Specialist"}
PRIOR_STRENGTH = 3.0   # pseudo-games of the solo-queue position prior
EM_ITERATIONS = 8
SUB_MIN_MOVE_A, SUB_MIN_MOVE_B = 0.10, 0.05


# --- classes --------------------------------------------------------------------

def _classify(roles: set, toughness) -> tuple[str | None, str]:
    """Meraki lists several (alphabetical) tags per champion, so pick by precedence:
    real tanks > marksmen > enchanters/catchers > divers/juggernauts > assassins/skirmishers > mages."""
    if roles & {"VANGUARD", "WARDEN"} or ("TANK" in roles and toughness == 3):
        return "Tank", next(iter(sorted(roles & {"VANGUARD", "WARDEN"})), "")
    for subs, cls in ((("MARKSMAN",), "Marksman"), (("ENCHANTER", "CATCHER"), "Controller"),
                      (("DIVER", "JUGGERNAUT"), "Fighter"), (("ASSASSIN", "SKIRMISHER"), "Slayer"),
                      (("BURST", "BATTLEMAGE", "ARTILLERY"), "Mage")):
        hit = sorted(roles & set(subs))
        if hit:
            return cls, hit[0]
    for tag, cls in TOP_LEVEL.items():
        if tag in roles:
            return cls, ""
    return None, ""


def champion_classes(log=print) -> pd.DataFrame:
    """champ_key -> class, subclass, toughness, solo-queue positions."""
    cache = ASSETS / "meraki_champions.json"
    if not cache.exists():
        ASSETS.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(requests.get(MERAKI, timeout=120).content)
    meraki = json.loads(cache.read_text())
    version = requests.get(f"{DDRAGON}/api/versions.json", timeout=30).json()[0]
    dd = requests.get(f"{DDRAGON}/cdn/{version}/data/en_US/champion.json", timeout=30).json()["data"]

    rows = {}
    for c in dd.values():
        tags = [t.upper() for t in c["tags"]]
        rows[champ_key(c["name"])] = dict(champion=c["name"], cls=TOP_LEVEL.get(tags[0], tags[0].title()),
                                          subclass="", toughness=np.nan, positions="", source="ddragon")
    for c in meraki.values():
        key = champ_key(c["name"])
        roles = {r.upper() for r in c.get("roles") or []}
        toughness = (c.get("attributeRatings") or {}).get("toughness", np.nan)
        cls, sub = _classify(roles, toughness)
        if cls is None and key in rows:
            cls = rows[key]["cls"]
        rows[key] = dict(champion=c["name"], cls=cls, subclass=sub.title(), toughness=toughness,
                         positions="|".join(POSITION_MAP[p] for p in c.get("positions") or [] if p in POSITION_MAP),
                         source="meraki")
    out = pd.DataFrame.from_dict(rows, orient="index").rename_axis("champ_key").reset_index()
    log(f"  classes for {len(out)} champions ({(out.source == 'meraki').sum()} from Meraki, rest Data Dragon tags)")
    return out


# --- roles ----------------------------------------------------------------------

def _true_roles() -> pd.DataFrame:
    """Role-ordered champions from cached single-game pages: one row per (game_uid, team, champion, role)."""
    from .events import EVENTS
    from .scrape_matches import list_series, list_tournaments, parse_game
    tournaments = list_tournaments() + [t for ev in EVENTS.values() for t in ev["tournaments"]]
    rows = []
    for t in tournaments:
        for s in list_series(t):
            if s["page"] != "game":
                continue
            html = http.get(f"https://gol.gg/game/stats/{s['series_id']}/page-game/") or ""
            players = list(re.finditer(r"players/player-stats/\d+/[^']*' title='[^']+ stats'", html))
            champs = []
            for m in players:
                alts = re.findall(r"alt='([^']+)'", html[max(0, m.start() - 1500):m.start()])
                champs.append(alts[-1] if alts else None)
            game = parse_game(html)
            if len(champs) != 10 or not game:
                continue
            for side, team_champs in zip(game[0]["sides"], (champs[:5], champs[5:])):
                for role, champ in zip(ROLES, team_champs):
                    rows.append((f"{s['series_id']}-1", side["team"], champ, role))
    return pd.DataFrame(rows, columns=["game_uid", "team", "champion", "role"])


def _assign(team_champs: list[str], logp: dict) -> tuple:
    """Most likely role permutation for one team's five champions."""
    best, best_score = None, -np.inf
    for perm in itertools.permutations(range(5)):
        score = sum(logp[c][r] for c, r in zip(team_champs, perm))
        if score > best_score:
            best, best_score = perm, score
    return best


def infer_roles(picks: pd.DataFrame, classes: pd.DataFrame, truth: pd.DataFrame,
                holdout: set | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """EM over team compositions. Returns (pick-level roles, champion role distribution)."""
    prior = {}
    pos = classes.set_index("champ_key").positions
    for champ in picks.champion.unique():
        listed = [p for p in str(pos.get(champ_key(champ), "") or "").split("|") if p]
        base = np.full(5, 0.2)
        if listed:
            base = np.full(5, 0.02)
            for p in listed:
                base[ROLES.index(p)] += 0.9 / len(listed)
        prior[champ] = base / base.sum() * PRIOR_STRENGTH

    known = truth[~truth.game_uid.isin(holdout or set())]
    known_role = {(g, t, c): ROLES.index(r) for g, t, c, r in known.itertuples(index=False)}
    teams = picks.groupby(["game_uid", "team"]).champion.apply(list)
    teams = teams[teams.map(len) == 5]

    counts = {c: p.copy() for c, p in prior.items()}
    for g, t, c in known_role:
        if c in counts:
            counts[c][known_role[(g, t, c)]] += 1
    assignment = {}
    for _ in range(EM_ITERATIONS):
        logp = {c: np.log(v / v.sum()) for c, v in counts.items()}
        new_counts = {c: p.copy() for c, p in prior.items()}
        for (g, t), champs in teams.items():
            if all((g, t, c) in known_role for c in champs):
                perm = tuple(known_role[(g, t, c)] for c in champs)
            else:
                perm = _assign(champs, logp)
            assignment[(g, t)] = perm
            for c, r in zip(champs, perm):
                new_counts[c][r] += 1
        counts = new_counts

    pick_rows = [(g, t, c, ROLES[r]) for (g, t), champs in teams.items()
                 for c, r in zip(champs, assignment[(g, t)])]
    pick_roles = pd.DataFrame(pick_rows, columns=["game_uid", "team", "champion", "role"])
    dist = pd.DataFrame({c: (v - prior[c]) for c, v in counts.items()}, index=ROLES).T.clip(lower=0)
    dist = dist.div(dist.sum(axis=1).replace(0, 1), axis=0)
    dist["primary_role"] = dist[ROLES].idxmax(axis=1)
    return pick_roles, dist.rename_axis("champion").reset_index()


def validate_roles(picks: pd.DataFrame, classes: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Hold out every single-game page, infer, and score against the true roles."""
    holdout = set(truth.game_uid)
    pick_roles, _ = infer_roles(picks, classes, truth, holdout=holdout)
    m = truth.merge(pick_roles, on=["game_uid", "team", "champion"], suffixes=("_true", "_inferred"))
    return dict(team_games=int(m.groupby(["game_uid", "team"]).ngroups), picks=len(m),
                accuracy=float((m.role_true == m.role_inferred).mean()),
                per_role={r: float((g.role_true == g.role_inferred).mean()) for r, g in m.groupby("role_true")})


# --- substitutions --------------------------------------------------------------

def with_roles(transitions: pd.DataFrame, champ: pd.DataFrame) -> pd.DataFrame:
    return transitions.assign(role=transitions.champ_key.map(champ.set_index("champ_key").primary_role))


def substitutions(feat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A changed champion moved a lot; which untouched same-role champions moved the other way?"""
    rows = []
    for (fp, tp), g in feat.groupby(["from_patch", "to_patch"], sort=False):
        movers = g[g.changed & (g.delta_presence.abs() >= SUB_MIN_MOVE_A)]
        for a in movers.itertuples():
            cands = g[~g.changed & (g.role == a.role) & (g.champ_key != a.champ_key)
                      & (np.sign(g.delta_presence) == -np.sign(a.delta_presence))
                      & (g.delta_presence.abs() >= SUB_MIN_MOVE_B)]
            for b in cands.nlargest(3, "delta_presence" if a.delta_presence < 0 else "prev_presence").itertuples():
                rows.append(dict(from_patch=fp, to_patch=tp, role=a.role, changed=a.champion, change=a.verdict,
                                 change_patches=a.change_patches, changed_delta=a.delta_presence,
                                 changed_before=a.prev_presence, changed_after=a.next_presence,
                                 other=b.champion, other_delta=b.delta_presence, other_before=b.prev_presence,
                                 other_after=b.next_presence))
    subs = pd.DataFrame(rows)
    if subs.empty:
        return subs, subs
    pairs = subs.groupby(["changed", "other", "role"]).agg(
        times=("to_patch", "size"), patches=("to_patch", lambda s: ", ".join(sorted(set(s), key=patch_key))),
        avg_changed_delta=("changed_delta", "mean"), avg_other_delta=("other_delta", "mean"),
    ).reset_index().sort_values(["times", "avg_other_delta"], ascending=[False, False])
    return subs, pairs


def run(log=print) -> dict:
    games, changes = load_games(), load_changes()
    events = pd.read_csv(PROCESSED / "event_games.csv", dtype={"patch": str, "series_id": str})
    picks_all = long_draft(pd.concat([games, events], ignore_index=True)).query("action == 'pick'")

    classes = champion_classes(log=log)
    truth = _true_roles()
    validation = validate_roles(picks_all, classes, truth)
    log(f"  role inference: {validation['accuracy']:.1%} correct on {validation['picks']} held-out picks "
        f"with known roles ({validation['team_games']} team-games)")
    pick_roles, dist = infer_roles(picks_all, classes, truth)
    exact_path = PROCESSED / "players.csv"
    if exact_path.exists():
        # Exact roles from every game page: score the inference on all of them, then use the exact roles.
        exact = pd.read_csv(exact_path)[["game_uid", "team", "champion", "role"]]
        m = pick_roles.merge(exact, on=["game_uid", "team", "champion"], suffixes=("_inferred", ""))
        validation["all_games"] = dict(picks=len(m), accuracy=float((m.role == m.role_inferred).mean()))
        log(f"  role inference vs exact roles on all games: {validation['all_games']['accuracy']:.1%} "
            f"of {len(m):,} picks")
        pick_roles = pick_roles.merge(exact, on=["game_uid", "team", "champion"], how="left", suffixes=("_inferred", ""))
        pick_roles["role"] = pick_roles.role.fillna(pick_roles.role_inferred)
        pick_roles = pick_roles.drop(columns="role_inferred")
        dist = pd.crosstab(pick_roles.champion, pick_roles.role, normalize="index").reindex(columns=ROLES, fill_value=0)
        dist["primary_role"] = dist[ROLES].idxmax(axis=1)
        dist = dist.reset_index()
    pick_roles.to_csv(PROCESSED / "pick_roles.csv", index=False)
    dist["champ_key"] = dist.champion.map(champ_key)
    champ = classes.merge(dist.drop(columns="champion").drop_duplicates("champ_key"), on="champ_key", how="left")
    champ.to_csv(PROCESSED / "champion_meta.csv", index=False)
    (PROCESSED / "role_validation.json").write_text(json.dumps(validation, indent=2))

    stats = champion_stats(games, "ALL")
    subs, pairs = substitutions(with_roles(build_transitions(stats, changes), champ))
    subs.to_csv(PROCESSED / "substitutions.csv", index=False)
    pairs.to_csv(PROCESSED / "substitute_pairs.csv", index=False)
    log(f"  {len(subs)} substitution events, {int((pairs.times >= 2).sum()) if len(pairs) else 0} recurring pairs")
    return dict(validation=validation)


# --- champion groups: who fell as a group rose ------------------------------------

def group_presence(games: pd.DataFrame, members: list[str]) -> pd.Series:
    """Per domestic patch: share of games where at least one member was picked or banned."""
    d = long_draft(games)
    hit = d[d.champion.isin(members)].groupby("patch").game_uid.nunique()
    total = d.groupby("patch").game_uid.nunique()
    total = total[total >= MIN_GAMES_PER_PATCH]
    out = (hit.reindex(total.index).fillna(0) / total)
    return out[sorted(out.index, key=patch_key)]


def rise_window(series: pd.Series) -> tuple[str, str]:
    """From the group's low point before its peak, to its peak."""
    peak = series.idxmax()
    before = series.loc[:peak]
    return before.idxmin(), peak


def displacement(stats: pd.DataFrame, group: pd.Series, members: list[str], start: str, end: str,
                 champ: pd.DataFrame, min_presence: float = 0.08) -> pd.DataFrame:
    """Every other champion's move across [start, end], and how its patch-to-patch moves track the group's.

    Ends are averaged over two patches to damp single-patch noise. r uses
    patch-to-patch changes, so two unrelated season-long trends don't look linked.
    """
    from scipy.stats import spearmanr
    patches = [p for p in group.index if patch_key(start) <= patch_key(p) <= patch_key(end)]
    if len(patches) < 3:
        return pd.DataFrame()
    head, tail = patches[:2], patches[-2:]
    pres = stats.pivot_table(index="champion", columns="patch", values="presence").reindex(columns=patches).fillna(0)
    pres = pres[~pres.index.isin(members)]
    pres = pres[pres.max(axis=1) >= min_presence]
    g_diff = group.reindex(patches).diff().dropna()
    info = champ.assign(k=champ.champ_key).set_index("k")
    member_roles = {info.primary_role.get(champ_key(m)) for m in members}
    rows = []
    for c, row in pres.iterrows():
        r, p = spearmanr(row.diff().dropna(), g_diff) if len(g_diff) >= 4 else (np.nan, np.nan)
        role = info.primary_role.get(champ_key(c))
        rows.append(dict(champion=c, role=role, same_role=role in member_roles,
                         before=row[head].mean(), after=row[tail].mean(), peak=row.max(),
                         r=r, p_value=p))
    out = pd.DataFrame(rows)
    out["change"] = out.after - out.before
    return out.sort_values("change")
