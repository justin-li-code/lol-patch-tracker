"""Draft answers: which picks do teams make *in response* to an opponent's pick?

Draft order: gol.gg groups each team's picks by consecutive draft turns, so the
first-pick team's picks read 1 | 2 2 (global picks 1 | 4,5 | 8,9) and the other
team's 2 2 | 1 (2,3 | 6,7 | 10). That gives every pick's global position.

Two kinds of answer:
  * Lane answers: one team picks X for a role; the opponent fills the same role
    later with Y ("Y into X"). Roles come from gol.gg game pages (exact) where
    available, otherwise from meta.py's inferred roles.
  * First-pick answers: the global first pick X, and what the other team takes
    with picks 2-3.

A pair is an answer pattern when Y follows X much more often than Y is picked
in that spot in general (lift), with a binomial test corrected for testing
many pairs (Benjamini-Hochberg). Each pattern is then tagged by where it comes
from: Worlds, one region, one team or player, or the broad meta.

Outputs (data/processed/): draft_lane_events.csv, lane_answers.csv,
first_pick_answers.csv, answer_breakdown.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

from .analyze import load_games
from .config import PROCESSED
from .regions import long_draft, team_regions

FIRST_PICK_POS = [1, 4, 5, 8, 9]
SECOND_PICK_POS = [2, 3, 6, 7, 10]
MIN_OPENER, MIN_PAIR, MIN_LIFT, MAX_Q = 10, 4, 1.5, 0.10
EVENT_NAMES = {"first_stand": "First Stand", "msi": "MSI", "ewc": "EWC", "demacia": "Demacia Cup", "worlds": "Worlds"}


def load_all_games() -> pd.DataFrame:
    dom = load_games()
    ev = pd.read_csv(PROCESSED / "event_games.csv", dtype={"patch": str, "version": str, "series_id": str})
    games = pd.concat([dom.assign(scope=dom.league), ev.assign(scope=ev.league.map(EVENT_NAMES))], ignore_index=True)
    return games


def ordered_picks(games: pd.DataFrame) -> pd.DataFrame:
    """One row per pick with its global draft position (games with known order only)."""
    d = long_draft(games).query("action == 'pick'")
    meta_cols = games[["game_uid", "team", "first_pick", "scope", "tournament"]].drop_duplicates(["game_uid", "team"])
    d = d.merge(meta_cols, on=["game_uid", "team"])
    ok = d.groupby("game_uid").first_pick.transform(lambda s: sorted(s.dropna().unique().tolist()) == [0, 1])
    d = d[ok & d.first_pick.notna()].copy()
    d["pos"] = [FIRST_PICK_POS[s - 1] if fp == 1 else SECOND_PICK_POS[s - 1] for s, fp in zip(d.slot, d.first_pick)]
    return d


def attach_roles(picks: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Exact roles from game pages where we have them, inferred roles otherwise."""
    inferred = pd.read_csv(PROCESSED / "pick_roles.csv")
    out = picks.merge(inferred, on=["game_uid", "team", "champion"], how="left")
    source = "inferred"
    players_path = PROCESSED / "players.csv"
    if players_path.exists():
        exact = pd.read_csv(players_path)[["game_uid", "team", "champion", "role", "player"]]
        out = out.merge(exact, on=["game_uid", "team", "champion"], how="left", suffixes=("_inferred", ""))
        n_exact = out.role.notna().mean()
        out["role"] = out.role.fillna(out.role_inferred)
        out = out.drop(columns="role_inferred")
        source = f"exact for {n_exact:.0%} of picks, inferred for the rest"
    else:
        out["player"] = np.nan
    return out, source


def lane_events(picks: pd.DataFrame) -> pd.DataFrame:
    """For each game and role: who opened the lane, and who answered."""
    p = picks.dropna(subset=["role"])
    rows = []
    for (g, role), lane in p.groupby(["game_uid", "role"]):
        if len(lane) != 2 or lane.team.nunique() != 2:
            continue
        a, b = lane.sort_values("pos").itertuples()
        rows.append(dict(game_uid=g, role=role, scope=a.scope, tournament=a.tournament, patch=a.patch,
                         opener=a.champion, opener_team=a.team, opener_pos=a.pos, opener_player=a.player,
                         answer=b.champion, answer_team=b.team, answer_pos=b.pos, answer_player=b.player,
                         answer_won=b.result, counterpick=b.pos == 10))
    return pd.DataFrame(rows)


def _score(pairs: pd.DataFrame) -> pd.DataFrame:
    """pairs needs n, opener_n and expected (patch-matched expected count of Y after X)."""
    pairs = pairs.copy()
    pairs["answer_rate"] = pairs.n / pairs.opener_n
    pairs["base_rate"] = pairs.expected / pairs.opener_n
    pairs["lift"] = pairs.n / pairs.expected
    pairs["p_value"] = [binomtest(int(n), int(m), min(max(b, 1e-6), 1 - 1e-6), alternative="greater").pvalue
                        for n, m, b in zip(pairs.n, pairs.opener_n, pairs.base_rate)]
    pairs["q_value"] = multipletests(pairs.p_value, method="fdr_bh")[1] if len(pairs) else []
    return pairs


def _expected(openers: pd.DataFrame, base: pd.Series, by: list[str]) -> pd.DataFrame:
    """Expected count of each answer after each opener, using the answer's usual rate *on the same patch*.

    Without patch matching, two champions that were simply popular at the same
    time would look like a response pattern.
    """
    counts = openers.groupby(by + ["opener", "patch"]).size().rename("k").reset_index()
    rates = base.rename("rate").reset_index()
    m = counts.merge(rates, on=by + ["patch"])
    m["expected"] = m.k * m.rate
    return m.groupby(by + ["opener", "answer"]).expected.sum().reset_index()


def lane_answers(ev: pd.DataFrame) -> pd.DataFrame:
    opener_n = ev.groupby(["role", "opener"]).size().rename("opener_n")
    pairs = ev.groupby(["role", "opener", "answer"]).agg(
        n=("game_uid", "size"), answer_wins=("answer_won", "sum"), counterpicks=("counterpick", "sum"),
    ).reset_index().join(opener_n, on=["role", "opener"])
    pairs = pairs[pairs.opener_n >= MIN_OPENER]
    # Usual rate of Y as the later pick in this role, on this patch.
    base = ev.groupby(["role", "patch", "answer"]).size() / ev.groupby(["role", "patch"]).size()
    exp = _expected(ev[ev.opener.isin(pairs.opener)], base, ["role"])
    pairs = pairs.merge(exp, on=["role", "opener", "answer"], how="left")
    pairs = _score(pairs[pairs.expected > 0])
    usual_wr = ev.groupby(["role", "answer"]).answer_won.mean()
    pairs["answer_win_rate"] = pairs.answer_wins / pairs.n
    pairs["answer_usual_win_rate"] = [usual_wr.get((r, a), np.nan) for r, a in zip(pairs.role, pairs.answer)]
    return pairs.sort_values(["q_value", "lift"], ascending=[True, False])


def first_pick_answers(picks: pd.DataFrame) -> pd.DataFrame:
    """Global pick 1 -> the other team's picks 2-3."""
    first = picks[picks.pos == 1][["game_uid", "champion", "patch"]].rename(columns={"champion": "opener"})
    resp = picks[picks.pos.isin([2, 3])][["game_uid", "champion", "result", "patch"]].rename(columns={"champion": "answer"})
    ev = first.merge(resp.drop(columns="patch"), on="game_uid")
    opener_n = first.groupby("opener").size().rename("opener_n")
    pairs = ev.groupby(["opener", "answer"]).agg(n=("game_uid", "size"), answer_wins=("result", "sum")) \
        .reset_index().join(opener_n, on="opener")
    pairs = pairs[pairs.opener_n >= MIN_OPENER]
    # Usual share of games (same patch) in which Y is taken with picks 2-3.
    base = resp.groupby(["patch", "answer"]).size() / first.groupby("patch").size()
    exp = _expected(first[first.opener.isin(pairs.opener)], base, [])
    pairs = pairs.merge(exp, on=["opener", "answer"], how="left")
    pairs = _score(pairs[pairs.expected > 0])
    return pairs.sort_values(["q_value", "lift"], ascending=[True, False])


def breakdown(ev: pd.DataFrame, pairs: pd.DataFrame, regions: dict) -> pd.DataFrame:
    """For each solid lane pattern: answer rate by scope, region, team and player, and a 'where from' tag."""
    ev = ev.assign(answer_region=ev.answer_team.map(regions).fillna("Other"))
    rows, tags = [], {}
    for pr in pairs.itertuples():
        lane = ev[(ev.role == pr.role) & (ev.opener == pr.opener)]
        hit = lane.answer == pr.answer
        found = []
        for dim, col in (("event or league", "scope"), ("region of answering team", "answer_region"),
                         ("answering team", "answer_team"), ("answering player", "answer_player")):
            if lane[col].isna().all():
                continue
            g = lane.groupby(col).agg(openers=("answer", "size"), answers=("answer", lambda s: (s == pr.answer).sum()))
            for name, r in g.iterrows():
                if r.answers == 0:
                    continue
                rest_open = len(lane) - r.openers
                rest_rate = (hit.sum() - r.answers) / rest_open if rest_open else np.nan
                rate = r.answers / r.openers
                rows.append(dict(role=pr.role, opener=pr.opener, answer=pr.answer, dimension=dim, group=name,
                                 openers=int(r.openers), answers=int(r.answers), rate=rate, rest_rate=rest_rate))
                share = r.answers / hit.sum()
                concentrated = r.answers >= 3 and (pd.isna(rest_rate) or rate >= 2 * max(rest_rate, 0.01))
                if dim == "event or league" and name == "Worlds" and concentrated:
                    found.append("Worlds")
                elif dim == "region of answering team" and concentrated and share >= 0.6:
                    found.append(f"{name} (region)")
                elif dim in ("answering team", "answering player") and share >= 0.5 and r.answers >= 3:
                    found.append(f"{name} ({dim.split()[-1]})")
        tags[(pr.role, pr.opener, pr.answer)] = ", ".join(dict.fromkeys(found)) or "Broad meta"
    return pd.DataFrame(rows), tags


def run(log=print) -> dict:
    games = load_all_games()
    picks, role_source = attach_roles(ordered_picks(games))
    ev = lane_events(picks)
    ev.to_csv(PROCESSED / "draft_lane_events.csv", index=False)
    lanes = lane_answers(ev)
    solid = lanes[(lanes.n >= MIN_PAIR) & (lanes.lift >= MIN_LIFT) & (lanes.q_value <= MAX_Q)]
    regions = team_regions(load_games())
    bd, tags = breakdown(ev, solid, regions)
    lanes["where_from"] = [tags.get(k, "") for k in zip(lanes.role, lanes.opener, lanes.answer)]
    lanes["solid"] = lanes.index.isin(solid.index)
    lanes.to_csv(PROCESSED / "lane_answers.csv", index=False)
    bd.to_csv(PROCESSED / "answer_breakdown.csv", index=False)
    fp = first_pick_answers(picks)
    fp["solid"] = (fp.n >= MIN_PAIR) & (fp.lift >= MIN_LIFT) & (fp.q_value <= MAX_Q)
    fp.to_csv(PROCESSED / "first_pick_answers.csv", index=False)
    meta = dict(games=int(picks.game_uid.nunique()), lanes=len(ev), role_source=role_source,
                lane_pairs_tested=len(lanes), lane_solid=int(lanes.solid.sum()),
                fp_pairs_tested=len(fp), fp_solid=int(fp.solid.sum()))
    import json
    (PROCESSED / "draft_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"  {meta['games']} games with known draft order, {meta['lanes']} lane match-ups (roles: {role_source})")
    log(f"  lane answers: {meta['lane_solid']} solid patterns of {meta['lane_pairs_tested']} pairs tested; "
        f"first-pick answers: {meta['fp_solid']} of {meta['fp_pairs_tested']}")
    return meta
