"""Player ratings for earlier seasons (2024, 2025), for the site's Players page.

Same game pages and rating method as 2026 (matches.py, ratings.py), each season rated within
its own year: 3.0 is a typical pro game that year. Expected results use the team-strength
settings tuned on 2026 (expected_eval.json), fit day by day on that season's games only.

Leagues follow what each year had: in 2025 the LCS and CBLOL were merged into the LTA (North
and South conferences plus cross-conference playoffs), so 2025 has one "LTA" board.

Outputs: data/history/<year>/ratings.csv (the site reads this), plus match_*.csv and
player_ratings.csv working files that stay local.
"""
import json
import re

import pandas as pd

from . import http
from .config import DATA, PROCESSED
from .scrape_matches import BASE

HISTORY = DATA / "history"
SEASONS = {
    2025: dict(code="S15", leagues=("LCK", "LPL", "LEC", "LTA"), events={
        "first_stand": ("First Stand 2025", ["First Stand 2025"]),
        "msi": ("MSI 2025", ["2025 Mid-Season Invitational"]),
        "ewc": ("Esports World Cup 2025", ["Esports World Cup 2025"]),
        "worlds": ("Worlds 2025", ["Worlds 2025 Play-In", "Worlds 2025 Main Event"])}),
    2024: dict(code="S14", leagues=("LCK", "LPL", "LEC", "LCS", "CBLOL"), events={
        "msi": ("MSI 2024", ["MSI 2024"]),
        "ewc": ("Esports World Cup 2024", ["Esports World Cup 2024"]),
        "worlds": ("Worlds 2024", ["Worlds Play-In 2024", "Worlds Main Event 2024"])}),
}
# What the Players page needs; the full player_ratings.csv is ~16 MB a season.
KEEP = ["game_uid", "event", "league", "date", "team", "opponent", "role", "player", "champion", "result",
        "rating", "expected_rating", "Kills", "Deaths", "Assists"]


def league_tournaments(year: int) -> dict[str, list[str]]:
    """Each league's tournaments that season (regular splits, playoffs, finals), oldest first."""
    cfg = SEASONS[year]
    html = http.get(f"{BASE}/champion/list/season-{cfg['code']}/split-ALL/tournament-ALL/")
    select = html[html.find("id='cbtournament'"):]
    names = re.findall(r"<option[^>]*value=['\"]([^'\"]+)['\"]", select[: select.find("</select>")])
    out = {}
    for name in reversed(names):  # gol.gg lists newest first
        words = name.split()
        # Academy leagues (LCK CL, CBLOL Academy) and LPLOL aren't the main league.
        if words[0] in cfg["leagues"] and str(year) in name and "Academy" not in name and words[1] != "CL":
            out.setdefault(words[0], []).append(name)
    return out


def scrape(year: int, log=print) -> dict:
    """Download (or reuse cached) game pages for every league and international event that season."""
    from .matches import build
    out = HISTORY / str(year)
    out.mkdir(parents=True, exist_ok=True)
    sources = [(key, None, title, ts) for key, (title, ts) in SEASONS[year]["events"].items()]
    sources += [(None, lg, f"{lg} {year}", ts) for lg, ts in league_tournaments(year).items()]
    return build(incremental=True, log=log, sources=sources, out_dir=out)


def rate(year: int, log=print) -> pd.DataFrame:
    """Expected results and player ratings for one season, from its match_*.csv files."""
    from .ratings import player_ratings, predict_events
    d = HISTORY / str(year)
    games = pd.read_csv(d / "match_games.csv", dtype={"event": str, "league": str, "series_id": str})
    dom = games[games.event.isna()]
    # A team's region is the league it played most of its games in; event-only teams are "Other".
    played = pd.concat([dom[["blue", "league"]].rename(columns={"blue": "team"}),
                        dom[["red", "league"]].rename(columns={"red": "team"})])
    regions = played.groupby("team").league.agg(lambda s: s.mode().iloc[0]).to_dict()
    lead = pd.read_csv(d / "match_gold.csv").groupby("game_uid").blue_lead.mean()
    pairs = pd.DataFrame({
        "game_uid": games.game_uid, "date": pd.to_datetime(games.date), "event": games.event,
        "team_a": games.blue, "team_b": games.red,
        "region_a": games.blue.map(regions).fillna("Other"), "region_b": games.red.map(regions).fillna("Other"),
        "fp_a": 0.0, "y": (games.winner == games.blue).astype(int), "margin": games.game_uid.map(lead),
    })
    s = json.loads((PROCESSED / "expected_eval.json").read_text())["settings"]
    pairs["p_a"] = predict_events(pairs, s["C"], s["region_scale"], use_draft=False, all_days=True,
                                  margin_weight=s["margin_weight"]).fillna(0.5)
    expected = pd.concat([
        pd.DataFrame({"game_uid": pairs.game_uid, "team": pairs.team_a, "opponent": pairs.team_b, "p_win": pairs.p_a}),
        pd.DataFrame({"game_uid": pairs.game_uid, "team": pairs.team_b, "opponent": pairs.team_a,
                      "p_win": 1 - pairs.p_a}),
    ], ignore_index=True)
    # Cross-region results feed each region's strength in the expected player rating.
    cross = pd.concat([
        pd.DataFrame({"game_uid": pairs.game_uid, "date": pairs.date, "team": pairs.team_a,
                      "region": pairs.region_a, "result": pairs.y}),
        pd.DataFrame({"game_uid": pairs.game_uid, "date": pairs.date, "team": pairs.team_b,
                      "region": pairs.region_b, "result": 1 - pairs.y}),
    ], ignore_index=True)
    rated = player_ratings(expected, log=log, src=d, out=d, regions=regions, cross=cross)
    slim = rated[KEEP + [c for c in rated.columns if c.startswith("z_")]]
    slim.to_csv(d / "ratings.csv", index=False)
    log(f"  {year}: saved ratings for {slim.game_uid.nunique()} games to {d / 'ratings.csv'}")
    return slim


def run(years=tuple(SEASONS), log=print):
    for year in years:
        log(f"{year}:")
        scrape(year, log=log)
        rate(year, log=log)
