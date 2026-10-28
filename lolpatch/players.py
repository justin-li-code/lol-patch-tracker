"""Per-game player, role and champion data from gol.gg game pages.

Series summary pages (used for picks/bans) don't name players. Each game's own
page lists both teams' players in role order (top, jungle, mid, bot, support)
with their champion, so this fetches every game page once (cached) and writes
data/processed/players.csv: game_uid, team, role, player, champion.
"""
import re

import pandas as pd

from . import http
from .config import PROCESSED
from .scrape_matches import BASE, list_series, list_tournaments, parse_game

ROLES = ["top", "jungle", "mid", "bot", "support"]


def game_ids(series_id: str, page: str) -> list[tuple[int, str]]:
    """(game_no, gol.gg game id) for every game in a series."""
    if page == "game":
        return [(1, series_id)]
    html = http.get(f"{BASE}/game/stats/{series_id}/page-summary/") or ""
    tabs = re.findall(r"game/stats/(\d+)/page-game/' title='[^']*'>Game (\d+)<", html)
    return sorted({(int(n), gid) for gid, n in tabs})


def parse_players(html: str) -> list[dict]:
    """Ten rows: both teams' players in role order, each with the champion shown beside them."""
    game = parse_game(html)
    if not game:
        return []
    players = list(re.finditer(r"players/player-stats/\d+/[^']*' title='([^']+) stats'", html))
    rows = []
    for m in players:
        alts = re.findall(r"alt='([^']+)'", html[max(0, m.start() - 1500):m.start()])
        rows.append((m.group(1), alts[-1] if alts else None))
    if len(rows) != 10:
        return []
    out = []
    for side, chunk in zip(game[0]["sides"], (rows[:5], rows[5:])):
        for role, (player, champ) in zip(ROLES, chunk):
            out.append(dict(team=side["team"], role=role, player=player, champion=champ))
    return out


def scrape(events_only: bool = False, incremental: bool = False, log=print) -> pd.DataFrame:
    """Every game page (or only international events'), optionally keeping games already in players.csv."""
    from .events import EVENTS
    event_tournaments = [t for ev in EVENTS.values() for t in ev["tournaments"]]
    tournaments = event_tournaments if events_only else list_tournaments() + event_tournaments
    path = PROCESSED / "players.csv"
    existing = pd.read_csv(path) if incremental and path.exists() else pd.DataFrame()
    known = set(existing.game_uid) if len(existing) else set()
    rows, n_series = [], 0
    for t in tournaments:
        for s in list_series(t):
            n_series += 1
            for game_no, gid in game_ids(s["series_id"], s["page"]):
                if f"{s['series_id']}-{game_no}" in known:
                    continue
                try:
                    html = http.get(f"{BASE}/game/stats/{gid}/page-game/")
                except RuntimeError as e:
                    log(f"    skipped: {e}")
                    continue
                for r in parse_players(html or ""):
                    rows.append(dict(game_uid=f"{s['series_id']}-{game_no}", tournament=t, **r))
        log(f"  {t}: done ({len(rows) // 10} games so far)")
    df = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True) if len(existing) else pd.DataFrame(rows)
    df.to_csv(path, index=False)
    log(f"Saved {df.game_uid.nunique()} games with players ({len(rows) // 10} new) to {path}")
    return df
