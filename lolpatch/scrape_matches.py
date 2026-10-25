"""Scrape pro games (picks, bans, winner, patch) from gol.gg.

Output: data/processed/games.csv, one row per team per game:
    game_uid, series_id, league, tournament, date, version, patch, team,
    result (1 win / 0 loss), picks (|-joined), bans (|-joined)
"""
import re
from urllib.parse import quote

import pandas as pd

from . import http
from .config import GOL_SEASON, LEAGUES, PROCESSED, YEAR, version_to_brand

BASE = "https://gol.gg"
_ALT = re.compile(r"alt='([^']+)'")
_CHAMP_LINK = re.compile(r"champion-stats/\d+/[^']*' title='([^']+) stats'")


def list_tournaments(refresh: bool = False) -> list[str]:
    """Tournament names for the configured leagues/year, as gol.gg names them."""
    html = http.get(f"{BASE}/champion/list/season-{GOL_SEASON}/split-ALL/tournament-ALL/", refresh=refresh)
    select = html[html.find("id='cbtournament'"):]
    select = select[: select.find("</select>")]
    names = re.findall(r"<option[^>]*value=['\"]([^'\"]+)['\"]", select)
    league_re = re.compile(rf"^({'|'.join(LEAGUES)})( Cup)? {YEAR}\b|^({'|'.join(LEAGUES)}) Cup {YEAR}$")
    out = []
    for name in names:
        if not league_re.match(name):
            continue
        # "LCK CL" is the academy league, not the LCK.
        if name.split()[1] in ("CL",):
            continue
        out.append(name)
    return out


def league_of(tournament: str) -> str:
    return tournament.split()[0]


def list_series(tournament: str, refresh: bool = False) -> list[dict]:
    url = f"{BASE}/tournament/tournament-matchlist/{quote(tournament)}/"
    html = http.get(url, refresh=refresh) or ""
    row_re = re.compile(
        r"game/stats/(\d+)/page-([a-z]+)/' title='[^']*'>[^<]*</a></td>"
        r"<td[^>]*>([^<]*)</td><td[^>]*>([^<]*)</td><td[^>]*>([^<]*)</td>"
        r"<td[^>]*>([^<]*)</td><td[^>]*>([^<]*)</td><td[^>]*>([^<]*)</td>"
    )
    series = []
    for sid, page, _t1, score, _t2, stage, version, date in row_re.findall(html):
        # Unplayed/forfeited series have no score.
        if not re.match(r"\d+ - \d+", score.strip()):
            continue
        series.append(dict(series_id=sid, page=page, stage=stage.strip(),
                           version=version.strip(), date=date.strip()))
    return series


def _names(fragment: str) -> list[str]:
    return _ALT.findall(fragment)


def _first_pick(picks_fragment: str, pattern=_ALT) -> int | None:
    """Which team picked first, from how gol.gg groups a team's picks with "|".

    Picks are grouped by consecutive turns in the draft: the first-pick team
    picks 1 | 4,5 | 8,9 (groups 1-2-2), the other team 2,3 | 6,7 | 10 (2-2-1).
    Returns 1, 0, or None when the page shows no grouping.
    """
    groups = [len(pattern.findall(g)) for g in picks_fragment.split("|")]
    return {(1, 2, 2): 1, (2, 2, 1): 0}.get(tuple(groups))


def parse_summary(html: str) -> list[dict]:
    """Series summary page -> list of games, each with two team dicts."""
    team_names = re.findall(r"<div class=\"col-4 col-sm-5 text-center\"><h1><a [^>]*title='([^']+) stats'", html)
    blocks = re.split(r"<h1 class='text_(victory|defeat)'>", html)
    sides = []
    for k in range(1, len(blocks), 2):
        result, body = blocks[k], blocks[k + 1]
        body = body[: body.find("</div>")]
        bans, _, picks = body.partition("Picks")
        sides.append(dict(result=int(result == "victory"), bans=_names(bans), picks=_names(picks),
                          first_pick=_first_pick(picks)))
    games = []
    for g in range(0, len(sides) - 1, 2):
        left, right = sides[g], sides[g + 1]
        if len(team_names) >= 2:
            left["team"], right["team"] = team_names[0], team_names[1]
        games.append(dict(game_no=g // 2 + 1, sides=[left, right], version=None))
    return games


def parse_game(html: str) -> list[dict]:
    """Single game page -> one game with blue/red team dicts."""
    m = re.search(r"v(\d+\.\d+)\s*</div>", html)
    version = m.group(1) if m else None
    sides = []
    for side in ("blue", "red"):
        i = html.find(f"{side}-line-header")
        if i < 0:
            return []
        header = html[i: html.find("</div>", i)]
        team = re.search(r"title='([^']+) stats'", header)
        result = 1 if "WIN" in header else 0
        chunk = html[i: html.find("red-line-header") if side == "blue" else len(html)]
        b = chunk.find(">Bans<")
        p = chunk.find(">Picks")
        if b < 0 or p < 0:
            return []
        picks_end = chunk.find("</div>", chunk.find("col-10", p))
        sides.append(dict(team=team.group(1) if team else side, side=side, result=result,
                          bans=_CHAMP_LINK.findall(chunk[b:p]),
                          picks=_CHAMP_LINK.findall(chunk[p:])[:5],
                          first_pick=_first_pick(chunk[p:picks_end], _CHAMP_LINK)))
    return [dict(game_no=1, sides=sides, version=version)]


def scrape_tournaments(tournaments: list[str], league_fn=league_of, refresh: bool = False,
                       refetch_recent_days: int = 0, log=print) -> pd.DataFrame:
    """Scrape every completed series of the given tournaments.

    refetch_recent_days re-downloads series pages played in the last N days of
    the tournament, so a page cached mid-series gets completed on the next run.
    """
    rows = []
    for t in tournaments:
        series = list_series(t, refresh=refresh)
        log(f"  {t}: {len(series)} series")
        latest = max((pd.Timestamp(s["date"]) for s in series), default=None)
        for s in series:
            recent = refetch_recent_days and latest is not None and \
                (latest - pd.Timestamp(s["date"])).days < refetch_recent_days
            try:
                html = http.get(f"{BASE}/game/stats/{s['series_id']}/page-{s['page']}/", refresh=bool(recent))
            except RuntimeError as e:
                log(f"    skipped: {e}")  # re-run later; cached pages are not re-fetched
                continue
            if not html:
                continue
            games = parse_summary(html) if s["page"] == "summary" else parse_game(html)
            for g in games:
                version = g["version"] or s["version"]
                for side in g["sides"]:
                    if len(side["picks"]) != 5:
                        continue  # incomplete data on gol.gg
                    rows.append(dict(
                        game_uid=f"{s['series_id']}-{g['game_no']}",
                        series_id=s["series_id"], league=league_fn(t), tournament=t,
                        stage=s["stage"], date=s["date"], version=version,
                        patch=version_to_brand(version), team=side.get("team", ""),
                        result=side["result"], picks="|".join(side["picks"]),
                        bans="|".join(side["bans"]), first_pick=side.get("first_pick"),
                    ))
    df = pd.DataFrame(rows)
    # A game needs both sides to count.
    return df[df.groupby("game_uid")["team"].transform("size") == 2]


def scrape(refresh: bool = False, log=print) -> pd.DataFrame:
    """Domestic league games -> data/processed/games.csv (the training data)."""
    tournaments = list_tournaments(refresh=refresh)
    log(f"{len(tournaments)} tournaments: {', '.join(tournaments)}")
    df = scrape_tournaments(tournaments, refresh=refresh, log=log)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED / "games.csv", index=False)
    log(f"Saved {df.game_uid.nunique()} games to {PROCESSED / 'games.csv'}")
    return df
