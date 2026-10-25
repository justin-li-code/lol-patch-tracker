"""Champion portraits (Riot Data Dragon) and team logos (gol.gg) for the dashboard.

Outputs (data/assets/):
    champions/*.png      Data Dragon square portraits, downloaded once
    champion_icons.json  champ_key -> portrait file name
    logos/*.webp         team logos, downloaded once
    team_logos.json      team name -> logo file name
"""
import json
import re
import time

import pandas as pd
import requests

from . import http
from .champions import champ_key
from .config import DATA, PROCESSED, REQUEST_DELAY, USER_AGENT

ASSETS = DATA / "assets"
LOGOS = ASSETS / "logos"
CHAMPIONS = ASSETS / "champions"
DDRAGON = "https://ddragon.leagueoflegends.com"
GOL = "https://gol.gg"


def champion_icons(log=print) -> dict[str, str]:
    """Download square portraits to data/assets/champions/<id>.png; map champ_key -> file name."""
    version = requests.get(f"{DDRAGON}/api/versions.json", timeout=30).json()[0]
    data = requests.get(f"{DDRAGON}/cdn/{version}/data/en_US/champion.json", timeout=30).json()["data"]
    CHAMPIONS.mkdir(parents=True, exist_ok=True)
    icons = {}
    for c in data.values():
        fname = c["image"]["full"]
        path = CHAMPIONS / fname
        if not path.exists():
            resp = requests.get(f"{DDRAGON}/cdn/{version}/img/champion/{fname}", timeout=30)
            resp.raise_for_status()
            path.write_bytes(resp.content)
        # Map both display name ("Kai'Sa") and internal id ("Kaisa", "MonkeyKing").
        icons[champ_key(c["name"])] = fname
        icons.setdefault(champ_key(c["id"]), fname)
    (ASSETS / "champion_icons.json").write_text(json.dumps(icons, indent=1))
    log(f"  {len(data)} champion portraits (Data Dragon {version})")
    return icons


def _team_ids(event_games: pd.DataFrame) -> dict[str, str]:
    """Team name -> gol.gg team id, read from the (cached) series pages."""
    ids = {}
    for sid, page in event_games[["series_id", "page"]].drop_duplicates().itertuples(index=False):
        html = http.get(f"{GOL}/game/stats/{sid}/page-{page}/") or ""
        for tid, name in re.findall(r"teams/team-stats/(\d+)/[^']*' title='([^']+) stats'", html):
            ids.setdefault(name, tid)
    return ids


def team_logos(log=print) -> dict[str, str]:
    """Download logos for every team in the domestic and event data."""
    teams = set(pd.read_csv(PROCESSED / "event_games.csv").team) | set(pd.read_csv(PROCESSED / "games.csv").team)
    ids = _team_ids(_series_pages())
    manifest_path = ASSETS / "team_logos.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    LOGOS.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    for team in sorted(teams):
        if team in manifest and (LOGOS / manifest[team]).exists():
            continue
        tid = ids.get(team)
        if not tid:
            log(f"  no gol.gg id for {team}")
            continue
        html = http.get(f"{GOL}/teams/team-stats/{tid}/split-ALL/tournament-ALL/") or ""
        m = re.search(r"_img/teams_icon/([^'\"]+)", html)
        if not m:
            log(f"  no logo for {team}")
            continue
        resp = None
        for attempt in range(4):
            time.sleep(REQUEST_DELAY * (attempt + 1))
            try:
                resp = session.get(f"{GOL}/_img/teams_icon/{m.group(1)}", timeout=30)
                break
            except requests.RequestException:
                resp = None
        if resp is None or resp.status_code != 200 or not resp.headers.get("content-type", "").startswith("image"):
            log(f"  logo download failed for {team} (re-run to retry)")
            continue
        fname = re.sub(r"[^A-Za-z0-9_.-]", "_", m.group(1))
        (LOGOS / fname).write_bytes(resp.content)
        manifest[team] = fname
        manifest_path.write_text(json.dumps(manifest, indent=1))
    manifest_path.write_text(json.dumps(manifest, indent=1))
    log(f"  {len(manifest)} team logos")
    return manifest


def _series_pages() -> pd.DataFrame:
    """Which gol.gg page each series was scraped from (summary for multi-game series, game for Bo1)."""
    from .events import EVENTS
    from .scrape_matches import list_series, list_tournaments
    tournaments = list_tournaments() + [t for ev in EVENTS.values() for t in ev["tournaments"]]
    rows = [(s["series_id"], s["page"]) for t in tournaments for s in list_series(t)]
    return pd.DataFrame(rows, columns=["series_id", "page"])


def run(log=print):
    champion_icons(log=log)
    team_logos(log=log)
