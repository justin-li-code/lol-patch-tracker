"""What each pro player built: final items and runes, from gol.gg game pages.

The same game pages players.py reads (already cached) show every player's final
items (Data Dragon item ids) and runes (ids in a tooltip). Names come from Data
Dragon so they line up with the patch notes. Writes data/processed/builds.csv:
game_uid, team, role, player, champion, keystone, runes, items ("|"-joined names).
"""
import json
import re

import pandas as pd

from . import http
from .config import PROCESSED
from .players import ROLES, game_ids
from .scrape_matches import BASE, list_series, list_tournaments, parse_game

DDRAGON = "https://ddragon.leagueoflegends.com"
PLAYER_LINK = re.compile(r"players/player-stats/\d+/[^']*' title='([^']+) stats'")
ITEM = re.compile(r"alt='Item_(\d+)")
RUNE_IMG = re.compile(r"_img/perkstyle/(\d+)\.png")
# Consumables, trinkets and wards say nothing about a champion's build.
SKIP_ITEMS = re.compile(r"potion|elixir|ward|lens|farsight|biscuit|control ward|refillable", re.I)


def _names() -> tuple[dict, dict, set[str]]:
    """Per game version ("16.5"): item id -> name and rune id -> name; plus rune-tree ids.

    Ids are matched against the game's own version because Riot reuses them: 8230 was Phase
    Rush until 26.9 and is Stormraider's Surge after.
    """
    versions = json.loads(http.get(f"{DDRAGON}/api/versions.json"))
    season = [v for v in versions if v.startswith("16.")]
    items, runes, trees = {}, {}, set()
    for v in season:
        key = ".".join(v.split(".")[:2])
        if key in items:
            continue  # keep the last hotfix of each patch (versions are newest first)
        data = json.loads(http.get(f"{DDRAGON}/cdn/{v}/data/en_US/item.json"))["data"]
        items[key] = {k: d["name"] for k, d in data.items()}
        runes[key] = {}
        for tree in json.loads(http.get(f"{DDRAGON}/cdn/{v}/data/en_US/runesReforged.json")):
            trees.add(str(tree["id"]))
            runes[key][str(tree["id"])] = tree["name"]
            for slot in tree["slots"]:
                runes[key] |= {str(r["id"]): r["name"] for r in slot["runes"]}
    return items, runes, trees


def parse_builds(html: str, items: dict, runes: dict, trees: set) -> list[dict]:
    """Ten rows in role order: final items and runes per player."""
    game = parse_game(html)
    links = list(PLAYER_LINK.finditer(html))
    if not game or len(links) != 10:
        return []
    newest = max(items, key=lambda v: tuple(map(int, v.split("."))))
    version = game[0].get("version") or newest
    item_names, rune_names = items.get(version, items[newest]), runes.get(version, runes[newest])
    out = []
    for k, m in enumerate(links):
        seg = html[m.end(): links[k + 1].start() if k + 1 < 10 else len(html)]
        tab = seg.find("perkztab")
        # Item icons sit before the rune tooltip's detail table; rune ids inside it.
        built = [item_names.get(i, "") for i in ITEM.findall(seg)]
        built = [b for b in built if b and not SKIP_ITEMS.search(b)]
        rune_ids = RUNE_IMG.findall(seg[tab: seg.find("</table>", tab)]) if tab >= 0 else []
        chosen = [rune_names[r] for r in rune_ids if r in rune_names and r not in trees]
        side = game[0]["sides"][0 if k < 5 else 1]
        out.append(dict(team=side["team"], role=ROLES[k % 5], player=m.group(1),
                        keystone=chosen[0] if chosen else "", runes="|".join(chosen), items="|".join(built)))
    return out


def build(incremental: bool = True, events_only: bool = False, log=print) -> pd.DataFrame:
    """Every game players.csv knows about (pages are cached, so this downloads almost nothing).
    events_only skips league tournaments: what the scheduled refresh uses on a cold cache."""
    from .events import EVENTS
    path = PROCESSED / "builds.csv"
    existing = pd.read_csv(path) if incremental and path.exists() else pd.DataFrame()
    known = set(existing.game_uid) if len(existing) else set()
    champs = pd.read_csv(PROCESSED / "players.csv").set_index(["game_uid", "team", "role"]).champion
    items, runes, trees = _names()
    event_tournaments = [t for ev in EVENTS.values() for t in ev["tournaments"]]
    tournaments = event_tournaments if events_only else list_tournaments() + event_tournaments
    rows = []
    for t in tournaments:
        for s in list_series(t):
            for game_no, gid in game_ids(s["series_id"], s["page"]):
                uid = f"{s['series_id']}-{game_no}"
                if uid in known:
                    continue
                try:
                    html = http.get(f"{BASE}/game/stats/{gid}/page-game/") or ""
                except RuntimeError as e:
                    log(f"    skipped: {e}")
                    continue
                for r in parse_builds(html, items, runes, trees):
                    rows.append(dict(game_uid=uid, champion=champs.get((uid, r["team"], r["role"])), **r))
    new = pd.DataFrame(rows)
    df = pd.concat([existing, new], ignore_index=True) if len(existing) else new
    df.to_csv(path, index=False)
    log(f"Saved builds for {df.game_uid.nunique()} games ({new.game_uid.nunique() if len(new) else 0} new) to {path}")
    return df
