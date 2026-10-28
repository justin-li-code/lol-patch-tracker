"""Per-game detail for every 2026 game: teams, objectives, timeline, gold lead and player stats.

Sources (gol.gg, cached):
  page-game       teams, result, duration, kills/towers/dragons/barons/gold, objective timeline,
                  gold-lead graph, plates and voidgrubs
  page-fullstats  ~60 stats per player (KDA, CS, gold%, damage%, vision, GD@15, ...)

Outputs (data/processed/): match_games.csv (one row per game), match_teams.csv (one row per
team per game), match_players.csv, match_timeline.csv, match_gold.csv
"""
import re

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from . import http
from .config import PROCESSED
from .players import game_ids, parse_players
from .scrape_matches import BASE, list_series

ALT_IMG = re.compile(r"<img[^>]*alt='([^']*)'[^>]*>")


def _num(text: str) -> float:
    text = (text or "").strip().replace(",", "")
    if not text or text == "-":
        return np.nan
    mult = 1000 if text.endswith("k") else 1
    text = text.rstrip("k%")
    try:
        return float(text) * mult
    except ValueError:
        return np.nan


def parse_game_page(html: str) -> dict | None:
    """Team-level stats, objective timeline, gold lead by minute, plates and grubs."""
    m = re.search(r"Game Time<br/>\s*<h1>([\d:]+)</h1>", html)
    if not m:
        return None
    mins, secs = (int(x) for x in m.group(1).split(":"))
    out = dict(duration=mins + secs / 60, duration_str=m.group(1))
    v = re.search(r"v(\d+\.\d+)\s*</div>", html)
    out["version"] = v.group(1) if v else None
    teams = []
    for side in ("blue", "red"):
        i = html.find(f"{side}-line-header")
        header = html[i: html.find("</div></div>", i)]
        name = re.search(r"title='([^']+) stats'", header).group(1)
        block = html[i: html.find("red-line-header") if side == "blue" else html.find(">Bans<", i)]
        block = ALT_IMG.sub(r"[\1]", block)
        text = re.sub(r"<[^>]+>", " ", block)
        text = re.sub(r"\s+", " ", text)
        grab = lambda label: _num((re.search(rf"\[{label}\]\s*([\d.]+k?)", text) or [None, ""])[1])
        teams.append(dict(side=side, team=name, result=int("WIN" in header),
                          kills=grab("Kills"), towers=grab("Towers"), dragons=grab("Dragons"),
                          barons=grab("Nashor"), gold=grab("Team Gold"),
                          dragon_types=", ".join(re.findall(r"\[(\w+) Drake\]", text))))
    # Objective timeline: blue_action / red_action spans hold "time<br/>[objective]".
    events = []
    for side, body in re.findall(r"<span class='(blue|red)_action'>(.*?)</span>", html, re.S):
        body = ALT_IMG.sub(r"[\1]", body)
        t = re.search(r"(\d+):(\d+)", body)
        objs = re.findall(r"\[([^\]]+)\]", body)
        for o in objs:
            events.append(dict(side=side, minute=int(t.group(1)) + int(t.group(2)) / 60 if t else np.nan,
                               time=f"{t.group(1)}:{t.group(2)}" if t else "", objective=o))
    out["timeline"] = events
    for tm in teams:
        side_events = [e["objective"] for e in events if e["side"] == tm["side"]]
        tm["heralds"] = side_events.count("Rift Herald")
        tm["first_blood"] = int("First blood" in side_events)
        tm["first_tower"] = int("First Tower" in side_events)
    # Plates / voidgrubs table: label row then blue value, red value.
    plates_html = html[html.find("<th>Plates</th>"):]
    plates_html = plates_html[: plates_html.find("</table>")]
    cells = [re.sub(r"<[^>]+>", " ", c).strip() for c in re.findall(r"<div class=\"col-4[^\"]*\">(.*?)</div>", plates_html, re.S)]
    for k in range(0, len(cells) - 2, 3):
        label = re.sub(r"\s+", " ", ALT_IMG.sub("", cells[k])).strip().lower()
        key = {"voidgrubs": "voidgrubs", "plates": "plates", "plates top": "plates_top", "plates mid": "plates_mid",
               "plates bot": "plates_bot"}.get(re.sub(r"[^a-z ]", "", label).strip())
        if key:
            teams[0][key], teams[1][key] = _num(cells[k + 1]), _num(cells[k + 2])
    out["teams"] = teams
    g = re.search(r"label: 'Gold',\s*data: \[([^\]]*)\]", html)
    out["gold_lead"] = [float(x) for x in g.group(1).split(",") if x.strip()] if g else []
    return out


def parse_fullstats(html: str) -> pd.DataFrame:
    """The 'All stats' table: one column per player (blue top..support, then red), one row per stat."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="completestats")
    if table is None:
        return pd.DataFrame()
    rows = {}
    for tr in table.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        if len(cells) >= 2:
            rows[cells[0]] = cells[1:]
    players = rows.get("Player", [])
    if len(players) != 10:
        return pd.DataFrame()
    df = pd.DataFrame({"player": players, "slot": range(10)})
    for stat, values in rows.items():
        if stat in ("Player", "Role"):
            continue
        values = (values + [""] * 10)[:10]
        df[stat] = [_num(v) for v in values]
    df["role"] = [r.lower().replace("adc", "bot") for r in (rows.get("Role", []) + [""] * 10)[:10]]
    df["side"] = ["blue"] * 5 + ["red"] * 5
    return df


def build(domestic: bool = True, incremental: bool = False, log=print) -> dict:
    """International events, plus every domestic league game when `domestic` (needed for player history).

    `incremental` keeps the games already in the match_*.csv files and only downloads new ones
    (what the scheduled cloud refresh uses, since it has no page cache)."""
    from .events import EVENTS
    from .scrape_matches import league_of, list_tournaments
    sources = [(key, None, ev["title"], ev["tournaments"]) for key, ev in EVENTS.items()]
    if domestic:
        by_league = {}
        for t in list_tournaments():
            by_league.setdefault(league_of(t), []).append(t)
        sources += [(None, lg, f"{lg} (domestic)", ts) for lg, ts in by_league.items()]
    games, teams, plist, timeline, gold = [], [], [], [], []
    names_out = ["games", "teams", "players", "timeline", "gold"]
    existing = {}
    if incremental and (PROCESSED / "match_games.csv").exists():
        existing = {n: pd.read_csv(PROCESSED / f"match_{n}.csv", dtype={"event": str, "league": str, "series_id": str})
                    for n in names_out}
    known = set(existing["games"].game_uid) if existing else set()
    for key, league, title, tournaments in sources:
        n_before = len(games)
        for t in tournaments:
            for s in list_series(t):
                for game_no, gid in game_ids(s["series_id"], s["page"]):
                    uid = f"{s['series_id']}-{game_no}"
                    if uid in known:
                        continue
                    try:
                        page = http.get(f"{BASE}/game/stats/{gid}/page-game/")
                        full = http.get(f"{BASE}/game/stats/{gid}/page-fullstats/")
                    except RuntimeError as e:
                        log(f"    skipped {uid}: {e}")
                        continue
                    info = parse_game_page(page or "")
                    stats = parse_fullstats(full or "")
                    if not info or stats.empty:
                        continue
                    blue, red = info["teams"]
                    games.append(dict(game_uid=uid, gol_id=gid, event=key, league=league or key,
                                      series_id=s["series_id"], tournament=t, stage=s["stage"],
                                      date=s["date"], game_no=game_no, version=info["version"],
                                      duration=info["duration"], duration_str=info["duration_str"],
                                      blue=blue["team"], red=red["team"],
                                      winner=blue["team"] if blue["result"] else red["team"]))
                    for tm in info["teams"]:
                        teams.append(dict(game_uid=uid, **tm))
                    names = {"blue": blue["team"], "red": red["team"]}
                    stats["team"] = stats.side.map(names)
                    stats["result"] = stats.side.map({"blue": blue["result"], "red": red["result"]})
                    stats["game_uid"] = uid
                    stats["duration"] = info["duration"]
                    # Champions come from the same game page (role order), so new games need no extra data.
                    champ = {(r["team"], r["role"]): r["champion"] for r in parse_players(page or "")}
                    stats["champion"] = [champ.get((tm, r)) for tm, r in zip(stats.team, stats.role)]
                    plist.append(stats)
                    for e in info["timeline"]:
                        timeline.append(dict(game_uid=uid, team=names[e["side"]], **e))
                    for minute, lead in enumerate(info["gold_lead"]):
                        gold.append(dict(game_uid=uid, minute=minute, blue_lead=lead))
        log(f"  {title}: {len(games) - n_before} games")
    new = dict(games=pd.DataFrame(games), teams=pd.DataFrame(teams),
               players=pd.concat(plist, ignore_index=True) if plist else pd.DataFrame(),
               timeline=pd.DataFrame(timeline), gold=pd.DataFrame(gold))
    out = {n: pd.concat([existing[n], new[n]], ignore_index=True) if existing else new[n] for n in names_out}
    for name, df in out.items():
        df.to_csv(PROCESSED / f"match_{name}.csv", index=False)
    log(f"  saved {len(out['games'])} games ({len(new['games'])} new), {len(out['players'])} player-games")
    return out
