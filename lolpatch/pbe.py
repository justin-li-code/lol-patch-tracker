"""Forecast the next patch from the PBE, before Riot publishes the patch notes.

Riot tests the next patch on the PBE (Public Beta Environment) about two weeks before release.
Community Dragon mirrors both the live and the PBE game files, so comparing the two gives every
champion number that's about to change: base stats, ability values, cooldowns and costs. Each
difference becomes a "before => after" line, scored the same way as a line in the patch notes,
and the forecast model then predicts each champion's pro pick/ban rate on that patch.

PBE changes are still being tested: Riot often adjusts or reverts them before release, so the
forecast is re-run every time the PBE files change.

Outputs (data/processed): pbe_lines.csv (one row per changed number), pbe_changes.csv (one
verdict per champion), pbe_predictions.csv (the forecast) and pbe_meta.json.
"""
import json
import re
import time

import pandas as pd
import requests

from .champions import champ_key
from .config import GAME_VERSION_MAJOR, PATCH_BRAND_MAJOR, PROCESSED, RAW, USER_AGENT
from .scrape_patches import LOWER_IS_BETTER, score_line, summarise

CDRAGON = "https://raw.communitydragon.org"
# Skins, emotes and test dummies carry spell data too, but never change how a champion plays.
NOISE = re.compile(r"Skin\d|VFX|Emote|Dummy|Recall|Taunt|Dance|Joke|Laugh|Jade_|Cosmetic", re.I)
BASE_STATS = {
    "baseHPModifiable": "Health", "hpPerLevelModifiable": "Health per level",
    "baseStaticHPRegenModifiable": "Health regen", "hpRegenPerLevelModifiable": "Health regen per level",
    "baseDamageModifiable": "Attack damage", "damagePerLevelModifiable": "Attack damage per level",
    "baseArmorModifiable": "Armor", "armorPerLevelModifiable": "Armor per level",
    "baseMR": "Magic resist", "mrPerLevel": "Magic resist per level",
    "baseMoveSpeedModifiable": "Move speed", "attackRangeModifiable": "Attack range",
    "attackSpeedModifiable": "Attack speed", "attackSpeedRatioModifiable": "Attack speed ratio",
    "attackSpeedPerLevelModifiable": "Attack speed per level",
}
_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT


def _get(version: str, path: str, cache: bool) -> dict | list | None:
    """A Community Dragon file. Numbered versions never change, so they're cached on disk."""
    file = RAW / "cdragon" / version / path.replace("/", "_")
    if cache and file.exists():
        return json.loads(file.read_text())
    for attempt in range(4):
        try:
            r = _session.get(f"{CDRAGON}/{version}/{path}", timeout=60)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        time.sleep(0.2)
        if r.status_code == 404:
            return None
        if r.ok:
            if cache:
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text(r.text)
            return r.json()
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Community Dragon unavailable: {version}/{path}")


def versions() -> tuple[str, str]:
    """(live, pbe) game versions, e.g. ('16.23', '16.24')."""
    v = lambda d: ".".join(_get(d, "content-metadata.json", cache=False)["version"].split(".")[:2])
    return v("latest"), v("pbe")


def to_patch(version: str) -> str:
    """'16.24' -> '26.24'"""
    major, minor = version.split(".")
    return f"{int(major) + PATCH_BRAND_MAJOR - GAME_VERSION_MAJOR}.{minor}"


def _words(name: str) -> str:
    """'ADRatio' -> 'AD Ratio', 'cooldownTime' -> 'Cooldown Time'."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", name).strip().title().replace("Ad ", "AD ") \
        .replace("Ap ", "AP ")


def numbers(record: dict) -> dict[tuple[str, str], tuple]:
    """Every gameplay number in a champion's file: (ability, stat) -> values."""
    out = {}
    for path, obj in record.items():
        if not isinstance(obj, dict) or NOISE.search(path):
            continue
        if obj.get("__type") == "CharacterRecord":
            for key, label in BASE_STATS.items():
                v = obj.get(key)
                v = v.get("baseValue") if isinstance(v, dict) else v
                if isinstance(v, (int, float)):
                    out[("Base Stats", label)] = (round(v, 4),)
        elif obj.get("__type") == "SpellObject":
            spell, name = obj.get("mSpell") or {}, obj.get("mScriptName", path.rsplit("/", 1)[-1])
            for dv in spell.get("DataValues") or []:
                if "values" in dv and "name" in dv:
                    out[(name, _words(dv["name"]))] = tuple(round(x, 4) for x in dv["values"])
            for key, label in [("cooldownTime", "Cooldown"), ("mana", "Mana")]:
                v = spell.get(key)
                if isinstance(v, list):
                    out[(name, label)] = tuple(round(x, 4) for x in v)
    return out


def _show(values: tuple) -> str:
    """Ranks 1-5 as '10/20/30/40/50' (rank 0 and 6 are placeholders in the files)."""
    ranks = values[1:6] if len(values) == 7 else values
    uniq = list(dict.fromkeys(ranks))
    return "/".join(f"{x:g}" for x in (uniq if len(uniq) == 1 else ranks))


def diff(live: dict, pbe: dict, patch: str, champion: str) -> list[dict]:
    """One row per number that differs, in the same columns as the patch-notes lines."""
    a, b = numbers(live), numbers(pbe)
    rows = []
    for ability, stat in sorted(set(a) | set(b)):
        x, y = a.get((ability, stat)), b.get((ability, stat))
        if x == y:
            continue
        before, after = (_show(x) if x else ""), (_show(y) if y else "")
        if x and y:
            direction, magnitude = score_line(stat, before, after, ability)
            # "Cooldown Reduction" or "Damage Taken Reduction" going up is good for the champion.
            if re.search(r"reduction|refund", stat, re.I) and LOWER_IS_BETTER.search(stat):
                direction = -direction
            text = f"{stat}: {before} ⇒ {after}"
        else:
            # Added or removed values are usually renames or reworks: listed, not scored.
            direction, magnitude = 0.0, 0.0
            text = f"{stat}: new ({after})" if y else f"{stat}: removed"
        rows.append(dict(patch=patch, champion=champion, champ_key=champ_key(champion), ability=ability,
                         stat=stat, before=before, after=after, text=text, direction=direction,
                         magnitude=magnitude, context=""))
    return rows


def scrape(log=print) -> tuple[pd.DataFrame, dict]:
    """Compare live and PBE files for every champion."""
    live_v, pbe_v = versions()
    meta = dict(live_version=live_v, pbe_version=pbe_v, live_patch=to_patch(live_v), pbe_patch=to_patch(pbe_v))
    summary = _get(live_v, "plugins/rcp-be-lol-game-data/global/default/v1/champion-summary.json", cache=True)
    champs = [c for c in summary if 0 < c["id"] < 60000]  # 60000+ are Jade_ mode copies
    rows = []
    for c in champs:
        alias = c["alias"].lower()
        path = f"game/data/characters/{alias}/{alias}.bin.json"
        live, pbe = _get(live_v, path, cache=True), _get("pbe", path, cache=False)
        if live and pbe:
            rows += diff(live, pbe, meta["pbe_patch"], c["name"])
    lines = pd.DataFrame(rows, columns=["patch", "champion", "champ_key", "ability", "stat", "before", "after",
                                        "text", "direction", "magnitude", "context"])
    meta["n_champions"] = int(lines.champ_key.nunique())
    log(f"  PBE {pbe_v} vs live {live_v}: {len(lines)} changed numbers on {meta['n_champions']} champions")
    return lines, meta


def run(log=print, force: bool = False) -> bool:
    """PBE forecast. Files are only rewritten when the PBE changed, so the scheduled job doesn't
    commit a new timestamp every time it checks. Returns whether anything changed."""
    from .analyze import build_transitions, champion_stats, load_changes, load_games, pro_patches
    from .model import predict_future
    lines, meta = scrape(log=log)
    meta_path, lines_path = PROCESSED / "pbe_meta.json", PROCESSED / "pbe_lines.csv"
    old = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    same = (lines_path.exists() and lines.to_csv(index=False) == lines_path.read_text()
            and all(old.get(k) == meta[k] for k in ("live_version", "pbe_version")))
    if same and not force:
        log("  No PBE changes since the last check.")
        return False
    lines.to_csv(lines_path, index=False)
    scored = lines[lines.direction != 0]
    # Between cycles the PBE runs the live patch, so there's nothing to forecast.
    changes = summarise(scored) if meta["pbe_version"] != meta["live_version"] and len(scored) else pd.DataFrame()
    if len(changes):
        changes["source"] = "pbe"
    changes.to_csv(PROCESSED / "pbe_changes.csv", index=False)
    pred_path = PROCESSED / "pbe_predictions.csv"
    if len(changes):
        games, official = load_games(), load_changes()
        stats = champion_stats(games, "ALL")
        # Official notes since the last pro patch, plus the PBE changes for the patch after them.
        predict_future(stats, official, build_transitions(stats, official), meta["pbe_patch"],
                       changes).to_csv(pred_path, index=False)
        meta["base_patch"] = pro_patches(stats)[-1]
        meta["missing_notes"] = [p for p in _patches_between(meta["base_patch"], meta["live_patch"])
                                 if p not in set(official.patch)]
        if meta["missing_notes"]:
            log(f"  warning: no patch notes yet for {', '.join(meta['missing_notes'])}")
    else:
        pred_path.unlink(missing_ok=True)
    meta["updated"] = pd.Timestamp.now(tz="America/Los_Angeles").strftime("%Y-%m-%d %H:%M")
    meta_path.write_text(json.dumps(meta, indent=2))
    log(f"  Saved the PBE forecast for {meta['pbe_patch']}")
    return True


def refresh(log=print) -> bool:
    """What the scheduled job runs: the newest official patch notes (if the live patch moved on),
    then the PBE forecast. Returns whether anything changed."""
    from .analyze import load_changes
    live_patch = to_patch(versions()[0])
    new_notes = live_patch not in set(load_changes().patch)
    if new_notes:
        from . import model, scrape_patches
        log(f"Patch notes for {live_patch}:")
        scrape_patches.scrape(log=log)
        new_notes = live_patch in set(load_changes().patch)
        if new_notes:
            model.run(log=log)
    log("PBE:")
    return run(log=log, force=new_notes) or new_notes


def _patches_between(after: str, through: str) -> list[str]:
    major, lo = after.split(".")
    hi = int(through.split(".")[1])
    return [f"{major}.{m}" for m in range(int(lo) + 1, hi + 1)]
