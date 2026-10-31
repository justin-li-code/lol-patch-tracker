"""How exposed each champion is to item and rune changes.

A champion is exposed to a changed item in proportion to how often pros built it on that
champion before the change: if 80% of Jinx games had Kraken Slayer and Kraken Slayer is
nerfed (score -0.75), Jinx's item exposure is 0.8 x -0.75 = -0.6. Rune exposure works the
same way. Usage is measured on the pro patches before the change window, so a forecast
never sees builds from the patch it predicts.

Inputs: data/processed/builds.csv (builds.py), item_changes.csv (scrape_patches.py).
"""
from functools import lru_cache

import pandas as pd

from .champions import champ_key
from .config import PROCESSED, patch_key

COLUMNS = ["item_exposure", "rune_exposure"]
# Exposure counts each champion as if it had this many extra games without the item, so a
# champion with one game (100% "built it") isn't treated as fully exposed.
SHARE_PRIOR_GAMES = 4


def _stamp(*names: str) -> tuple[float, ...]:
    """File change times, so the caches below refresh when the data does (the site stays up for days)."""
    return tuple((PROCESSED / n).stat().st_mtime if (PROCESSED / n).exists() else 0.0 for n in names)


def usage() -> pd.DataFrame:
    """Per patch, champion and item/rune: games it was built or taken (n) and the champion's games."""
    return _usage(_stamp("builds.csv", "games.csv", "event_games.csv"))


@lru_cache(maxsize=2)
def _usage(_stamps) -> pd.DataFrame:
    path = PROCESSED / "builds.csv"
    if not path.exists():
        return pd.DataFrame(columns=["patch", "champ_key", "kind", "key", "n", "games"])
    b = pd.read_csv(path)
    patch = pd.concat([pd.read_csv(PROCESSED / f, usecols=["game_uid", "patch"], dtype={"patch": str})
                       for f in ("games.csv", "event_games.csv") if (PROCESSED / f).exists()]).drop_duplicates("game_uid")
    b = b.merge(patch, on="game_uid")
    b["champ_key"] = b.champion.map(champ_key)
    games = b.groupby(["patch", "champ_key"]).size().rename("games")
    rows = []
    for kind, col in (("item", "items"), ("rune", "runes")):
        x = b.assign(key=b[col].fillna("").str.split("|")).explode("key")
        x = x[x.key != ""]
        x["key"] = x.key.map(champ_key)
        # Count each item once per player-game (two Needlessly Large Rods are one "built it").
        x = x.drop_duplicates(["game_uid", "team", "role", "key"])
        rows.append(x.groupby(["patch", "champ_key", "key"]).size().rename("n").reset_index().assign(kind=kind))
    return pd.concat(rows, ignore_index=True).join(games, on=["patch", "champ_key"])


def item_changes() -> pd.DataFrame:
    return _item_changes(_stamp("item_changes.csv"))


@lru_cache(maxsize=2)
def _item_changes(_stamps) -> pd.DataFrame:
    path = PROCESSED / "item_changes.csv"
    if not path.exists():
        return pd.DataFrame(columns=["patch", "kind", "name", "key", "status", "verdict", "score"])
    return pd.read_csv(path, dtype={"patch": str})


def changes_in(after: str | None, upto: str) -> pd.DataFrame:
    """Item and rune changes in Riot patches (after, upto]."""
    ch = item_changes()
    lo = patch_key(after) if after else (0, 0)
    return ch[ch.patch.map(lambda p: lo < patch_key(p) <= patch_key(upto))]


def build_shares(before: list[str]) -> pd.DataFrame:
    """Share of each champion's games with each item/rune, pooled over the given pro patches."""
    u = usage()
    u = u[u.patch.isin(before)]
    games = u.drop_duplicates(["patch", "champ_key"]).groupby("champ_key").games.sum()
    share = u.groupby(["champ_key", "kind", "key"]).n.sum().reset_index()
    share["games"] = share.champ_key.map(games)
    share["share"] = share.n / share.games
    share["weight"] = share.n / (share.games + SHARE_PRIOR_GAMES)
    return share


def exposure(before: list[str], after: str | None, upto: str) -> pd.DataFrame:
    """item_exposure / rune_exposure per champ_key for changes in (after, upto], weighted by how
    often each champion used the changed thing on the `before` patches. Champions that never
    built a changed item get 0."""
    ch = changes_in(after, upto)
    if ch.empty:
        return pd.DataFrame(columns=COLUMNS, dtype=float)
    score = ch.groupby(["kind", "key"]).score.sum().rename("score").reset_index()
    sh = build_shares(before).merge(score, on=["kind", "key"])
    sh["x"] = sh.weight * sh.score
    out = sh.pivot_table(index="champ_key", columns="kind", values="x", aggfunc="sum")
    return out.rename(columns={"item": "item_exposure", "rune": "rune_exposure"}).reindex(columns=COLUMNS).fillna(0.0)


def top_exposed(before: list[str], after: str | None, upto: str, min_share: float = 0.25) -> pd.DataFrame:
    """For display: each champion's most-used changed items/runes (share >= min_share)."""
    ch = changes_in(after, upto)
    if ch.empty:
        return pd.DataFrame(columns=["champ_key", "kind", "name", "verdict", "score", "patch", "share", "games"])
    sh = build_shares(before)
    names = ch.drop_duplicates(["kind", "key"], keep="last")[["kind", "key", "name", "verdict", "score", "patch"]]
    out = sh.merge(names, on=["kind", "key"])
    return out[out.share >= min_share].sort_values("share", ascending=False)
