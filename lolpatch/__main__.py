"""CLI: python -m lolpatch <command>

    fetch      scrape pro games (gol.gg) and patch notes (Riot)
    analyze    build per-patch stats and change-effect tables
    predict    backtest the model and predict the next patch
    events     scrape international events and score the forecasts (re-run after each Worlds round)
    regions    region/team champion priorities and whether priority wins (runs after events)
    meta       ripple effects: pro roles, substitutions, champion groups
    players    fetch every game page for players and exact roles (slow, ~1 hour first time; cached)
    draft      draft answers: which picks teams make in response to an opponent's pick
    matches    game details, expected results and player ratings for international games
    assets     champion portraits (Data Dragon) and team logos (gol.gg) for the dashboard
    all        fetch + analyze + predict + events + regions + meta + players + draft + matches + assets
    dashboard  open the Streamlit dashboard
"""
import argparse
import subprocess
import sys

import pandas as pd

from .champions import champ_key
from .config import ROOT


def main():
    p = argparse.ArgumentParser(prog="lolpatch", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["fetch", "analyze", "predict", "events", "regions", "meta", "players", "draft", "matches", "assets", "all", "dashboard"])
    p.add_argument("--refresh", action="store_true", help="re-download index pages (new games / patches)")
    p.add_argument("--target", help="patch to predict, e.g. 26.22 (default: latest patch notes)")
    p.add_argument("--notes-url", help="patch notes URL for --target, if not on the usual Riot URL")
    p.add_argument("--changes-csv", help="manual changes for --target: columns champion,verdict[,score]")
    a = p.parse_args()

    if a.command in ("fetch", "all"):
        from . import scrape_matches, scrape_patches
        print("Patch notes:")
        scrape_patches.scrape(refresh=a.refresh)
        print("Pro games:")
        scrape_matches.scrape(refresh=a.refresh)
    if a.command in ("analyze", "all"):
        from . import analyze
        analyze.build_all()
    if a.command in ("predict", "all"):
        from . import model, scrape_patches
        extra = None
        if a.notes_url:
            if not a.target:
                p.error("--notes-url needs --target")
            extra = scrape_patches.changes_from_url(a.notes_url, a.target)
        if a.changes_csv:
            if not a.target:
                p.error("--changes-csv needs --target")
            manual = pd.read_csv(a.changes_csv)
            default = {"buff": 0.75, "nerf": -0.75, "adjust": 0.0}
            manual["score"] = manual.get("score", manual.verdict.map(default)).fillna(manual.verdict.map(default))
            manual = manual.assign(patch=a.target, champ_key=manual.champion.map(champ_key), n_lines=1,
                                   pro_targeted=False, base_stats=False)
            extra = manual if extra is None else pd.concat([extra, manual], ignore_index=True)
        model.run(target_patch=a.target, extra_changes=extra)
    if a.command in ("events", "all"):
        from . import events
        # Event match lists always refresh: Worlds is in progress.
        events.run(refresh=True)
    if a.command in ("events", "regions", "all"):
        from . import regions
        regions.run()
    if a.command in ("events", "meta", "all"):
        from . import meta
        meta.run()
    if a.command in ("players", "all"):
        from . import players
        players.scrape()
    if a.command in ("events", "meta", "players", "draft", "all"):
        from . import draft
        draft.run()
    if a.command in ("events", "matches", "all"):
        from . import matches, ratings
        matches.build()
        ratings.run()
    if a.command in ("assets", "all"):
        from . import assets
        assets.run()
    if a.command == "dashboard":
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard.py")]))


if __name__ == "__main__":
    main()
