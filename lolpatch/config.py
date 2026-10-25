"""Project-wide settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
OVERRIDES = DATA / "overrides"

YEAR = 2026
# gol.gg season code and the game-version prefix Riot uses internally.
# Riot brands patches as "26.N" but the client/game version is "16.N".
GOL_SEASON = "S16"
GAME_VERSION_MAJOR = 16
PATCH_BRAND_MAJOR = 26

LEAGUES = ["LCK", "LPL", "LCS", "LEC", "CBLOL"]

# Regions for teams that only appear at international events (no domestic data here).
# Teams from LEAGUES are mapped automatically from their league games.
EXTRA_TEAM_REGIONS = {
    "GAM Esports": "LCP", "Team Secret Whales": "LCP", "CTBC Flying Oyster": "LCP",
    "MVK Esports": "LCP", "Deep Cross Gaming": "LCP",
}

# Seconds between live requests (cached pages are free).
REQUEST_DELAY = 1.5
USER_AGENT = "Mozilla/5.0 (lol-patch-analysis; personal research project)"

# Minimum games on a patch (per scope) before we trust its rates.
MIN_GAMES_PER_PATCH = 20


def brand_to_version(patch: str) -> str:
    """'26.15' -> '16.15'"""
    major, minor = patch.split(".")
    return f"{int(major) - (PATCH_BRAND_MAJOR - GAME_VERSION_MAJOR)}.{int(minor)}"


def version_to_brand(version: str) -> str:
    """'16.15' -> '26.15'"""
    major, minor = version.split(".")[:2]
    return f"{int(major) + (PATCH_BRAND_MAJOR - GAME_VERSION_MAJOR)}.{int(minor)}"


def patch_key(patch: str) -> tuple[int, int]:
    major, minor = patch.split(".")[:2]
    return int(major), int(minor)
