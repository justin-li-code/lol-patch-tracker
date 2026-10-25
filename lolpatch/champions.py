"""Champion name normalisation so gol.gg and Riot patch notes line up."""
import re

# Keys after stripping punctuation/spaces and lowercasing.
_ALIASES = {
    "nunu": "nunuwillump",
    "nunuandwillump": "nunuwillump",
    "monkeyking": "wukong",
    "renata": "renataglasc",
}


def champ_key(name: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", name.lower().replace("&", "and"))
    return _ALIASES.get(key, key)
