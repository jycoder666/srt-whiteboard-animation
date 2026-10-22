#!/usr/bin/env python3
"""Load paper themes from assets/themes.json. No render logic."""
from __future__ import annotations

import json
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_THEMES_PATH = _SCRIPT_DIR.parent / "assets" / "themes.json"
FALLBACK_DEFAULT = "paper-cream"
FALLBACK_HEX = "#F6F1E3"


def load_themes(path: Path | None = None) -> dict:
    p = path or DEFAULT_THEMES_PATH
    if not p.is_file():
        return {"default": FALLBACK_DEFAULT, "themes": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def resolve_canvas_hex(
    theme: str | None = None,
    canvas_hex: str | None = None,
    themes_path: Path | None = None,
) -> tuple[str, str]:
    """Return (theme_id_or_override, canvas_hex).

    Priority: canvas_hex > theme > themes.json default > paper-cream.
    """
    if canvas_hex:
        hx = canvas_hex.strip()
        if not hx.startswith("#"):
            hx = "#" + hx
        return ("override", hx)

    data = load_themes(themes_path)
    themes = data.get("themes") or {}
    theme_id = (theme or data.get("default") or FALLBACK_DEFAULT).strip()
    if theme_id not in themes:
        known = ", ".join(sorted(themes)) or "(missing themes.json)"
        raise ValueError(f"unknown theme: {theme_id}; known: {known}")
    return (theme_id, themes[theme_id]["canvas_hex"])
