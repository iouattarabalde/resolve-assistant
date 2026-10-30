#!/usr/bin/env python3
"""Builds the MagicScript copy of Shot Match from this repo.

  python magicscript/build.py [OUT]      (default: dist/magicscript)

Writes OUT/Colour/shot_match.py (the MagicScript script, stamped with the commit) and
OUT/Colour/_shot_match_engine.py (a copy of shot_match.py). Copy the Colour folder's two files
into %APPDATA%\\Blackmagic Design\\DaVinci Resolve\\Support\\MagicScript\\scripts\\Colour\\ and press
Reload in the MagicScript window. Edit the repo files, never the copies.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build(out: Path) -> Path:
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        if subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD"]).returncode:
            commit += "+edits"
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    script = (ROOT / "magicscript" / "shot_match.py").read_text(encoding="utf-8")
    if 'BUILD = "dev"' not in script:
        raise SystemExit("magicscript/shot_match.py has no BUILD = \"dev\" line to stamp")
    folder = out / "Colour"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "shot_match.py").write_text(script.replace('BUILD = "dev"', f'BUILD = "{commit}"', 1), encoding="utf-8")
    (folder / "_shot_match_engine.py").write_bytes((ROOT / "shot_match.py").read_bytes())
    return folder


if __name__ == "__main__":
    folder = build(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dist" / "magicscript")
    print(f"Built {folder / 'shot_match.py'} and {folder / '_shot_match_engine.py'}")
