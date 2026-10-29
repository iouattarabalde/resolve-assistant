# MagicScript script: first-pass shot match. Park the playhead on the hero, select the other clips of the
# scene, run. Each clip is matched to the hero as graded, with Gain R/G/B in the node labelled Linear.
#
# How: the work is done by _shot_match_engine.py, which sits next to this file (MagicScript skips files
# starting with "_") and is a copy of shot_match.py from the resolve-assistant repo: edit it there, then
# run magicscript/build.py. It runs in a SEPARATE ResolvePython connected to Resolve, so numpy and
# MediaPipe never load into Resolve itself (MagicScript's rule, as in flip_upside_down_shots). This file
# only installs the packages, starts it with the dialog's settings and relays its lines to the log.
# - Preview: values, CSV and a before/after contact sheet in the output folder. Nothing written.
# - Apply: the same, then Gain into each shot's Linear node through SetCDL, which has no layer argument
#   and writes into the layer the node editor SHOWS. The hero is left as it is.
# - Collect: on a scene matched by hand, measures each Linear node and zips stills + truth.json.
# - Collect Final: the same, plus each clip exported with its whole grade on (final/).
# MediaPipe is installed without its OpenCV, matplotlib and sounddevice dependencies (the segmenter
# never uses them; OpenCV's full wheel carries an LGPL FFmpeg): the engine stubs them at import.
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.request

ENGINE = "_shot_match_engine.py"
BUILD = "dev"                                            # replaced by build.py with the repo commit
PIP_PYZ_URL = "https://bootstrap.pypa.io/pip/pip.pyz"
PACKAGES = ["numpy", "tifffile", "pillow", "flatbuffers~=25.9", "absl-py~=2.3", "certifi"]
MEDIAPIPE = "mediapipe==1.2.0"                           # --no-deps, see above

MANIFEST = {
    "id": "shot_match",
    "name": "Shot match to the hero",
    "category": "Colour",
    "keywords": "shot match matching hero scene exposure white balance gain linear node first pass collect",
    "description": "Matches the selected clips to the hero under the playhead, with Gain in each clip's Linear node.",
    "warning": "Apply writes into the layer the node editor is showing, with no undo: show the layer below first.",
    "needs": "timeline",
    "destructive": True,                                 # Apply; destructive_if only reads a yes/no box
    "save_first": True,
    "version": "1.0",
    "author": "Ismael",
    "source": "AI-authored 2026-10-28 for Ismael, from resolve-assistant shot_match.py",
    "third_party": [
        {"name": "numpy", "licence": "BSD-3-Clause", "note": "Installed on first use into your MagicScript pydeps folder with pip; runs in a separate Python process."},
        {"name": "tifffile", "licence": "BSD-3-Clause", "note": "Reads the 16-bit stills. Same route."},
        {"name": "pillow", "licence": "MIT-CMU", "note": "Writes the contact sheet. Same route."},
        {"name": "mediapipe", "licence": "Apache-2.0", "note": "Finds skin. Same route, without its optional dependencies; with flatbuffers and absl-py (Apache-2.0) and certifi (MPL-2.0)."},
        {"name": "MediaPipe selfie multiclass segmentation model", "licence": "Apache-2.0", "note": "Downloaded from Google on first use into ~/.shot_match."},
    ],
    "variables": [
        {"id": "mode", "label": "Mode", "type": "choice", "default": "Preview", "choices": ["Preview", "Apply", "Collect", "Collect Final"],
         "help": "Preview and both Collects write nothing into the grade. Apply writes Gain into each shot."},
        {"id": "layer", "label": "Layer of the Linear node", "type": "int", "default": 3, "min": 1, "max": 9, "pick": False},
        {"id": "node_label", "label": "Node label", "type": "text", "default": "Linear", "pick": False},
        {"id": "out_dir", "label": "Output folder", "type": "text", "default": "~/Documents/ShotMatch", "browse": "folder",
         "help": "Runs older than 30 days in it are deleted, only the ones this script made."},
        {"id": "hero_flag", "label": "Hero flag (several scenes in one run)", "type": "choice", "default": "Green",
         "choices": ["Blue", "Cyan", "Green", "Yellow", "Red", "Pink", "Purple", "Fuchsia", "Rose", "Lavender",
                     "Sky", "Mint", "Lemon", "Sand", "Cocoa", "Cream"],
         "help": "Flag one hero per scene in this colour and select them all. No flagged clip: the hero is under the playhead."},
        {"id": "keep_light", "label": "Keep part of a big light difference", "type": "bool", "default": True,
         "help": "Past 1/2 stop from the hero, only 30 % of the rest is corrected."},
    ],
}


def _resolve_python():
    """Resolve's bundled interpreter (the same lookup as flip_upside_down_shots)."""
    if sys.platform == "win32":
        base = os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                            "Blackmagic Design", "DaVinci Resolve", "ResolvePython")
        cands = [os.path.join(base, "ResolvePython.exe"), os.path.join(base, "python.exe")]
    else:
        cands = ["/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Applications/ResolvePython"]
    for c in cands:
        if os.path.isfile(c):
            return c
    raise RuntimeError("ResolvePython not found (looked in %s)" % ", ".join(cands))


def _run(cmd, timeout):
    flags = 0x08000000 if sys.platform == "win32" else 0         # CREATE_NO_WINDOW: no console flash
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=timeout, creationflags=flags)


def _ensure_packages(ctx, rp, deps, tools):
    probe = ("import sys, importlib.util; sys.path.insert(0, sys.argv[1]); import numpy, tifffile, PIL; "
             "print('OK', numpy.__version__, importlib.util.find_spec('mediapipe') is not None)")
    p = _run([rp, "-s", "-E", "-c", probe, deps], 120)
    if p.returncode == 0 and (p.stdout or "").split()[2:3] == ["True"]:
        return
    ctx.log("Installing numpy, tifffile, Pillow and MediaPipe into %s (one time, about 80 MB)" % deps)
    ctx.progress()
    pyz = os.path.join(tools, "pip.pyz")
    if not os.path.isfile(pyz):
        os.makedirs(tools, exist_ok=True)
        urllib.request.urlretrieve(PIP_PYZ_URL, pyz + ".part")
        os.replace(pyz + ".part", pyz)
    pip = [rp, "-s", "-E", pyz, "install", "--target", deps, "--upgrade", "--only-binary=:all:",
           "--disable-pip-version-check", "--no-input"]
    log = ""
    for cmd in (pip + PACKAGES, pip + ["--no-deps", MEDIAPIPE]):
        p = _run(cmd, 900)
        log += "$ %s\n%s\n%s\n" % (" ".join(cmd), p.stdout or "", p.stderr or "")
        ctx.progress()
    ctx.report("shot_match_pip", log)
    p = _run([rp, "-s", "-E", "-c", probe, deps], 120)
    if p.returncode != 0 or (p.stdout or "").split()[2:3] != ["True"]:
        raise RuntimeError("The packages installed but do not import: " + (p.stderr or "").strip()[-400:])


class _LiveWindow:
    """Progress window inside Resolve for runs started from the Stream Deck or the command line:
    magicscript_run.py collects the log and writes it only at the end, and shows nothing while
    the run lasts. Built the way that runner builds its own dialog (fusionscript.UIDispatcher on
    fusion.UIManager). Best effort: if Resolve refuses, the run carries on without it."""

    def __init__(self, ctx, title):
        self.win = None
        try:
            import DaVinciResolveScript as bmd          # ResolvePython has it built in
            ui = ctx.fusion.UIManager
            self.disp = bmd.UIDispatcher(ui)
            self.win = self.disp.AddWindow({"ID": "ShotMatchLive", "WindowTitle": title, "Geometry": [300, 260, 640, 320]},
                                           ui.VGroup([ui.TextEdit({"ID": "log", "ReadOnly": True})]))
            self.box = self.win.GetItems()["log"]
            self.win.Show()
        except Exception:
            self.win = None

    def add(self, line):
        if self.win is not None:
            try:
                self.box.Append(line)
            except Exception:
                self.win = None

    def pump(self):
        if self.win is not None:
            try:
                self.disp.StepLoop(False)
            except Exception:
                pass

    def close(self):
        if self.win is not None:
            self.win.Hide()


def run(ctx):
    v = ctx.vars
    mode = v["mode"].lower().replace(" ", "_")
    engine = os.path.join(os.path.dirname(os.path.abspath(__file__)), ENGINE)
    if not os.path.isfile(engine):
        raise ValueError("%s is missing: put it next to this script (%s)" % (ENGINE, os.path.dirname(engine)))
    rp = _resolve_python()
    deps = os.path.join(ctx.kit_dir, "pydeps")
    _ensure_packages(ctx, rp, deps, os.path.join(ctx.kit_dir, "pydeps-tools"))
    args = ["--resolve", mode, "--layer", str(int(v["layer"])), "--label", v["node_label"] or "Linear",
            "--out", ctx.lib.expand(v["out_dir"]), "--hero-flag", v.get("hero_flag") or "Green"]
    if not v["keep_light"]:
        args.append("--no-keep-light")
    if mode == "apply":
        args.append("--confirmed")                      # MagicScript's tick replaced the engine's own window
    boot = ("import sys, runpy; sys.path[:0] = [sys.argv[1]]; sys.argv = [sys.argv[2]] + sys.argv[3:]; "
            "runpy.run_path(sys.argv[0], run_name='__main__')")
    ctx.log("Shot match %s (build %s)" % (v["mode"], BUILD))
    flags = 0x08000000 if sys.platform == "win32" else 0
    proc = subprocess.Popen([rp, "-u", "-s", "-E", "-c", boot, deps, engine] + args, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                            creationflags=flags)
    # No window pump means no MagicScript window: started from the Stream Deck or a terminal
    live = _LiveWindow(ctx, "Shot match: " + v["mode"]) if getattr(ctx, "_pump", None) is None else None
    lines = queue.Queue()
    threading.Thread(target=lambda: [lines.put(l.rstrip()) for l in proc.stdout], daemon=True).start()
    seen = []
    try:
        while proc.poll() is None or not lines.empty():
            try:
                line = lines.get(timeout=0.1)
            except queue.Empty:
                ctx.progress()                          # keeps the MagicScript window responsive
                if live:
                    live.pump()
                continue
            if line:
                ctx.log(line)
                seen.append(line.strip())
                if live:
                    live.add(line)
    finally:
        if live:
            live.close()
    proc.stdout.close()
    if proc.returncode != 0:
        raise RuntimeError(seen[-1] if seen else "the engine stopped with code %s" % proc.returncode)
    ends = [l for l in seen if l.startswith(("Values and contact sheet:", "Saved:", "Written:", "Cancelled"))]
    return ends[-1] if ends else "Done"
