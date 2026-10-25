#!/usr/bin/env python3
"""First-pass shot match: every selected clip matched to a hero, with Gain only.

From the Resolve Scripts menu it:
  1. takes the selected clips of the current timeline, and the clip under the playhead
     as the hero (it is added to the scene if it was not selected);
  2. checks that each clip has exactly one node labelled "Linear" in layer 3;
  3. exports one still per clip, at its middle frame, with every node of layers 3 and 4
     and of the group Post-Clip graph switched off (except the hero's Linear node, so
     shots match the hero as graded), then switches them back on;
  4. solves exposure and white balance against the hero and prints Gain R/G/B;
  5. only when saved as "Shot Match Apply.py": writes that Gain into each shot's Linear
     node. Saved under any other name it writes nothing.
It moves the playhead (put back at the end) and toggles nodes off and on.

Offline, on stills already exported (one name containing HERO):
  python shot_match.py --stills FOLDER

Solver (v05, tuned on two hand-matched scenes): exposure from surfaces both shots share
(pixels grouped by chromaticity, each group votes how many stops apart they are), pulled
40 % towards skin when both frames show skin. Skin comes from MediaPipe's selfie
multiclass segmenter; without it, surfaces alone. White balance from the same groups.

Needs Python 3.9+, numpy, tifffile, pillow; mediapipe for skin (optional).
Offline, values are relative to the hero still as exported.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import re
import sys
import time
import urllib.request
from pathlib import Path

try:
    import numpy as np
    import tifffile
    from PIL import Image
except ImportError as missing:
    # Resolve loads its own Python: packages must be installed into that one
    print(f"Missing Python package: {missing.name}. In Resolve's Console (Py3), run\n"
          "  import sys; print(sys.prefix)\n"
          "then in a terminal:  \"<that folder>\\python.exe\" -m pip install numpy tifffile pillow mediapipe")
    raise SystemExit(2)

LINEAR_LAYER = 3
CAPTURE_OFF_LAYERS = (3, 4)          # shot and scene layers
LINEAR_LABEL = "Linear"
OUT_ROOT = Path.home() / "Documents" / "ShotMatch"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
             "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite")
MODEL_PATH = Path.home() / ".shot_match" / "selfie_multiclass_256x256.tflite"

# DaVinci Intermediate (Blackmagic white paper)
DI_A, DI_B, DI_C, DI_M = 0.0075, 7.0, 0.07329248, 10.44426855
DI_LOG_CUT = 0.02740668
LUMA_DWG = np.array([0.27411851, 0.8736319, -0.14775041])
DWG_TO_709 = np.array([[1.8986149, -0.79217618, -0.10643872],
                       [-0.16894879, 1.48897575, -0.32002697],
                       [-0.12153916, -0.31567585, 1.43721501]])

ANALYSIS_STEP = 2
CLIP_MARGIN_DI = 0.015     # sensor clip differs per shot, so it is measured per still
GRIDS = [(b, s) for b in (16, 20, 24, 28, 32) for s in (0.0, 0.5)]
MIN_BIN_PIXELS = 200
NEUTRAL_SCALE = 0.05       # chroma distance that doubles a group's weight
SKIN_WEIGHT = 0.4
MIN_SKIN_PIXELS = 300      # at 1/4 resolution
WB_BINS, WB_STEP, WB_ITERATIONS = 24, 0.5, 3


# ─── images ─────────────────────────────────────────────────────

def decode_intermediate(v):
    return np.where(v <= DI_LOG_CUT, v / DI_M, 2.0 ** (v / DI_C - DI_B) - DI_A)


def luma(lin):
    return lin @ LUMA_DWG


def load_still(path: Path) -> np.ndarray:
    img = tifffile.imread(str(path))
    if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
        img = np.moveaxis(img, 0, -1)
    img = img[..., :3]
    if img.dtype == np.uint16:
        return img.astype(np.float64) / 65535.0
    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0
    return img.astype(np.float64)


def preview(lin):
    """Rough DWG linear to Rec.709 view, for the contact sheet and the segmenter only."""
    rgb = np.maximum(lin @ DWG_TO_709.T, 0)
    return np.clip((rgb / (rgb + 0.6) * 1.6) ** (1 / 2.4), 0, 1)


def sky_mask(lin):
    """Bright, not warm, connected to the top edge column by column."""
    y = lin @ np.array([0.2627, 0.6780, 0.0593])
    candidate = (y > 2.0 * np.median(y)) & (lin[..., 2] >= 0.9 * lin[..., 0])
    return np.cumprod(candidate, axis=0).astype(bool)


def analysis_pixels(encoded):
    """Linear pixels worth comparing: not clipped, above the noise floor, no sky."""
    small = encoded[::ANALYSIS_STEP, ::ANALYSIS_STEP]
    lin = decode_intermediate(small)
    clip = small.reshape(-1, 3).max(axis=0) - CLIP_MARGIN_DI
    valid = (small < clip).all(axis=-1) & (small.min(axis=-1) > 0.005)
    valid &= ~sky_mask(lin)
    # Overcast sky the first mask misses: bright, grey, in the upper 60 % of frame
    y = luma(lin)
    sat = (lin.max(-1) - lin.min(-1)) / np.maximum(lin.max(-1), 1e-6)
    upper = np.zeros_like(valid)
    upper[:int(lin.shape[0] * 0.6)] = True
    valid &= ~(upper & (y > np.percentile(y, 60)) & (sat < 0.25))
    return lin[valid]


# ─── skin ───────────────────────────────────────────────────────

class SkinFinder:
    """MediaPipe multiclass selfie segmenter: classes 2 (body skin) and 3 (face skin)."""

    def __init__(self):
        self.segmenter, self.reason = None, ""
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError:
            self.reason = "mediapipe is not installed"
            return
        try:
            if not MODEL_PATH.exists():
                MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            self.segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=str(MODEL_PATH)), output_category_mask=True))
            self.mp = mp
        except Exception as exc:  # a missing model or GPU library must not stop the match
            self.reason = f"skin model unavailable ({exc})"

    def pixels(self, encoded):
        if self.segmenter is None:
            return np.empty((0, 3))
        small = encoded[::4, ::4]
        lin = decode_intermediate(small)
        rgb = np.ascontiguousarray((preview(lin) * 255).astype(np.uint8))
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        classes = np.squeeze(self.segmenter.segment(image).category_mask.numpy_view())
        sat = (lin.max(-1) - lin.min(-1)) / np.maximum(lin.max(-1), 1e-6)
        clip = small.reshape(-1, 3).max(0).min() - 0.02
        ok = ((classes == 2) | (classes == 3)) & (small.max(-1) < clip) & (small.min(-1) > 0.005)
        ok &= (sat > 0.08) & (sat < 0.6)   # drops saturated props the segmenter swallows
        return lin[ok]

    def close(self):
        if self.segmenter is not None:
            self.segmenter.close()   # before interpreter shutdown, or MediaPipe errors on exit


# ─── solver ─────────────────────────────────────────────────────

def chroma(px):
    return px[:, :2] / np.maximum(px.sum(-1, keepdims=True), 1e-6)


def surface_vote(src, hero, bin_count, shift):
    """Stops between shot and hero over the colour groups both show. Grey stone, pavement
    and black leather share one group across many brightnesses, so groups far from
    neutral weigh more, and only the middle half of the votes is averaged."""
    def bins(c):
        idx = np.clip((c * bin_count + shift).astype(int), 0, bin_count)
        return idx[:, 0] * (bin_count + 1) + idx[:, 1]
    hc, sc = chroma(hero), chroma(src)
    hb, sb = bins(hc), bins(sc)
    stops, weights = [], []
    for b in np.intersect1d(sb, hb):
        a, h = sb == b, hb == b
        if a.sum() < MIN_BIN_PIXELS or h.sum() < MIN_BIN_PIXELS:
            continue
        ratio = np.percentile(luma(hero[h]), [25, 50, 75]) / np.percentile(luma(src[a]), [25, 50, 75])
        stops.append(np.median(np.log2(ratio)))
        distance = np.linalg.norm(np.median(hc[h], 0) - 1 / 3)
        weights.append(np.sqrt(min(a.sum(), h.sum())) * (1 + distance / NEUTRAL_SCALE))
    if not stops:
        return np.nan
    stops, weights = np.array(stops), np.array(weights)
    order = np.argsort(stops)
    stops, weights = stops[order], weights[order]
    cum = np.cumsum(weights) / weights.sum()
    keep = (cum >= 0.25) & (cum - weights / weights.sum() <= 0.75)
    return np.average(stops[keep], weights=weights[keep])


def white_balance(src, hero):
    """Luma-neutral channel multipliers from the colour shift of shared groups, in half steps."""
    def bins(px):
        idx = np.clip((chroma(px) * WB_BINS).astype(int), 0, WB_BINS - 1)
        return idx[:, 0] * WB_BINS + idx[:, 1]
    wb = np.ones(3)
    hero_bins = bins(hero)
    for _ in range(WB_ITERATIONS):
        src_bins = bins(src * wb)
        shifts, weights = [], []
        for b in np.intersect1d(src_bins, hero_bins):
            a, h = src[src_bins == b] * wb, hero[hero_bins == b]
            if len(a) < MIN_BIN_PIXELS or len(h) < MIN_BIN_PIXELS:
                continue
            shifts.append(np.log(np.median(h, 0) / np.median(a, 0)))
            weights.append(min(len(a), len(h)))
        if not shifts:
            break
        shift = np.average(np.array(shifts), axis=0, weights=weights)
        shift -= np.log(luma(np.exp(shift)))
        wb = wb * np.exp(shift * WB_STEP)
    return wb / luma(wb)


def solve(src, hero, src_skin, hero_skin) -> dict:
    votes = np.array([surface_vote(src, hero, b, s) for b, s in GRIDS])
    votes = votes[~np.isnan(votes)]
    if votes.size == 0:
        return {"gain": None, "note": "no surface in common with the hero"}
    surfaces = float(np.median(votes))
    skin = np.nan
    exposure = surfaces
    if len(src_skin) >= MIN_SKIN_PIXELS and len(hero_skin) >= MIN_SKIN_PIXELS:
        skin = float(np.log2(np.median(luma(hero_skin)) / np.median(luma(src_skin))))
        exposure = surfaces + SKIN_WEIGHT * (skin - surfaces)
    # Balancing before voting was tried: lower RMS, but 8/11 instead of 10/11 within 1/3 stop
    gain = 2 ** exposure * white_balance(src, hero)
    return {"gain": gain, "surfaces": surfaces, "skin": skin, "spread": float(votes.std()), "note": ""}


def match_stills(stills: list[tuple[str, Path]], hero_name: str, out_dir: Path) -> list[dict]:
    """stills: (clip name, still path). Writes match_values.csv and contact_sheet.jpg."""
    skin_finder = SkinFinder()
    if skin_finder.reason:
        print(f"  Skin: off, {skin_finder.reason}. Matching on surfaces only.")
    encoded = {name: load_still(path) for name, path in stills}
    hero_enc = encoded[hero_name]
    hero_px, hero_skin = analysis_pixels(hero_enc), skin_finder.pixels(hero_enc)
    rows, sheet = [], []
    for name, _ in stills:
        if name == hero_name:
            r = {"gain": np.ones(3), "surfaces": 0.0, "skin": np.nan, "spread": 0.0, "note": "hero"}
        else:
            enc = encoded[name]
            r = solve(analysis_pixels(enc), hero_px, skin_finder.pixels(enc), hero_skin)
        r["clip"] = name
        rows.append(r)
        lin = decode_intermediate(encoded[name][::4, ::4])
        after = lin * r["gain"] if r["gain"] is not None else lin
        sheet.append(np.hstack([preview(lin), preview(after)]))
    skin_finder.close()
    write_outputs(rows, sheet, out_dir)
    return rows


def write_outputs(rows: list[dict], sheet: list[np.ndarray], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "match_values.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "exposure_stops", "gain_r", "gain_g", "gain_b", "offset", "gamma",
                    "surfaces_stops", "skin_stops", "grid_spread_stops", "note"])
        for r in rows:
            g = r["gain"]
            if g is None:
                w.writerow([r["clip"], "", "", "", "", "", "", "", "", "", r["note"]])
                continue
            w.writerow([r["clip"], f"{np.log2(luma(g)):+.2f}", f"{g[0]:.2f}", f"{g[1]:.2f}", f"{g[2]:.2f}",
                        "25.00", "0.00", f"{r['surfaces']:+.2f}",
                        "" if np.isnan(r["skin"]) else f"{r['skin']:+.2f}", f"{r['spread']:.2f}", r["note"]])
    height = 216
    tiles = [Image.fromarray((t * 255).astype(np.uint8)) for t in sheet]
    tiles = [t.resize((int(t.width * height / t.height), height)) for t in tiles]
    canvas = Image.new("RGB", (max(t.width for t in tiles) + 20, (height + 10) * len(tiles) + 10), (40, 40, 40))
    for i, t in enumerate(tiles):
        canvas.paste(t, (10, 10 + i * (height + 10)))
    canvas.save(out_dir / "contact_sheet.jpg", quality=90)


def print_table(rows: list[dict]) -> None:
    print("\n  Linear node, Gain R / G / B (Offset 25.00, Gamma 0.00), matched to the hero as graded.")
    for r in rows:
        g = r["gain"]
        if g is None:
            print(f"  {r['clip'][:40]:40s}  skipped: {r['note']}")
            continue
        flag = "  (hero)" if r["note"] == "hero" else ""
        print(f"  {r['clip'][:40]:40s}  {g[0]:.2f}  {g[1]:.2f}  {g[2]:.2f}   {np.log2(luma(g)):+.2f} stops{flag}")


# ─── Resolve ────────────────────────────────────────────────────

class Stop(Exception):
    """A plain-language reason to stop, printed as is."""


def get_resolve():
    injected = globals().get("resolve")   # the Scripts menu provides it
    if injected:
        return injected
    try:
        import DaVinciResolveScript as dvr
    except ImportError:
        raise Stop("Run this from Workspace > Scripts inside Resolve, or set up the Resolve "
                   "scripting environment variables for the terminal.")
    r = dvr.scriptapp("Resolve")
    if not r:
        raise Stop("Resolve is not running, or external scripting is off "
                   "(Preferences > System > General > External scripting using: Local).")
    return r


def frames_to_tc(frame: int, fps: int) -> str:
    s, f = divmod(frame, fps)
    return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}:{f:02d}"


def selected_clips(timeline) -> list:
    clips = timeline.GetSelectedClips()
    if clips is None:
        raise Stop("This Resolve has no Timeline.GetSelectedClips() (it arrived in 21.0.4). "
                   "Update Resolve, then run this again.")
    if isinstance(clips, dict):
        clips = list(clips.values())
    return [c for c in clips if c]


def linear_node(item) -> int | None:
    """Index of the one node labelled Linear in layer 3, or None when there are 0 or 2+."""
    graph = item.GetNodeGraph(LINEAR_LAYER)
    if not graph:
        return None
    hits = [i for i in range(1, (graph.GetNumNodes() or 0) + 1) if graph.GetNodeLabel(i) == LINEAR_LABEL]
    return hits[0] if len(hits) == 1 else None


def capture_graphs(item) -> list[tuple[str, object]]:
    graphs = [(f"layer {layer}", item.GetNodeGraph(layer)) for layer in CAPTURE_OFF_LAYERS]
    group = item.GetColorGroup()
    if group:
        graphs.append(("Post-Clip", group.GetPostClipNodeGraph()))
    return [(where, g) for where, g in graphs if g]


def export_ungraded_still(project, timeline, item, fps: int, path: Path, keep_linear: int | None = None) -> list[str]:
    """Middle frame of the clip with layers 3-4 and Post-Clip off, except the Linear node
    `keep_linear` when given (the hero's, so shots match the hero as graded). Returns problems."""
    problems = []
    mid = (int(item.GetStart()) + int(item.GetEnd())) // 2
    if not timeline.SetCurrentTimecode(frames_to_tc(mid, fps)):
        return ["could not move the playhead to the clip"]
    switched = []
    try:
        for where, graph in capture_graphs(item):
            for i in range(1, (graph.GetNumNodes() or 0) + 1):
                if where == f"layer {LINEAR_LAYER}" and i == keep_linear:
                    continue
                # No documented way to read a node's on/off state: re-enable only what we switched
                # Resolve objects answer every attribute name, with None for the missing ones
                getter = getattr(graph, "GetNodeEnabled", None)
                was_on = getter(i) if callable(getter) else None
                if was_on is False:
                    continue
                if graph.SetNodeEnabled(i, False):
                    switched.append((graph, i))
                else:
                    problems.append(f"could not switch off node {i}")
        time.sleep(0.5)   # let the viewer re-render before grabbing it
        if not project.ExportCurrentFrameAsStill(str(path)) or not path.exists():
            problems.append("still export failed")
    finally:
        for graph, i in switched:
            if not graph.SetNodeEnabled(i, True):
                problems.append(f"COULD NOT SWITCH NODE {i} BACK ON, check this clip by hand")
    return problems


def write_gains(rows: list[dict], targets: dict, out_dir: Path) -> None:
    """Gain into each shot's Linear node through SetCDL. Tested on 21.1: Slope lands on Gain
    R/G/B one to one, but a CDL Offset is folded into Gain, so Offset stays 0 and Power 1.
    SetCDL replaces the node's wheels: hand-set Lift, Gamma or Gain in that node are lost."""
    written, failed = [], []
    for r in rows:
        if r["gain"] is None or r["note"] == "hero" or r["clip"] not in targets:
            continue
        item, node = targets[r["clip"]]
        g = r["gain"]
        cdl = {"NodeIndex": str(node), "Slope": f"{g[0]:.4f} {g[1]:.4f} {g[2]:.4f}",
               "Offset": "0.0000 0.0000 0.0000", "Power": "1.0000 1.0000 1.0000", "Saturation": "1.0000"}
        (written if item.SetCDL(cdl) else failed).append(r["clip"])
    with open(out_dir / "written.txt", "w") as f:
        f.write("Gain written to the Linear node (layer 3) of:\n" + "\n".join(written) + "\n")
        if failed:
            f.write("\nSetCDL refused:\n" + "\n".join(failed) + "\n")
    print(f"\n  Written: {len(written)} clips." + (f" REFUSED: {', '.join(failed)}" if failed else ""))


def run_in_resolve(apply: bool = False) -> int:
    resolve = get_resolve()
    project = resolve.GetProjectManager().GetCurrentProject()
    timeline = project.GetCurrentTimeline() if project else None
    if not timeline:
        raise Stop("Open a project and a timeline first, then run this again.")
    if timeline.GetSetting("timelineDropFrameTimecode") == "1":
        raise Stop("Drop-frame timelines are not supported yet.")
    fps = round(float(timeline.GetSetting("timelineFrameRate")))
    hero = timeline.GetCurrentVideoItem()
    if not hero:
        raise Stop("Park the playhead on the hero clip, then run this again.")
    clips = selected_clips(timeline)
    if hero.GetUniqueId() not in {c.GetUniqueId() for c in clips}:
        clips.append(hero)
    if len(clips) < 2:
        raise Stop("Select the clips of the scene (the hero can be one of them), then run this again.")

    resolve.OpenPage("color")
    start_tc = timeline.GetCurrentTimecode()
    out_dir = OUT_ROOT / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    still_dir = out_dir / "stills"
    still_dir.mkdir(parents=True, exist_ok=True)
    if apply:
        print(f"Shot match APPLY: {len(clips)} clips, hero {hero.GetName()}. "
              "Gain will be written into each Linear node, the hero's is left as it is.")
    else:
        print(f"Shot match preview: {len(clips)} clips, hero {hero.GetName()}. Nothing is written to the grade.")

    stills, report, targets = [], [], {}
    try:
        for n, item in enumerate(sorted(clips, key=lambda c: c.GetStart()), 1):
            name = f"{n:02d}_{item.GetName()}"
            node = linear_node(item)
            if node is None:
                report.append(f"  {name}: skipped, needs exactly one node labelled Linear in layer {LINEAR_LAYER}")
                continue
            path = still_dir / f"{name}.tif"
            is_hero = item.GetUniqueId() == hero.GetUniqueId()
            problems = export_ungraded_still(project, timeline, item, fps, path, node if is_hero else None)
            report.append(f"  {name}: Linear is node {node} of layer {LINEAR_LAYER}"
                          + (f"; {'; '.join(problems)}" if problems else ""))
            if path.exists():
                stills.append((name, path, is_hero))
                targets[name] = (item, node)
    finally:
        timeline.SetCurrentTimecode(start_tc)
    print("\n".join(report))

    heroes = [s for s in stills if s[2]]
    if not heroes:
        raise Stop("No still of the hero, so nothing to match against. See the lines above.")
    sample = tifffile.imread(str(heroes[0][1]))
    print(f"\n  Still format: {sample.shape[1]}x{sample.shape[0]}, {sample.dtype}")
    rows = match_stills([(s[0], s[1]) for s in stills], heroes[0][0], out_dir)
    print_table(rows)
    if apply:
        write_gains(rows, targets, out_dir)
    print(f"\n  Stills, values and contact sheet: {out_dir}")
    return 0


def run_offline(folder: Path) -> int:
    stills = sorted((p for p in folder.iterdir() if p.suffix.lower() in (".tif", ".tiff")),
                    key=lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)])
    heroes = [p for p in stills if "HERO" in p.name.upper()]
    if len(heroes) != 1:
        raise Stop(f"Name exactly one still with HERO in it (found {len(heroes)}).")
    rows = match_stills([(p.stem, p) for p in stills], heroes[0].stem, folder / "_shot_match")
    print_table(rows)
    print(f"\n  Values and contact sheet: {folder / '_shot_match'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stills", type=Path, help="match exported stills offline instead of driving Resolve")
    args = parser.parse_args(argv)
    try:
        if args.stills:
            return run_offline(args.stills)
        # Saved as "Shot Match Apply.py" in the Scripts folder, the same file writes the grade
        return run_in_resolve(apply="apply" in Path(globals().get("__file__", "")).stem.lower())
    except Stop as stop:
        print(stop)
        return 2


if __name__ == "__main__":
    if "resolve" in globals():
        main([])   # inside Resolve: sys.exit would only print a traceback in the console
    else:
        sys.exit(main())
