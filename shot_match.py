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

Saved as "Shot Match Collect.py", on a scene you already matched by hand, it writes nothing
either: it exports each clip with its Linear node off and on (every other node of layers 3 and 4
off both times), measures the Gain, Gamma and Offset of your node (and whether Offset was
used at all), and saves stills plus truth.json, so the scene can tune the solver.

Scoring the solver against collected scenes:
  python shot_match.py --bench FOLDER [FOLDER ...]

Offline, on stills already exported (one name containing HERO):
  python shot_match.py --stills FOLDER

Solver (v07, tuned on five hand-matched scenes): exposure from surfaces both shots share
(pixels grouped by chromaticity, each group votes how many stops apart they are), pulled
10 % towards skin when both frames show skin. Skin comes from MediaPipe's selfie
multiclass segmenter; without it, surfaces alone. Past 1/2 stop from the hero only 30 % of
the rest is corrected (other light is kept). White balance from the same groups.
Shots of one camera angle (thumbnails alike) share one Gain unless the light moved, and
shots are solved under the hero's own Gamma, measured from one extra hero export.

Needs Python 3.9+, numpy, tifffile, pillow; mediapipe for skin (optional).
Offline, values are relative to the hero still as exported.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime
import json
import re
import shutil
import subprocess
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
KEEP_DAYS = 30                       # older runs in OUT_ROOT are deleted at the start of a run
RUN_NAME = re.compile(r"(collect_)?(\d{8}_\d{6})(\.zip)?")
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
PROGRESS_EVERY = 10        # seconds between progress lines in the Console
EXTRA_FRAMES = (0.2, 0.8)  # more frames per clip, as fractions of its length (the middle is the main one)
EXTRA_SIZE = 960           # longer edge of the extra frames, pixels
CLIP_MARGIN_DI = 0.015     # sensor clip differs per shot, so it is measured per still
GRIDS = [(b, s) for b in (16, 20, 24, 28, 32) for s in (0.0, 0.5)]
MIN_BIN_PIXELS = 200
NEUTRAL_SCALE = 0.05       # chroma distance that doubles a group's weight
KEEP_LIGHT_FROM = 0.5      # stops: beyond this, only part of the gap to the hero is closed
KEEP_LIGHT_SHARE = 0.3     # share of the gap beyond KEEP_LIGHT_FROM that is corrected
SKIN_WEIGHT = 0.1   # 0.4 until 2026-10-28; 0.1 scores best on the 4 bench scenes (56/60)
MIN_SKIN_PIXELS = 300      # at 1/4 resolution
SAME_ANGLE = 0.9           # thumbnail correlation above which two shots are one camera angle
LIGHT_CHANGE_STOPS = 1 / 3 # a shot solved this far from its group keeps its own value
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


def analysis_pixels(encoded, step: int = ANALYSIS_STEP):
    """Linear pixels worth comparing: not clipped, above the noise floor, no sky."""
    small = encoded[::step, ::step]
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

    def pixels(self, encoded, step: int = 4):
        if self.segmenter is None:
            return np.empty((0, 3))
        small = encoded[::step, ::step]
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


def keep_light(exposure: float) -> float:
    """A shot far brighter or darker than the hero is usually other light (sun against the
    shade of a shelter), which a first pass keeps: close the first KEEP_LIGHT_FROM stops,
    then only KEEP_LIGHT_SHARE of the rest. Tuned on the 5 bench scenes, 2026-10-28."""
    size = abs(exposure)
    if size <= KEEP_LIGHT_FROM:
        return exposure
    return float(np.sign(exposure) * (KEEP_LIGHT_FROM + (size - KEEP_LIGHT_FROM) * KEEP_LIGHT_SHARE))


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
    exposure = keep_light(exposure)
    # Balancing before voting was tried: lower RMS, but 8/11 instead of 10/11 within 1/3 stop
    gain = 2 ** exposure * white_balance(src, hero)
    return {"gain": gain, "surfaces": surfaces, "skin": skin, "spread": float(votes.std()), "note": ""}


def node_parts(node: dict | None):
    """Gamma exponent and Offset lift of a measured Linear node (none: neutral)."""
    if not node:
        return np.ones(3), np.zeros(3)
    return np.asarray(node.get("power", [1, 1, 1]), float), node_lift(node.get("offset"))


def extra_paths(path: Path) -> list[Path]:
    """The extra frames saved next to a still: NAME__f1.tif, NAME__f2.tif..."""
    return sorted(path.parent.glob(glob_escape(path.stem) + "__f*.tif"))


def glob_escape(text: str) -> str:
    return re.sub(r"([\[\]*?])", r"[\1]", text)


def match_stills(stills: list[tuple[str, Path]], hero_name: str, out_dir: Path,
                 hero_node: dict | None = None) -> list[dict]:
    """stills: (clip name, still path), the hero as graded. Writes match_values.csv and
    contact_sheet.jpg. hero_node: the hero's Linear node as measured (Gain, Gamma, Offset).
    The hero is taken back before its Gamma and Offset, so each Gain is the one that, under
    the same Gamma and Offset, matches the hero: the value for a node that keeps them. Shots
    of the hero's angle get the hero's own Gain."""
    skin_finder = SkinFinder()
    if skin_finder.reason:
        print(f"  Skin: off, {skin_finder.reason}. Matching on surfaces only.")
    encoded = {name: load_still(path) for name, path in stills}

    def pooled(name, path):
        """Surface and skin pixels of the main frame and of its extra frames, if any."""
        extras = [load_still(x) for x in extra_paths(path)]
        if not extras:
            enc = encoded[name]
            return analysis_pixels(enc), skin_finder.pixels(enc)
        # Extra frames are smaller: bring the main one to their size so each frame counts alike
        factor = max(1, round(encoded[name].shape[1] / extras[0].shape[1]))
        frames = [encoded[name][::factor, ::factor]] + extras
        step, skin_step = max(1, ANALYSIS_STEP // factor), max(1, 4 // factor)
        return (np.concatenate([analysis_pixels(f, step) for f in frames]),
                np.concatenate([skin_finder.pixels(f, skin_step) for f in frames]))
    paths = dict(stills)
    hero_px, hero_skin = pooled(hero_name, paths[hero_name])
    power, lift = node_parts(hero_node)
    if not (np.allclose(power, 1, atol=0.005) and not lift.any()):
        def before_node(px):
            return np.maximum(px ** (1 / power) - lift, 1e-6)
        hero_px, hero_skin = before_node(hero_px), before_node(hero_skin)
    rows, sheet = [], []
    progress = Progress("Solve", len(stills))
    for done, (name, _) in enumerate(stills):
        if done:
            progress.step(done)
        if name == hero_name:
            r = {"gain": np.ones(3), "surfaces": 0.0, "skin": np.nan, "spread": 0.0, "note": "hero"}
        else:
            px, skin = pooled(name, paths[name])
            r = solve(px, hero_px, skin, hero_skin)
        r["clip"] = name
        rows.append(r)
    progress.step(len(stills))
    skin_finder.close()
    group_by_angle(rows, {name: fingerprint(enc) for name, enc in encoded.items()},
                   hero_node["gain"] if hero_node else None)
    for r in rows:
        lin = decode_intermediate(encoded[r["clip"]][::4, ::4])
        after = np.maximum(lin * r["gain"] + lift, 0) ** power if r["gain"] is not None else lin
        sheet.append(np.hstack([preview(lin), preview(after)]))
    write_outputs(rows, sheet, out_dir)
    return rows


def fingerprint(encoded: np.ndarray) -> np.ndarray:
    """48 x 27 thumbnail of log luma, centred and scaled to unit length: two shots of one
    camera angle correlate above SAME_ANGLE whatever their exposure or contrast."""
    y = np.log2(np.maximum(luma(decode_intermediate(encoded[::8, ::8])), 1e-4))
    v = np.asarray(Image.fromarray(y.astype(np.float32)).resize((48, 27), Image.BILINEAR)).ravel()
    v = v - v.mean()
    return v / max(float(np.linalg.norm(v)), 1e-9)


def camera_tag(name: str) -> str | None:
    """Camera letter from names like CAMA_0043, CAM_B0090 or CAMC_0003, else None."""
    found = re.search(r"CAM_?([A-Z])(?=[_\d])", name.upper())
    return found.group(1) if found else None


def group_by_angle(rows: list[dict], prints: dict, hero_gain=None) -> None:
    """One Gain per camera angle, as in a hand first pass. Shots whose thumbnails correlate
    above SAME_ANGLE (chained) form a group; each takes the group's median Gain, or the
    hero's own Gain when the hero is in it. A shot solved more than LIGHT_CHANGE_STOPS from
    its group keeps its own value: the light moved. Two cameras named in the clip names
    (CAMA, CAM_B...) are never one angle, however alike the thumbnails."""
    names = [r["clip"] for r in rows if r["gain"] is not None]
    group = {n: n for n in names}

    def root(n):
        while group[n] != n:
            n = group[n]
        return n
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            cams = camera_tag(a), camera_tag(b)
            if None not in cams and cams[0] != cams[1]:
                continue
            if float(prints[a] @ prints[b]) > SAME_ANGLE:
                group[root(b)] = root(a)
    by_name = {r["clip"]: r for r in rows}
    groups = {}
    for n in names:
        groups.setdefault(root(n), []).append(by_name[n])
    for members in groups.values():
        shots = [r for r in members if r["note"] != "hero"]
        if len(members) < 2 or not shots:
            continue
        has_hero = len(shots) < len(members)
        if has_hero and hero_gain is not None:
            target = np.asarray(hero_gain, float)
        elif has_hero:
            continue   # the hero's own Gain is unknown, so each shot keeps its solve
        else:
            target = 2 ** np.median([np.log2(r["gain"]) for r in shots], axis=0)
        label = ", ".join(r["clip"][:2] for r in members)
        for r in shots:
            if abs(np.log2(luma(r["gain"]) / luma(target))) > LIGHT_CHANGE_STOPS:
                r["note"] = (r["note"] + "; " if r["note"] else "") + "same angle as " + label + ", light differs"
                continue
            r["solved"], r["gain"] = r["gain"], target
            r["note"] = (r["note"] + "; " if r["note"] else "") + "angle " + label


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


def print_table(rows: list[dict], hero_node: dict | None = None) -> None:
    power, lift = node_parts(hero_node)
    gamma = " ".join(f"{gamma_wheel(p):+.2f}" for p in power)
    offset = " ".join(f"{25 + v / OFFSET_STEP:.2f}" for v in lift)
    print(f"\n  Linear node, Gain R / G / B (Offset {offset}, Gamma {gamma} as on the hero), matched to the hero as graded.")
    for r in rows:
        g = r["gain"]
        if g is None:
            print(f"  {r['clip'][:40]:40s}  skipped: {r['note']}")
            continue
        note = "  (hero)" if r["note"] == "hero" else (f"  {r['note']}" if r["note"] else "")
        print(f"  {r['clip'][:40]:40s}  {g[0]:.2f}  {g[1]:.2f}  {g[2]:.2f}   {np.log2(luma(g)):+.2f} stops{note}")


# ─── Resolve ────────────────────────────────────────────────────

def gamma_wheel(power: float) -> float:
    """Gamma wheel value of a Linear node from its exponent (calibrated on 21.1: pivot at
    linear 1.0, exponent 1 / (1 + 4 g) for g > 0 and 1 - 4 g below)."""
    return (1 / power - 1) / 4 if power <= 1 else (1 - power) / 4


OFFSET_STEP = 0.0073   # linear added per Offset wheel unit away from 25.00 (calibrated on 21.1)
OFFSET_USED = 0.25     # wheel units: below this, and without a clear gain in fit, Offset is untouched


def fit_node(x: np.ndarray, y: np.ndarray, offset: float):
    """Gain and Gamma exponent for on = (off * gain + k) ** power with a given Offset (wheel
    units), fitted in log on level medians x, y. Returns (rms, worst, gain, power) in stops."""
    k, gain = (offset - 25) * OFFSET_STEP, 1.0
    for _ in range(4):   # k sits inside the power, next to the Gain: refine both in turn
        lx = np.log2(np.maximum(x + k / gain, 1e-6))
        (power, lift), *_ = np.linalg.lstsq(np.c_[lx, np.ones_like(lx)], np.log2(y), rcond=None)
        gain = 2.0 ** (lift / power)
    error = np.log2(y) - (power * lx + lift)
    return float(np.sqrt(np.mean(error ** 2))), float(np.abs(error).max()), gain, power


def measure_node(off: np.ndarray, on: np.ndarray) -> dict | None:
    """Gain, Gamma and Offset of a Linear node, from one frame exported with only that node
    off and on (DI stills). Per channel, pixels are grouped in 40 levels and on =
    (off * gain + (offset - 25) * OFFSET_STEP) ** power is fitted to the group medians. The
    Offset is only kept when it is clearly there: one wheel unit is 0.0073 linear, so an
    untouched wheel would otherwise read anywhere from 24.9 to 25.3. Misfit is the worst level
    left unexplained, in stops: curves or saturation in the node show there."""
    a = off[::ANALYSIS_STEP, ::ANALYSIS_STEP].reshape(-1, 3)
    b = on[::ANALYSIS_STEP, ::ANALYSIS_STEP].reshape(-1, 3)
    ok = ((a < a.max(0) - CLIP_MARGIN_DI).all(1) & (b < b.max(0) - CLIP_MARGIN_DI).all(1)
          & (a.min(1) > 0.01) & (b.min(1) > 0.01))
    if ok.sum() < 1000:
        return None
    la, lb = decode_intermediate(a[ok]), np.maximum(decode_intermediate(b[ok]), 1e-6)
    gain, power, offset, misfit = [], [], [], 0.0
    for c in range(3):
        level = np.clip(np.searchsorted(np.quantile(la[:, c], np.linspace(0, 1, 41)), la[:, c]) - 1, 0, 39)
        x = np.array([np.median(la[level == i, c]) for i in range(40)])
        y = np.array([np.median(lb[level == i, c]) for i in range(40)])
        plain = fit_node(x, y, 25.0)
        best_offset, best = min(((o, fit_node(x, y, o)) for o in np.arange(20, 30.001, 0.05)),
                                key=lambda t: t[1][0])
        if abs(best_offset - 25) < OFFSET_USED or plain[0] - best[0] < 0.005:
            best_offset, best = 25.0, plain
        offset.append(round(float(best_offset), 2))
        gain.append(best[2])
        power.append(best[3])
        misfit = max(misfit, best[1])
    return {"gain": [round(float(g), 4) for g in gain],
            "gamma": [round(gamma_wheel(p), 3) for p in power],
            "power": [round(float(p), 4) for p in power],
            "offset": offset, "offset_used": any(o != 25.0 for o in offset),
            "exposure_stops": round(float(np.median(np.log2(luma(lb) / np.maximum(luma(la), 1e-6)))), 3),
            "misfit_stops": round(misfit, 3)}


def node_lift(offset) -> np.ndarray:
    """Linear added by Offset wheels (25.00 = none)."""
    return ((np.full(3, 25.0) if offset is None else np.asarray(offset, float)) - 25.0) * OFFSET_STEP


def output_exposure(encoded: np.ndarray, gain, power=None, offset=None) -> float:
    """Median exposure change, in stops, of a Linear node with this Gain, Gamma exponent and Offset."""
    small = encoded[::ANALYSIS_STEP, ::ANALYSIS_STEP].reshape(-1, 3)
    lin = decode_intermediate(small[(small < small.max(0) - CLIP_MARGIN_DI).all(1) & (small.min(1) > 2 * DI_LOG_CUT)])
    out = np.maximum(lin * np.asarray(gain) + node_lift(offset), 1e-6) ** (1 if power is None else np.asarray(power))
    return float(np.median(np.log2(np.maximum(luma(out), 1e-6) / np.maximum(luma(lin), 1e-6))))


def run_bench(folders: list[Path]) -> int:
    """Solver against hand matches saved by Shot Match Collect (stills/ + truth.json)."""
    errors = []
    for folder in folders:
        truth = json.loads((folder / "truth.json").read_text())
        hero = truth["hero"]
        stills = [(name, folder / "stills" / f"{name}.tif") for name in truth["clips"]]
        # Shots are matched to the hero as graded, as in Resolve
        stills = [(n, folder / "hero_graded.tif" if n == hero else p) for n, p in stills]
        rows = match_stills(stills, hero, folder / "_bench", truth["clips"][hero])
        print(f"\n  {folder.name}  (hero {hero})\n  clip                                      solver   hand    error")
        for r in rows:
            if r["note"] == "hero" or r["gain"] is None:
                continue
            hand = truth["clips"][r["clip"]]
            # Compared on what reaches the screen: median exposure change of the shot, the
            # solver's Gain under the hero's Gamma against what the hand node measured
            hand_gain, hand_ev = np.array(hand["gain"]), hand["exposure_stops"]
            ev = output_exposure(load_still(folder / "stills" / f"{r['clip']}.tif"), r["gain"],
                                 truth["clips"][hero].get("power"), truth["clips"][hero].get("offset"))
            wb = float(np.log2(r["gain"][2] / r["gain"][0]) - np.log2(hand_gain[2] / hand_gain[0]))
            errors.append((ev - hand_ev, wb))
            flag = "" if abs(ev - hand_ev) <= 1 / 3 else "   <-- over 1/3 stop"
            print(f"  {r['clip'][:40]:40s}  {ev:+.2f}   {hand_ev:+.2f}   {ev - hand_ev:+.2f}{flag}")
    if not errors:
        raise Stop("No shots to score.")
    ev_err, wb_err = np.array(errors).T
    print(f"\n  {len(errors)} shots. Exposure error RMS {np.sqrt(np.mean(ev_err ** 2)):.2f}, "
          f"max {np.abs(ev_err).max():.2f} stops, {int((np.abs(ev_err) <= 1 / 3).sum())} within 1/3 stop. "
          f"White balance (B/R) RMS {np.sqrt(np.mean(wb_err ** 2)):.2f} stops.")
    return 0


class Progress:
    """Prints how far a loop is and roughly how long is left, from the pace so far: at most
    every PROGRESS_EVERY seconds, and when done. Flushed, so the Console shows it live."""

    def __init__(self, label: str, total: int):
        self.label, self.total, self.start, self.last = label, total, time.time(), time.time()

    def step(self, done: int) -> None:
        now = time.time()
        if done < self.total and now - self.last < PROGRESS_EVERY:
            return
        self.last = now
        spent = now - self.start
        if done >= self.total:
            print(f"  {self.label}: {self.total}/{self.total} done in {clock(spent)}", flush=True)
        else:
            print(f"  {self.label}: {done}/{self.total}, about {clock(spent / done * (self.total - done))} left",
                  flush=True)


def clock(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60} min {seconds % 60:02d} s" if seconds >= 60 else f"{seconds} s"


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


@contextlib.contextmanager
def grade_switched_off(item, keep_linear: int | None = None):
    """Layers 3-4 and Post-Clip off for the duration, except the Linear node `keep_linear`
    when given. Yields the list of problems, which the caller reports."""
    problems = []
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
                    switched.append((where, graph, i))
                else:
                    problems.append(f"could not switch off {where} node {i}")
        yield problems
    finally:
        for where, graph, i in switched:
            # Seen once on 21.1: SetNodeEnabled(True) refused, so give Resolve a moment and retry
            for attempt in range(3):
                if graph.SetNodeEnabled(i, True):
                    break
                time.sleep(0.3)
            else:
                label = graph.GetNodeLabel(i) or "no label"
                problems.append(f"COULD NOT SWITCH {where} NODE {i} ({label}) BACK ON, switch it on by hand")


def export_ungraded_still(project, timeline, item, fps: int, path: Path, keep_linear: int | None = None,
                          extras: bool = False) -> list[str]:
    """Middle frame of the clip with layers 3-4 and Post-Clip off, except the Linear node
    `keep_linear` when given (the hero's, so shots match the hero as graded). Returns problems.
    Uses TimelineItem.ExportFrame (21.1.2: no playhead move; on Shark 2026-10-28 it gave the
    same 16-bit image and honoured switched-off nodes), else the playhead and a still."""
    export = getattr(item, "ExportFrame", None)   # Resolve answers None for unknown names
    with grade_switched_off(item, keep_linear) as problems:
        if callable(export):
            offset = (int(item.GetEnd()) - int(item.GetStart())) // 2
            result = export(offset, {"outputType": "file", "filePath": str(path)}) or {}
            if result.get("success") and path.exists():
                # More frames of the clip, smaller, so one odd moment weighs less (ExportFrame only)
                length = int(item.GetEnd()) - int(item.GetStart())
                for k, at in enumerate(EXTRA_FRAMES if extras else (), 1):
                    extra = path.with_name(f"{path.stem}__f{k}.tif")
                    export(int(length * at), {"outputType": "file", "filePath": str(extra),
                                              "maxDimension": EXTRA_SIZE})
                return problems
        mid = (int(item.GetStart()) + int(item.GetEnd())) // 2
        if not timeline.SetCurrentTimecode(frames_to_tc(mid, fps)):
            problems.append("could not move the playhead to the clip")
            return problems
        time.sleep(0.5)   # let the viewer re-render before grabbing it
        if not project.ExportCurrentFrameAsStill(str(path)) or not path.exists():
            problems.append("still export failed")
    return problems


def confirm_write(resolve, count: int, hero_name: str) -> bool:
    """Yes/Cancel window before Apply writes. The API cannot target a layer or tell which one
    the node editor shows (checked in the 21.1 and 21.1.2 DaVinciResolveScript.pyi), so ask. No window,
    no write."""
    fu = globals().get("fu") or resolve.Fusion()
    bmd = globals().get("bmd")   # the Scripts menu provides it, a terminal does not
    ui = fu.UIManager if fu else None
    if not ui or not bmd:
        print("\n  Could not open the confirm window, so nothing was written. "
              "Run Shot Match Apply from Workspace > Scripts.")
        return False
    disp = bmd.UIDispatcher(ui)
    text = (f"Write Gain to {count} clips? The hero, {hero_name}, is left as it is.\n\n"
            f"Resolve writes into the layer on screen. Check that layer {LINEAR_LAYER} is showing: "
            "a wrong layer is overwritten, with no undo.")
    button = {"Weight": 0, "MinimumSize": [88, 24], "MaximumSize": [88, 24]}
    # Weight 0 on the button row keeps the layout from stretching the buttons to fill the window
    win = disp.AddWindow({"ID": "ShotMatchConfirm", "WindowTitle": "Shot Match Apply",
                          "Geometry": [400, 300, 420, 140]},
                         ui.VGroup({"Spacing": 12}, [
                             ui.Label({"Text": text, "WordWrap": True, "Weight": 1}),
                             ui.HGroup({"Weight": 0, "Spacing": 8}, [
                                 ui.HGap(0, 1),
                                 ui.Button({"ID": "Cancel", "Text": "Cancel", **button}),
                                 ui.Button({"ID": "Write", "Text": "Write", **button})])]))
    answer = {"write": False}

    def close(ev, write=False):
        answer["write"] = write
        disp.ExitLoop()
    win.On.Write.Clicked = lambda ev: close(ev, True)
    win.On.Cancel.Clicked = close
    win.On.ShotMatchConfirm.Close = close
    win.Show()
    disp.RunLoop()
    win.Hide()
    return answer["write"]


def write_gains(rows: list[dict], targets: dict, out_dir: Path, hero_node: dict | None = None) -> list[str]:
    """Each shot's Linear node through SetCDL: the solved Gain as Slope, the hero's Offset as
    CDL Offset (linear) and its Gamma exponent as Power, so a shot gets the hero's Offset and
    Gamma with its own Gain. Seen on 21.1: Slope lands on Gain R/G/B one to one, and a CDL
    Offset does not move the Offset wheel (Resolve spreads it over other wheels), so which
    wheels show it is Resolve's business: verify_written() checks the result on the image.
    SetCDL replaces the node's wheels and has no layer argument: it writes into the layer the
    node editor is SHOWING (seen on 21.1: run from layer 4, it wrote node 1 of layer 4).
    Returns the clips written."""
    power, lift = node_parts(hero_node)
    written, failed = [], []
    for r in rows:
        if r["gain"] is None or r["note"] == "hero" or r["clip"] not in targets:
            continue
        item, node = targets[r["clip"]]
        g = r["gain"]
        cdl = {"NodeIndex": str(node), "Slope": f"{g[0]:.4f} {g[1]:.4f} {g[2]:.4f}",
               "Offset": f"{lift[0]:.5f} {lift[1]:.5f} {lift[2]:.5f}",
               "Power": f"{power[0]:.4f} {power[1]:.4f} {power[2]:.4f}", "Saturation": "1.0000"}
        (written if item.SetCDL(cdl) else failed).append(r["clip"])
    with open(out_dir / "written.txt", "w") as f:
        f.write("Written to the Linear node (layer 3) of:\n" + "\n".join(written) + "\n")
        if failed:
            f.write("\nSetCDL refused:\n" + "\n".join(failed) + "\n")
    print(f"\n  Written: {len(written)} clips." + (f" REFUSED: {', '.join(failed)}" if failed else ""))
    return written


def verify_written(project, timeline, fps: int, rows: list[dict], targets: dict, stills: dict,
                   hero_node: dict | None, still_dir: Path) -> None:
    """Exports each written shot with only its Linear node on and compares it with what the
    node should do to the Linear-off still already taken: Gain, then the hero's Offset and
    Gamma. A gap over 0.05 stop means Resolve did not put the values where intended."""
    power, lift = node_parts(hero_node)
    start_tc, gaps, bad = timeline.GetCurrentTimecode(), [], []
    try:
        for r in rows:
            if r["clip"] not in targets or r["clip"] not in stills or r["note"] == "hero" or r["gain"] is None:
                continue
            item, node = targets[r["clip"]]
            path = still_dir / f"_check_{r['clip']}.tif"
            export_ungraded_still(project, timeline, item, fps, path, node)
            if not path.exists():
                bad.append(f"{r['clip']}: could not export it to check")
                continue
            off = load_still(stills[r["clip"]])[::ANALYSIS_STEP, ::ANALYSIS_STEP].reshape(-1, 3)
            on = load_still(path)[::ANALYSIS_STEP, ::ANALYSIS_STEP].reshape(-1, 3)
            ok = ((off < off.max(0) - CLIP_MARGIN_DI).all(1) & (on < on.max(0) - CLIP_MARGIN_DI).all(1)
                  & (off.min(1) > 2 * DI_LOG_CUT))
            expected = np.maximum(decode_intermediate(off[ok]) * r["gain"] + lift, 1e-6) ** power
            got = np.maximum(decode_intermediate(on[ok]), 1e-6)
            gap = float(np.median(np.abs(np.log2(luma(got) / np.maximum(luma(expected), 1e-6)))))
            gaps.append(gap)
            if gap > 0.05:
                bad.append(f"{r['clip']}: {gap:.2f} stop from what was intended")
    finally:
        timeline.SetCurrentTimecode(start_tc)
    if bad:
        print("  CHECK FAILED, the written nodes do not do what was intended:\n    " + "\n    ".join(bad))
    elif gaps:
        print(f"  Checked {len(gaps)} shots on the image: the written nodes do what was intended "
              f"(within {max(gaps):.2f} stop).")


def clean_old_runs(now: datetime.datetime | None = None) -> None:
    """Delete runs older than KEEP_DAYS from OUT_ROOT. Only names this script creates
    (20261027_161135, collect_20261027_161135, collect_20261027_161135.zip) are touched."""
    if not OUT_ROOT.is_dir():
        return
    cutoff = (now or datetime.datetime.now()) - datetime.timedelta(days=KEEP_DAYS)
    removed, freed = 0, 0
    for entry in OUT_ROOT.iterdir():
        match = RUN_NAME.fullmatch(entry.name)
        if not match or datetime.datetime.strptime(match.group(2), "%Y%m%d_%H%M%S") >= cutoff:
            continue
        files = [entry] if entry.is_file() else [f for f in entry.rglob("*") if f.is_file()]
        size = sum(f.stat().st_size for f in files)
        try:
            entry.unlink() if entry.is_file() else shutil.rmtree(entry)
        except OSError:
            continue
        removed, freed = removed + 1, freed + size
    if removed:
        print(f"  Cleaned {OUT_ROOT}: {removed} runs older than {KEEP_DAYS} days deleted, {freed / 1e6:.0f} MB freed.")


def open_scene():
    """resolve, project, timeline, fps, hero, clips (hero included), or Stop."""
    resolve = get_resolve()
    project = resolve.GetProjectManager().GetCurrentProject()
    timeline = project.GetCurrentTimeline() if project else None
    if not timeline:
        raise Stop("Open a project and a timeline first, then run this again.")
    settings = timeline.GetSettings() or {}  # GetSetting(name) is deprecated in 21.1
    rate = str(settings.get("timelineFrameRate", ""))
    if str(settings.get("timelineDropFrameTimecode")) == "1" or "DF" in rate.upper():
        raise Stop("Drop-frame timelines are not supported yet.")
    try:
        fps = round(float(rate.split()[0]))
    except (IndexError, ValueError):
        raise Stop(f"Could not read the timeline frame rate (got {rate!r}).")
    hero = timeline.GetCurrentVideoItem()
    if not hero:
        raise Stop("Park the playhead on the hero clip, then run this again.")
    clips = selected_clips(timeline)
    if hero.GetUniqueId() not in {c.GetUniqueId() for c in clips}:
        clips.append(hero)
    if len(clips) < 2:
        raise Stop("Select the clips of the scene (the hero can be one of them), then run this again.")
    return resolve, project, timeline, fps, hero, clips


def run_in_resolve(apply: bool = False) -> int:
    resolve, project, timeline, fps, hero, clips = open_scene()
    clean_old_runs()
    resolve.OpenPage("color")
    start_tc = timeline.GetCurrentTimecode()
    out_dir = OUT_ROOT / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    still_dir = out_dir / "stills"
    still_dir.mkdir(parents=True, exist_ok=True)
    if apply:
        print(f"Shot match APPLY: {len(clips)} clips, hero {hero.GetName()}. "
              "Gain will be written into each Linear node, the hero's is left as it is.\n"
              f"  It writes into the layer the node editor shows: layer {LINEAR_LAYER} must be on screen.\n"
              "  A window asks you to confirm before anything is written.")
    else:
        print(f"Shot match preview: {len(clips)} clips, hero {hero.GetName()}. Nothing is written to the grade.")

    stills, report, targets = [], [], {}
    try:
        progress = Progress("Capture", len(clips))
        for n, item in enumerate(sorted(clips, key=lambda c: c.GetStart()), 1):
            if n > 1:
                progress.step(n - 1)
            name = f"{n:02d}_{item.GetName()}"
            node = linear_node(item)
            if node is None:
                report.append(f"  {name}: skipped, needs exactly one node labelled Linear in layer {LINEAR_LAYER}")
                continue
            path = still_dir / f"{name}.tif"
            is_hero = item.GetUniqueId() == hero.GetUniqueId()
            problems = export_ungraded_still(project, timeline, item, fps, path, node if is_hero else None,
                                             extras=True)
            report.append(f"  {name}: Linear is node {node} of layer {LINEAR_LAYER}"
                          + (f"; {'; '.join(problems)}" if problems else ""))
            if path.exists():
                stills.append((name, path, is_hero))
                targets[name] = (item, node)
        progress.step(len(clips))
    finally:
        timeline.SetCurrentTimecode(start_tc)
    print("\n".join(report))

    try:
        heroes = [s for s in stills if s[2]]
        if not heroes:
            raise Stop("No still of the hero, so nothing to match against. See the lines above.")
        sample = tifffile.imread(str(heroes[0][1]))
        extra = [tifffile.imread(str(x)) for x in extra_paths(heroes[0][1])]
        extra_format = (f"; {len(extra)} more frames a clip, {extra[0].shape[1]}x{extra[0].shape[0]}, "
                        f"{extra[0].dtype}" if extra else "")
        print(f"\n  Still format: {sample.shape[1]}x{sample.shape[0]}, {sample.dtype}{extra_format}")
        hero_node = measure_hero_node(project, timeline, hero, fps, targets[heroes[0][0]][1],
                                      load_still(heroes[0][1]), still_dir / "_hero_off.tif")
        rows = match_stills([(s[0], s[1]) for s in stills], heroes[0][0], out_dir, hero_node)
        print_table(rows, hero_node)
        if apply:
            count = sum(1 for r in rows if r["gain"] is not None and r["note"] != "hero" and r["clip"] in targets)
            if confirm_write(resolve, count, hero.GetName()):
                write_gains(rows, targets, out_dir, hero_node)
                verify_written(project, timeline, fps, rows, targets,
                               {name: path for name, path, _ in stills}, hero_node, still_dir)
            else:
                print("\n  Cancelled: nothing written.")
    finally:
        shutil.rmtree(still_dir, ignore_errors=True)   # 12 MB a clip; the contact sheet shows them
    print(f"\n  Values and contact sheet: {out_dir}")
    return 0


def measure_hero_node(project, timeline, hero, fps: int, node: int, graded: np.ndarray, path: Path) -> dict | None:
    """Gain and Gamma of the hero's Linear node: one more export of the hero with it off."""
    start_tc = timeline.GetCurrentTimecode()
    try:
        problems = export_ungraded_still(project, timeline, hero, fps, path)
    finally:
        timeline.SetCurrentTimecode(start_tc)
    m = measure_node(load_still(path), graded) if path.exists() else None
    if m is None:
        print("  Hero's Linear node: could not be measured" + (f" ({'; '.join(problems)})" if problems else "")
              + ". Shots are matched without its Gamma.")
        return None
    if m["misfit_stops"] > 0.1:
        print("  Hero's Linear node does more than Gain, Gamma and Offset (curves?): values are approximate.")
    return m


def run_collect() -> int:
    """Measure the Linear node of a scene matched by hand. Writes nothing to the grade."""
    resolve, project, timeline, fps, hero, clips = open_scene()
    clean_old_runs()
    resolve.OpenPage("color")
    start_tc = timeline.GetCurrentTimecode()
    out_dir = OUT_ROOT / ("collect_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    still_dir, graded_dir = out_dir / "stills", out_dir / "_graded"
    still_dir.mkdir(parents=True, exist_ok=True)
    graded_dir.mkdir()
    print(f"Shot match COLLECT: {len(clips)} clips, hero {hero.GetName()}. "
          "Measures your Linear nodes; nothing is written to the grade.")
    truth, report, hero_name = {}, [], None
    try:
        progress = Progress("Capture", len(clips))
        for n, item in enumerate(sorted(clips, key=lambda c: c.GetStart()), 1):
            if n > 1:
                progress.step(n - 1)
            name = f"{n:02d}_{item.GetName()}"
            node = linear_node(item)
            if node is None:
                report.append(f"  {name}: skipped, needs exactly one node labelled Linear in layer {LINEAR_LAYER}")
                continue
            off, on = still_dir / f"{name}.tif", graded_dir / f"{name}.tif"
            is_hero = item.GetUniqueId() == hero.GetUniqueId()
            problems = (export_ungraded_still(project, timeline, item, fps, off, extras=True)
                        + export_ungraded_still(project, timeline, item, fps, on, node, extras=is_hero))
            m = measure_node(load_still(off), load_still(on)) if off.exists() and on.exists() else None
            if m is None:
                problems.append("could not measure the Linear node")
            else:
                truth[name] = m
                if is_hero:
                    hero_name = name
                    on.replace(out_dir / "hero_graded.tif")
                    for k, extra in enumerate(extra_paths(on), 1):
                        extra.replace(out_dir / f"hero_graded__f{k}.tif")
                if m["misfit_stops"] > 0.1:
                    problems.append("the node does more than Gain, Gamma and Offset (curves?): values are approximate")
            report.append(f"  {name}: " + ("; ".join(problems) if problems else "measured"))
        progress.step(len(clips))
    finally:
        timeline.SetCurrentTimecode(start_tc)
        for leftover in graded_dir.glob("*.tif"):
            leftover.unlink()
        graded_dir.rmdir()
    print("\n".join(report))
    if hero_name is None:
        raise Stop("The hero could not be measured, so nothing was saved. See the lines above.")
    (out_dir / "truth.json").write_text(json.dumps({"hero": hero_name, "clips": truth}, indent=1))
    print("\n  Your Linear node as measured:  Gain R / G / B    Gamma R / G / B    Offset R / G / B")
    for name, m in truth.items():
        g, y, o = m["gain"], m["gamma"], m["offset"]
        offset = f"{o[0]:.2f} {o[1]:.2f} {o[2]:.2f}" if m["offset_used"] else "not used"
        print(f"  {name[:40]:40s}  {g[0]:.2f}  {g[1]:.2f}  {g[2]:.2f}    {y[0]:+.2f} {y[1]:+.2f} {y[2]:+.2f}    {offset}"
              + ("  (hero)" if name == hero_name else ""))
    archive = Path(shutil.make_archive(str(out_dir), "zip", out_dir.parent, out_dir.name))
    shutil.rmtree(out_dir, ignore_errors=True)   # the zip holds it all
    print(f"\n  Saved: {archive}\n  Send that zip: it is one more scene to tune the solver on.")
    show_in_file_browser(archive)
    return 0


def show_in_file_browser(path: Path) -> None:
    """Explorer (or Finder) opened on the folder, with the file selected. Best effort."""
    try:
        if sys.platform == "win32":
            subprocess.Popen(["explorer", f"/select,{path}"])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(path)])
    except OSError:
        pass


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
    parser.add_argument("--bench", type=Path, nargs="+", help="score the solver on folders saved by Collect")
    args = parser.parse_args(argv)
    try:
        if args.stills:
            return run_offline(args.stills)
        if args.bench:
            return run_bench(args.bench)
        # The same file does three jobs, picked by the name it is saved under in the Scripts
        # folder. The Scripts menu does not set __file__, but the code object knows its file.
        stem = Path(main.__code__.co_filename).stem.lower()
        if "collect" in stem:
            return run_collect()
        return run_in_resolve(apply="apply" in stem)
    except Stop as stop:
        print(stop)
        return 2


if __name__ == "__main__":
    if "resolve" in globals():
        main([])   # inside Resolve: sys.exit would only print a traceback in the console
    else:
        sys.exit(main())
