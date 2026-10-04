#!/usr/bin/env python3
"""QC of graded renders against their sources.

Read-only: it never writes into the render or source folders, and never touches
Resolve. Needs ffprobe and ffmpeg on the PATH. Standard library only.

Checks, per render:
  inventory   every source (or EDL clip) has a render with the same name
  codec       codec and profile match the deliverable spec
  format      resolution and frame rate match the spec, or the source
  handles     the render covers each EDL event plus the handles, inside the source media
  frames      black frames and one-frame flashes inside the clip
  levels      luma outside video (legal) range
  audio       an audio stream is present, when required

Exit code: 0 = no FAIL, 1 = at least one FAIL, 2 = the check itself could not run.
A check that crashes is reported as a FAIL, never as a pass.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

MEDIA_EXT = {".mov", ".mxf", ".mp4", ".m4v"}
SOURCE_EXT = MEDIA_EXT | {".r3d", ".braw", ".ari", ".arx", ".crm", ".dng", ".nev"}

# codec spec -> (ffprobe codec_name, accepted ffprobe profiles)
CODECS = {
    "prores4444": ("prores", {"4444"}),
    "prores4444xq": ("prores", {"4444XQ"}),
    "prores422hq": ("prores", {"HQ"}),
    "dnxhr444": ("dnxhd", {"DNXHR 444"}),
    "dnxhrhqx": ("dnxhd", {"DNXHR HQX"}),
    "dnxhrhq": ("dnxhd", {"DNXHR HQ"}),
}

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
ORDER = {PASS: 0, WARN: 1, FAIL: 2}

SCENE_CUT = 0.30      # scene score above which two frames are "different shots"
FLASH_MAX = 3         # a black run this short, mid-clip, is a flash frame
LEVEL_TOL_8BIT = 4    # tolerance outside 16-235, in 8-bit code values


# ─── probing ────────────────────────────────────────────────────

def run(cmd: list[str]) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {out.stderr.strip()[:300]}")
    return out.stdout


def probe(path: Path) -> dict | None:
    """Return the facts QC needs, or None if ffprobe cannot read the file (R3D, BRAW)."""
    try:
        data = json.loads(run(["ffprobe", "-v", "error", "-show_streams",
                               "-show_format", "-of", "json", str(path)]))
    except (RuntimeError, json.JSONDecodeError):
        return None
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        return None
    fps = Fraction(video.get("r_frame_rate", "0/1"))
    tc = video.get("tags", {}).get("timecode")
    for s in data.get("streams", []):
        tc = tc or s.get("tags", {}).get("timecode")
    tc = tc or data.get("format", {}).get("tags", {}).get("timecode")
    frames = video.get("nb_frames")
    if frames and str(frames).isdigit() and int(frames) > 0:
        frames = int(frames)
    else:
        dur = float(video.get("duration") or data.get("format", {}).get("duration") or 0)
        frames = round(dur * fps) if fps else 0
    depth = re.search(r"p(\d+)", video.get("pix_fmt", ""))
    return {
        "codec": video.get("codec_name"),
        "profile": video.get("profile"),
        "width": video.get("width"),
        "height": video.get("height"),
        "fps": fps,
        "frames": frames,
        "timecode": tc,
        "bitdepth": int(depth.group(1)) if depth else 8,
        "audio": any(s.get("codec_type") == "audio" for s in data.get("streams", [])),
    }


# ─── timecode and EDL ───────────────────────────────────────────

def tc_to_frames(tc: str, fps: Fraction) -> int:
    """Non-drop-frame timecode to a frame count at the nominal rate (23.976 counts at 24)."""
    h, m, s, f = (int(x) for x in re.split(r"[:;.]", tc))
    base = round(fps)
    return ((h * 60 + m) * 60 + s) * base + f


def frames_to_tc(n: int, fps: Fraction) -> str:
    base = round(fps)
    f = n % base
    s = n // base
    return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}:{f:02d}"


def parse_edl(path: Path) -> dict[str, list[tuple[str, str]]]:
    """CMX3600 -> {clip stem (lowercase): [(source in, source out), ...]}.

    The clip is named by the "* FROM CLIP NAME:" comment when present, else by the reel.
    """
    event_re = re.compile(
        r"^\d{3,}\s+(\S+)\s+\S+\s+\S+\s+(?:\d+\s+)?"
        r"(\d\d:\d\d:\d\d[:;]\d\d)\s+(\d\d:\d\d:\d\d[:;]\d\d)\s+"
        r"\d\d:\d\d:\d\d[:;]\d\d\s+\d\d:\d\d:\d\d[:;]\d\d")
    events: list[dict] = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        m = event_re.match(line)
        if m:
            reel, src_in, src_out = m.groups()
            events.append({"name": reel, "in": src_in, "out": src_out, "named": False})
            continue
        c = re.match(r"^\*\s*FROM CLIP NAME:\s*(.+)$", line, re.I)
        if c and events and not events[-1]["named"]:
            events[-1]["name"] = Path(c.group(1).strip()).stem
            events[-1]["named"] = True
    out: dict[str, list[tuple[str, str]]] = {}
    for e in events:
        if not e["named"] and e["name"].upper() in {"BL", "BLK", "BLACK"}:
            continue
        out.setdefault(e["name"].lower(), []).append((e["in"], e["out"]))
    return out


# ─── frame analysis ─────────────────────────────────────────────

def analyse_frames(path: Path) -> list[dict]:
    """One decode pass: per-frame luma stats and scene-change score."""
    text = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
                "-map", "0:v:0", "-vf",
                "signalstats,select=gte(scene\\,0),metadata=mode=print:file=-",
                "-f", "null", "-"])
    frames: list[dict] = []
    for line in text.splitlines():
        if line.startswith("frame:"):
            frames.append({})
        elif "=" in line and frames:
            key, _, val = line.partition("=")
            key = key.rsplit(".", 1)[-1]
            try:
                frames[-1][key] = float(val)
            except ValueError:
                pass
    return frames


def legal_range(bitdepth: int) -> tuple[float, float, float]:
    scale = 2 ** (bitdepth - 8)
    return 16 * scale, 235 * scale, LEVEL_TOL_8BIT * scale


def black_runs(frames: list[dict], bitdepth: int) -> list[tuple[int, int]]:
    lo, hi, _ = legal_range(bitdepth)
    span = hi - lo
    runs, start = [], None
    for i, f in enumerate(frames):
        black = f.get("YMAX", hi) <= lo + 0.10 * span and f.get("YAVG", hi) <= lo + 0.03 * span
        if black and start is None:
            start = i
        if not black and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(frames) - 1))
    return runs


def flash_frames(frames: list[dict]) -> list[int]:
    """A frame that differs sharply from both neighbours: a one-frame insert."""
    s = [f.get("scene_score", 0.0) for f in frames]
    return [i for i in range(1, len(s) - 1) if s[i] > SCENE_CUT and s[i + 1] > SCENE_CUT]


# ─── checks ─────────────────────────────────────────────────────

class Result:
    def __init__(self, name: str):
        self.name = name
        self.items: list[tuple[str, str, str]] = []   # (check, status, detail)

    def add(self, check: str, status: str, detail: str) -> None:
        self.items.append((check, status, detail))

    @property
    def status(self) -> str:
        return max((s for _, s, _ in self.items), key=ORDER.get, default=PASS)


def check_render(render: Path, info: dict, src: dict | None, src_name: str | None,
                 edl_ranges: list[tuple[str, str]] | None, args) -> Result:
    r = Result(render.name)

    # codec
    if args.codec != "any":
        want_codec, want_profiles = CODECS[args.codec]
        ok = info["codec"] == want_codec and info["profile"] in want_profiles
        r.add("codec", PASS if ok else FAIL,
              f"{info['codec']} {info['profile']}" + ("" if ok else f", expected {args.codec}"))

    # resolution
    if args.resolution:
        want = tuple(int(x) for x in args.resolution.lower().split("x"))
    elif src:
        want = (src["width"], src["height"])
    else:
        want = None
    got = (info["width"], info["height"])
    if want:
        r.add("resolution", PASS if got == want else FAIL,
              f"{got[0]}x{got[1]}" + ("" if got == want else f", expected {want[0]}x{want[1]}"))
    else:
        r.add("resolution", WARN, f"{got[0]}x{got[1]}, nothing to compare against")

    # frame rate
    want_fps = Fraction(args.fps).limit_denominator(1001) if args.fps else (src["fps"] if src else None)
    if want_fps:
        ok = abs(float(info["fps"]) - float(want_fps)) < 0.001
        r.add("fps", PASS if ok else FAIL,
              f"{float(info['fps']):.3f}" + ("" if ok else f", expected {float(want_fps):.3f}"))
    else:
        r.add("fps", WARN, f"{float(info['fps']):.3f}, nothing to compare against")

    # handles and source bounds
    fps = info["fps"]
    if not info["timecode"]:
        r.add("handles", WARN, "render has no timecode, handles not checked")
    elif src and src["fps"] != fps:
        r.add("handles", FAIL, "frame rate differs from source, handles cannot be trusted")
    else:
        rs = tc_to_frames(info["timecode"], fps)
        re_ = rs + info["frames"]               # exclusive
        span = f"{info['timecode']} to {frames_to_tc(re_, fps)} ({info['frames']} fr)"
        ss = se = None
        if src and src["timecode"]:
            ss = tc_to_frames(src["timecode"], fps)
            se = ss + src["frames"]
            if rs < ss or re_ > se:
                r.add("handles", FAIL, f"{span} runs outside the source media "
                                       f"{src['timecode']} to {frames_to_tc(se, fps)}")
                edl_ranges = None
        if edl_ranges:
            need_s = min(tc_to_frames(a, fps) for a, _ in edl_ranges) - args.handles
            need_e = max(tc_to_frames(b, fps) for _, b in edl_ranges) + args.handles
            if ss is not None:
                need_s, need_e = max(need_s, ss), min(need_e, se)
            head, tail = rs - need_s, need_e - re_
            if head > 0 or tail > 0:
                miss = []
                if head > 0:
                    miss.append(f"head short by {head} fr")
                if tail > 0:
                    miss.append(f"tail short by {tail} fr")
                # Without the source we cannot tell a short render from short media.
                r.add("handles", FAIL if ss is not None else WARN,
                      f"{span}: {', '.join(miss)} for {args.handles}-frame handles"
                      + ("" if ss is not None else ", source unreadable so media length unknown"))
            else:
                r.add("handles", PASS, f"{span}, {args.handles}-frame handles covered")
        elif not any(c == "handles" for c, _, _ in r.items):
            r.add("handles", WARN, f"{span}, no EDL event for this clip, handles not checked")

    # frames and levels, one decode
    if not args.skip_frames:
        frames = analyse_frames(render)
        depth = int(frames[0].get("YBITDEPTH", info["bitdepth"])) if frames else info["bitdepth"]
        last = len(frames) - 1
        bad = []
        for a, b in black_runs(frames, depth):
            length = b - a + 1
            inside = a > 0 and b < last
            if inside and length <= FLASH_MAX:
                bad.append((FAIL, f"black frame at {a}" + (f"-{b}" if b > a else "")))
            else:
                bad.append((WARN, f"black run frames {a}-{b} ({length} fr)"
                                  + ("" if inside else ", at head or tail, check for a fade")))
        black_frames = {i for a, b in black_runs(frames, depth) for i in range(a, b + 1)}
        for i in flash_frames(frames):
            if i not in black_frames:
                bad.append((WARN, f"possible flash frame at {i}"))
        if bad:
            for status, detail in bad:
                r.add("frames", status, detail)
        else:
            r.add("frames", PASS, f"{len(frames)} frames, no black or flash frame")

        lo, hi, tol = legal_range(depth)
        out = [i for i, f in enumerate(frames)
               if f.get("YMIN", lo) < lo - tol or f.get("YMAX", hi) > hi + tol]
        if out:
            ymin = min(f.get("YMIN", lo) for f in frames)
            ymax = max(f.get("YMAX", hi) for f in frames)
            r.add("levels", WARN, f"{len(out)} of {len(frames)} frames outside video range "
                                  f"(Y {ymin:.0f}-{ymax:.0f}, legal {lo:.0f}-{hi:.0f} at {depth}-bit)")
        else:
            r.add("levels", PASS, f"inside video range at {depth}-bit")

    # audio
    if args.audio:
        r.add("audio", PASS if info["audio"] else FAIL,
              "audio present" if info["audio"] else "no audio stream, audio required")

    if src is None and src_name:
        r.add("source", WARN, f"source {src_name} unreadable by ffprobe, compared to spec only")
    return r


# ─── main ───────────────────────────────────────────────────────

def media_files(folder: Path, exts: set[str]) -> dict[str, Path]:
    return {p.stem.lower(): p for p in sorted(folder.rglob("*"))
            if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("._")}


def write_reports(results: list[Result], inventory: list[tuple[str, str, str]],
                  args, out_dir: Path) -> tuple[Path, Path | None]:
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    overall = max([r.status for r in results] + [s for _, s, _ in inventory],
                  key=ORDER.get, default=PASS)
    n = {k: sum(1 for r in results if r.status == k) for k in (PASS, WARN, FAIL)}
    lines = [f"# QC : {args.renders.name}", "",
             f"Run : {stamp.replace('_', ' ')}",
             f"Renders : {args.renders}",
             f"Sources : {args.sources or 'none given'}",
             f"EDL : {args.edl or 'none given'}",
             f"Spec : {args.codec}, {args.resolution or 'source resolution'}, "
             f"{args.fps or 'source fps'}, {args.handles}-frame handles"
             + (", audio required" if args.audio else ""),
             "", f"**Result : {overall}**  ({n[PASS]} pass, {n[WARN]} warn, {n[FAIL]} fail)", ""]
    if inventory:
        lines += ["## Inventory", ""] + [f"- {s} · {name} · {d}" for name, s, d in inventory] + [""]
    lines += ["## Clips", ""]
    for r in sorted(results, key=lambda r: (-ORDER[r.status], r.name)):
        lines.append(f"### {r.status} · {r.name}")
        lines += [f"- {s} · {c} · {d}" for c, s, d in r.items] + [""]
    report = out_dir / f"QC_{args.renders.name}_{stamp}.md"
    report.write_text("\n".join(lines), encoding="utf-8")

    summary = None
    if overall != FAIL and results:
        first = results[0]
        spec = next((d for c, _, d in first.items if c == "codec"), args.codec)
        res = next((d for c, _, d in first.items if c == "resolution"), "")
        fps = next((d for c, _, d in first.items if c == "fps"), "")
        summary = out_dir / f"QC_{args.renders.name}_{stamp}_client.md"
        summary.write_text("\n".join([
            f"QC : {args.renders.name}", "",
            f"Clips delivered : {len(results)}",
            f"Format : {spec}, {res.split(',')[0]}, {fps.split(',')[0]} fps",
            f"Handles : {args.handles} frames",
            "Checked : clip count and names against the edit, codec, resolution, frame rate, "
            "handles, black and flash frames, video levels.",
        ]) + "\n", encoding="utf-8")
    return report, summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("renders", type=Path, help="folder of graded renders")
    ap.add_argument("--sources", type=Path, help="folder of original camera media")
    ap.add_argument("--edl", type=Path, help="CMX3600 EDL of the cut, for handle coverage")
    ap.add_argument("--codec", default="prores4444", choices=sorted(CODECS) + ["any"])
    ap.add_argument("--resolution", help="WxH, default: same as the source")
    ap.add_argument("--fps", help="e.g. 24 or 24000/1001, default: same as the source")
    ap.add_argument("--handles", type=int, default=24)
    ap.add_argument("--audio", action="store_true", help="fail renders without audio")
    ap.add_argument("--skip-frames", action="store_true", help="skip the decode pass")
    ap.add_argument("--out", type=Path, default=Path("qc_reports"))
    args = ap.parse_args(argv)

    for tool in ("ffprobe", "ffmpeg"):
        if not shutil.which(tool):
            print(f"QC could not run: {tool} is not on the PATH.", file=sys.stderr)
            return 2
    if not args.renders.is_dir():
        print(f"QC could not run: {args.renders} is not a folder.", file=sys.stderr)
        return 2

    renders = media_files(args.renders, MEDIA_EXT)
    sources = media_files(args.sources, SOURCE_EXT) if args.sources else {}
    edl = parse_edl(args.edl) if args.edl else {}
    expected = set(edl) if edl else set(sources)

    inventory = []
    for name in sorted(expected - set(renders)):
        shown = sources[name].name if name in sources else name.upper()
        inventory.append((shown, FAIL, "no render with this name"))
    if expected:
        for name in sorted(set(renders) - expected):
            inventory.append((renders[name].name, WARN, "render matches no source or EDL clip"))

    results = []
    for stem, path in sorted(renders.items()):
        try:
            info = probe(path)
            if info is None:
                res = Result(path.name)
                res.add("probe", FAIL, "ffprobe cannot read this render")
            else:
                src_path = sources.get(stem)
                src = probe(src_path) if src_path else None
                res = check_render(path, info, src, src_path.name if src_path else None,
                                   edl.get(stem), args)
        except Exception as exc:          # a crashed check is a FAIL, never a pass
            res = Result(path.name)
            res.add("crash", FAIL, f"check crashed: {exc}")
        results.append(res)
        print(f"{res.status:4}  {path.name}")

    if not renders:
        inventory.append(("(renders)", FAIL, "no media files found in the render folder"))

    args.out.mkdir(parents=True, exist_ok=True)
    report, summary = write_reports(results, inventory, args, args.out)
    worst = max([r.status for r in results] + [s for _, s, _ in inventory],
                key=ORDER.get, default=PASS)
    print(f"\nResult : {worst}. Report : {report}")
    if summary:
        print(f"Client summary : {summary}")
    return 1 if worst == FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
