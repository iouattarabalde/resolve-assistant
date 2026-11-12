"""Read-only inventory of a Resolve timeline, and a scene map built from it without flags.

Stage 1 of the Color Assistant base: one pass lists every video clip with its camera, source
recording and timecode, then groups the clips into

    recording   clips cut from the same source file (they share one first-pass correction)
    camera      the camera that shot the recording
    scene       recordings whose shooting times touch or overlap, whatever the cut order

Nothing here writes to Resolve. `read_timeline` only calls getters; `build_scene_map` works on
the plain dict it returns, so it runs and is tested without Resolve.

    python timeline_inventory.py INVENTORY.json      # print the scene map of a saved inventory

Standard library only.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

SCENE_GAP_S = 600        # recordings further apart than this (seconds of shooting time) start a new scene
NEIGHBOURS = 2           # timeline neighbours each side that vote for a clip with no usable clock
CAMERA_KEYS = ("Camera #", "Camera ID", "Camera", "Camera Label")
DAY_KEYS = ("Date Recorded", "Shoot Date", "Date Created")


# ---------------------------------------------------------------- reading (inside Resolve)

def _call(obj, name, *args):
    """A getter's value, or None when this Resolve does not have it (missing methods are None)."""
    fn = getattr(obj, name, None)
    try:
        return fn(*args) if callable(fn) else None
    except Exception:
        return None


def read_clip(item, track: int, index: int) -> dict:
    media = _call(item, "GetMediaPoolItem")
    props = (_call(media, "GetClipProperty") or {}) if media else {}
    meta = (_call(media, "GetMetadata") or {}) if media else {}
    return {
        "track": track, "index": index,
        "name": _call(item, "GetName"),
        "id": _call(item, "GetUniqueId"),
        "record_start": _call(item, "GetStart"), "record_end": _call(item, "GetEnd"),
        "source_start": _call(item, "GetSourceStartFrame"), "source_end": _call(item, "GetSourceEndFrame"),
        "flags": list(_call(item, "GetFlagList") or []),
        "clip_color": _call(item, "GetClipColor"),
        "media_id": _call(media, "GetUniqueId") if media else None,
        "props": {k: str(v) for k, v in props.items() if v not in ("", None)},
        "metadata": {k: str(v) for k, v in meta.items() if v not in ("", None)},
    }


def read_timeline(timeline, first: int = 0, count: int | None = None) -> dict:
    """Every video clip of the timeline, tracks bottom-up then left to right. `first`/`count`
    slice the flat list, for callers that must read a long timeline in several short calls."""
    settings = _call(timeline, "GetSettings") or {}
    clips, seen = [], 0
    for track in range(1, (_call(timeline, "GetTrackCount", "video") or 0) + 1):
        for index, item in enumerate(_call(timeline, "GetItemListInTrack", "video", track) or []):
            if seen >= first and (count is None or len(clips) < count):
                clips.append(read_clip(item, track, index))
            seen += 1
    return {
        "timeline": _call(timeline, "GetName"),
        "frame_rate": str(settings.get("timelineFrameRate", "")),
        "drop_frame": str(settings.get("timelineDropFrameTimecode", "")),
        "start_frame": _call(timeline, "GetStartFrame"),
        "total": seen, "first": first, "clips": clips,
    }


# ---------------------------------------------------------------- grouping (no Resolve)

def tc_seconds(tc: str | None, fps: float) -> float | None:
    """'10:42:07:12' or '10:42:07;12' as seconds since midnight. Drop-frame is read as clock
    time, which is what it tracks; the non-drop drift (3.6 s an hour) does not matter here."""
    found = re.fullmatch(r"(\d+):(\d\d):(\d\d)[:;.](\d+)", (tc or "").strip())
    if not found or fps <= 0:
        return None
    h, m, s, f = map(int, found.groups())
    return h * 3600 + m * 60 + s + f / fps


def _fps(clip: dict, default: float) -> float:
    try:
        return float(str(clip["props"].get("FPS", "")).split()[0])
    except (IndexError, ValueError):
        return default


def camera_of(clip: dict) -> str:
    """Camera id: metadata first, then names like CAMA_0043 / CAM_B0090 / 'CAM A0121', then
    the reel's letters, then the letters a file name starts with (one family per camera
    model, which is what the input transform depends on), else '?'."""
    for key in CAMERA_KEYS:
        value = clip["metadata"].get(key) or clip["props"].get(key)
        if value:
            return value.strip().upper()
    name = (clip["props"].get("File Name") or clip.get("name") or "").upper()
    found = re.search(r"CAM[ _-]?([A-Z])(?=[ _\d-])", name)
    if found:
        return found.group(1)
    reel = re.match(r"[A-Za-z]+", clip["props"].get("Reel Name", ""))
    if reel:
        return reel.group(0).upper()
    family = re.match(r"[A-Za-z]+", name)   # C4025.MP4 -> C, DJI_2025... -> DJI, GX016564 -> GX
    return family.group(0) if family else "?"


def day_of(clip: dict) -> str:
    for key in DAY_KEYS:
        value = clip["metadata"].get(key) or clip["props"].get(key)
        found = re.search(r"\d{4}-\d\d-\d\d|\d\d/\d\d/\d{4}|\w{3} \w{3} +\d+ (?:[\d:]+ )?\d{4}", value or "")
        if found:
            text = found.group(0)
            return re.sub(r" [\d:]{5,8}(?= \d{4}$)", "", text)   # 'Thu Aug 6 12:52:41 2026' -> the date only
    return ""


def recording_of(clip: dict) -> str:
    return clip.get("media_id") or clip["props"].get("File Name") or clip.get("name") or f"clip{clip['track']}.{clip['index']}"


def recordings(inventory: dict) -> dict:
    """One row per source recording used in the timeline: camera, day, and its span of
    shooting time (seconds since midnight) when the media carries a time-of-day timecode."""
    try:
        default = float(inventory.get("frame_rate", "").split()[0])
    except (IndexError, ValueError):
        default = 24.0
    out: dict[str, dict] = {}
    for position, clip in enumerate(inventory["clips"]):
        key = recording_of(clip)
        fps = _fps(clip, default)
        start, end = tc_seconds(clip["props"].get("Start TC"), fps), tc_seconds(clip["props"].get("End TC"), fps)
        if start is not None and end is not None and end < start:
            end += 86400.0   # recording ran past midnight
        row = out.setdefault(key, {"recording": key, "name": clip["props"].get("File Name") or clip.get("name"),
                                   "camera": camera_of(clip), "day": day_of(clip),
                                   "start": start, "end": end if end is not None else start, "clips": []})
        row["clips"].append(position)
    return out


def has_clock(row: dict) -> bool:
    """False for media with no timecode, or record-run timecode that starts at zero."""
    return row["start"] is not None and row["start"] > 0


def build_scene_map(inventory: dict, scene_gap_s: float = SCENE_GAP_S) -> dict:
    """Scene number (from 1, in shooting order) for every recording and clip. Recordings with
    a clock are chained by shooting time, per day; the others take the vote of their timeline
    neighbours, or a scene of their own when no neighbour has one."""
    rows = recordings(inventory)
    timed = sorted((r for r in rows.values() if has_clock(r)), key=lambda r: (r["day"], r["start"]))
    scene, edge, day = 0, None, None
    for row in timed:
        if day != row["day"] or edge is None or row["start"] > edge + scene_gap_s:
            scene, edge, day = scene + 1, row["end"], row["day"]
        edge = max(edge, row["end"])
        row["scene"], row["placed_by"] = scene, "clock"

    clips = inventory["clips"]
    owner = {pos: row for row in rows.values() for pos in row["clips"]}
    by_track: dict[int, list[int]] = {}
    for pos, clip in enumerate(clips):
        by_track.setdefault(clip["track"], []).append(pos)
    for row in rows.values():
        if "scene" in row:
            continue
        votes: Counter = Counter()
        for pos in row["clips"]:
            lane = by_track[clips[pos]["track"]]
            at = lane.index(pos)
            for near in lane[max(0, at - NEIGHBOURS):at] + lane[at + 1:at + 1 + NEIGHBOURS]:
                if owner[near].get("placed_by") == "clock":
                    votes[owner[near]["scene"]] += 1
        if votes:
            row["scene"], row["placed_by"] = votes.most_common(1)[0][0], "neighbours"
        else:
            scene += 1
            row["scene"], row["placed_by"] = scene, "alone"

    return {
        "timeline": inventory.get("timeline"),
        "scene_gap_s": scene_gap_s,
        "recordings": sorted(rows.values(), key=lambda r: (r["scene"], r["camera"], r["start"] or 0)),
        "clips": [{"track": c["track"], "index": c["index"], "name": c.get("name"), "id": c.get("id"),
                   "recording": owner[pos]["recording"], "camera": owner[pos]["camera"],
                   "scene": owner[pos]["scene"], "placed_by": owner[pos]["placed_by"]}
                  for pos, c in enumerate(clips)],
    }


def summary(scene_map: dict) -> str:
    rows = scene_map["recordings"]
    lines = [f"{scene_map['timeline']}: {len(scene_map['clips'])} clips, {len(rows)} recordings, "
             f"{len({r['scene'] for r in rows})} scenes (gap {scene_map['scene_gap_s']:.0f} s)"]
    for how, n in Counter(r["placed_by"] for r in rows).most_common():
        lines.append(f"  placed by {how}: {n} recordings")
    for number in sorted({r["scene"] for r in rows}):
        inside = [r for r in rows if r["scene"] == number]
        cams = Counter(r["camera"] for r in inside)
        lines.append(f"  scene {number:3d}: {sum(len(r['clips']) for r in inside):4d} clips, "
                     + ", ".join(f"{cam} x{n}" for cam, n in sorted(cams.items())))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(__doc__)
        return 2
    print(summary(build_scene_map(json.loads(Path(args[0]).read_text(encoding="utf-8")))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
