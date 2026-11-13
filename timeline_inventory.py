"""Read-only inventory of a Resolve timeline, and a scene map built from it without flags.

Stage 1 of the Color Assistant base: one pass lists every video clip with its camera, source
recording and timecode, then groups the clips into

    recording   clips cut from the same source file
    setup       uses of one recording that sit close together in the source: these are the
                clips that share one first-pass correction (on a 1032-clip graded documentary,
                90 % agree within 1/3 stop at a 10 s gap, 80 % for whole recordings)
    camera      the camera that shot the recording
    scene       recordings whose shooting times touch or overlap, whatever the cut order

Nothing here writes to Resolve. `read_timeline` only calls getters; `build_scene_map` works on
the plain dict it returns, so it runs and is tested without Resolve.

    python timeline_inventory.py INVENTORY.json      # print the scene map of a saved inventory

Standard library only.
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

SCENE_GAP_S = 600        # recordings further apart than this (seconds of shooting time) start a new scene
SETUP_GAP_S = 10         # uses of one recording further apart than this in the source are separate setups
INTERCUT_MIN, INTERCUT_SHARE, INTERCUT_SURE = 3, 0.4, 0.75   # see _merge_by_cut
INTERCUT_CLOCK_S = 2700
CONTINUE_GAP_S, CONTINUE_NEAR = 1800, 20
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


def setups(inventory: dict, setup_gap_s: float = SETUP_GAP_S) -> dict[int, str]:
    """Setup id for every clip position: uses of one recording chained while the next one
    starts within `setup_gap_s` seconds of source time after the furthest end so far."""
    try:
        default = float(inventory.get("frame_rate", "").split()[0])
    except (IndexError, ValueError):
        default = 24.0
    uses: dict[str, list] = {}
    for position, clip in enumerate(inventory["clips"]):
        fps = _fps(clip, default)
        start = (clip.get("source_start") or 0) / fps
        end = (clip.get("source_end") or clip.get("source_start") or 0) / fps
        uses.setdefault(recording_of(clip), []).append((start, max(start, end), position))
    out: dict[int, str] = {}
    for key, spans in uses.items():
        part, edge = 0, None
        for start, end, position in sorted(spans):
            if edge is not None and start > edge + setup_gap_s:
                part += 1
            edge = end if edge is None else max(edge, end)
            out[position] = f"{key}#{part + 1}"
    return out


def has_clock(row: dict) -> bool:
    """False for media with no timecode, or record-run timecode that starts at zero."""
    return row["start"] is not None and row["start"] > 0


def _base_scenes(inventory: dict, rows: dict, scene_gap_s: float) -> None:
    """First pass: recordings with a clock chained by shooting time, per day; the others take
    the vote of their timeline neighbours, or a scene of their own when no neighbour has one."""
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
        if pos in owner:
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


def _merge_by_cut(rows: dict, order: list[int], owner: dict) -> list[tuple]:
    """Second pass, from the edit. Camera clocks disagree (a second body a day off, a clock
    reset to zero), and a shoot pauses, so the clock alone cuts one scene into several.
    Two scenes become one when
      - the cut alternates between them: at least INTERCUT_MIN changes from one to the other,
        making up INTERCUT_SHARE of the smaller scene's borders, and either their times of day
        are within INTERCUT_CLOCK_S (cameras a day apart) or nearly all its borders (INTERCUT_SURE);
      - or the same camera, the same day, carried on within CONTINUE_GAP_S and the two sit
        within CONTINUE_NEAR clips of each other in the cut.
    Then a scene left with one clip joins the larger of its neighbours in the cut."""
    scene_of = {pos: owner[pos]["scene"] for pos in order}
    parent = {s: s for s in set(scene_of.values())}

    def find(s):
        while parent[s] != s:
            s = parent[s]
        return s

    borders: Counter = Counter()
    for here, there in zip(order, order[1:]):
        if scene_of[here] != scene_of[there]:
            borders[frozenset((scene_of[here], scene_of[there]))] += 1
    total: Counter = Counter()
    for pair, n in borders.items():
        for s in pair:
            total[s] += n
    span: dict = {}        # scene -> [first, last] time of day
    cam_span: dict = {}    # (scene, camera, day) -> [first, last]
    places: dict = {}      # scene -> ranks in the cut
    rank = {pos: n for n, pos in enumerate(order)}
    for pos in order:
        row = owner[pos]
        places.setdefault(row["scene"], []).append(rank[pos])
        if row["placed_by"] == "clock":
            for key, table in ((row["scene"], span), ((row["scene"], row["camera"], row["day"]), cam_span)):
                lo, hi = table.setdefault(key, [row["start"], row["end"]])
                table[key] = [min(lo, row["start"]), max(hi, row["end"])]

    def apart(a, b):
        return max(a[0], b[0]) - min(a[1], b[1])

    merges = []
    for pair, n in borders.items():
        one, two = tuple(pair)
        small = min(one, two, key=lambda s: total[s])
        if n < INTERCUT_MIN or n < INTERCUT_SHARE * total[small]:
            continue
        close = one in span and two in span and apart(span[one], span[two]) <= INTERCUT_CLOCK_S
        if close or n >= INTERCUT_SURE * total[small]:
            parent[find(one)] = find(two)
            merges.append(("intercut", one, two, n))
    keys = sorted(cam_span)
    for i, (one, cam, day) in enumerate(keys):
        for two, cam2, day2 in keys[i + 1:]:
            if one == two or cam != cam2 or day != day2 or find(one) == find(two):
                continue
            gap = apart(cam_span[(one, cam, day)], cam_span[(two, cam2, day2)])
            if gap <= CONTINUE_GAP_S and min(abs(p - q) for p in places[one] for q in places[two]) <= CONTINUE_NEAR:
                parent[find(one)] = find(two)
                merges.append(("continues", one, two, round(gap / 60)))
    for row in rows.values():
        row["scene"] = find(row["scene"])

    for _ in range(3):   # a lone clip between two lone clips needs a second look
        size = Counter(owner[pos]["scene"] for pos in order)
        for n, pos in enumerate(order):
            row = owner[pos]
            if size[row["scene"]] != 1:
                continue
            before = owner[order[n - 1]]["scene"] if n else None
            after = owner[order[n + 1]]["scene"] if n + 1 < len(order) else None
            if before is None and after is None:
                continue
            pick = before if after is None or (before is not None and size[before] >= size[after]) else after
            merges.append(("single", row["scene"], pick, 1))
            row["scene"] = pick
    return merges


def build_scene_map(inventory: dict, scene_gap_s: float = SCENE_GAP_S, setup_gap_s: float = SETUP_GAP_S,
                    use_cut: bool = True) -> dict:
    """Scene number (from 1, in the order scenes first appear in the cut) for every recording
    and clip. Clips longer than half the timeline (reference renders) and items with no media
    (transitions, generators) get scene 0 and take no part."""
    clips = inventory["clips"]
    starts = [c.get("record_start") or 0 for c in clips]
    ends = [c.get("record_end") or 0 for c in clips]
    length = (max(ends) - min(starts)) if clips else 0
    aside = {pos for pos, c in enumerate(clips)
             if not c.get("media_id") and not c["props"]
             or (len(clips) > 2 and length and (ends[pos] - starts[pos]) > length / 2)}
    rows = recordings(inventory)
    kept = {key: row for key, row in rows.items() if not set(row["clips"]) & aside}
    _base_scenes(inventory, kept, scene_gap_s)
    owner = {pos: row for row in kept.values() for pos in row["clips"]}
    order = sorted(owner, key=lambda pos: (starts[pos], clips[pos]["track"]))
    merges = _merge_by_cut(kept, order, owner) if use_cut else []
    number: dict = {}
    for pos in order:
        number.setdefault(owner[pos]["scene"], len(number) + 1)
    for row in kept.values():
        row["scene"] = number[row["scene"]]
    for key, row in rows.items():
        if key not in kept:
            row["scene"], row["placed_by"] = 0, "aside"
    owner = {pos: row for row in rows.values() for pos in row["clips"]}
    setup = setups(inventory, setup_gap_s)
    return {
        "timeline": inventory.get("timeline"),
        "scene_gap_s": scene_gap_s, "setup_gap_s": setup_gap_s,
        "merges": [(why, number.get(a, a), number.get(b, b), n) for why, a, b, n in merges],
        "recordings": sorted(rows.values(), key=lambda r: (r["scene"], r["camera"], r["start"] or 0)),
        "clips": [{"track": c["track"], "index": c["index"], "name": c.get("name"), "id": c.get("id"),
                   "record_start": c.get("record_start") or 0,
                   "recording": owner[pos]["recording"], "setup": setup[pos], "camera": owner[pos]["camera"],
                   "scene": owner[pos]["scene"], "placed_by": owner[pos]["placed_by"]}
                  for pos, c in enumerate(clips)],
    }


CUT_SWITCH_COST = 1.5   # see smooth_by_cut
CUT_ISLAND_COST = 0.4
CUT_WEAK_COST = 0.01
CUT_WINDOW = 3
CUT_INTERCUT_COST = 4.0
LOOK_LUM_STOP = 1.0        # see split_by_look: one stop of median level is one unit of look distance
LOOK_WARM_UNIT = 0.5       # half a stop of red over blue is one unit
LOOK_PENALTY = 12.0        # cost of one more look segment inside a scene
LOOK_MIN_LEN = 6           # clips: a look segment shorter than this is not split off
LOOK_TOUCH_BONUS = 0.15    # cap_scenes with looks: each shared cut boundary (up to 3) counts this much
REFINE_SWITCH_COST = 2.0   # see refine_by_look
REFINE_CLOCK_COST = 0.6
MAIN_CAMERA_SHARE = 0.1


def smooth_by_cut(scene_map: dict, switch_cost: float = CUT_SWITCH_COST, island_cost: float = CUT_ISLAND_COST,
                  weak_cost: float = CUT_WEAK_COST, window: int = CUT_WINDOW,
                  intercut_cost: float = CUT_INTERCUT_COST) -> dict:
    """Neighbours in the cut over clocks, as a colourist groups a first pass: a scene is mostly a
    run of the edit. Along the cut, each clip keeps its scene unless changing it saves scene
    changes (each costs `switch_cost`):
      - clips of the main cameras (each at least MAIN_CAMERA_SHARE of the clips) placed by their
        clock cost 1 to move, or `island_cost` when no clip of their scene sits within `window`
        clips of them in the cut (a lone clip among another scene, not a back-and-forth intercut),
        and `intercut_cost` inside an A B A B intercut, so that one survives a high switch cost;
      - drone, action-camera, phone and stock clips, and clips placed without a clock, cost
        `weak_cost`: they go with the scene they are cut into.
    Solved for the whole cut at once (Viterbi). Sets "placed_by" to "cut" on the clips moved."""
    clips = scene_map["clips"]
    order = sorted((c for c in clips if c["scene"]), key=lambda c: (c["record_start"], c["track"]))
    if len(order) < 3:
        return scene_map
    labels = sorted({c["scene"] for c in order})
    index = {s: k for k, s in enumerate(labels)}
    cams = Counter(c["camera"] for c in order)
    main = {cam for cam, n in cams.items() if n >= MAIN_CAMERA_SHARE * len(order)}
    cost = []
    for n, c in enumerate(order):
        if c["camera"] in main and c.get("placed_by") == "clock":
            near = order[max(0, n - window):n] + order[n + 1:n + 1 + window]
            move = 1.0 if any(o["scene"] == c["scene"] for o in near) else island_cost
            # A back-and-forth intercut (A B A B A): the clips two away on both sides are its scene
            if any(0 <= n + 2 * d < len(order) and order[n + 2 * d]["scene"] == c["scene"]
                   and order[n + d]["scene"] != c["scene"] and 0 <= n - d < len(order)
                   and order[n - d]["scene"] == order[n + d]["scene"] for d in (-1, 1)):
                move = intercut_cost
        else:
            move = weak_cost
        row = [move] * len(labels)
        row[index[c["scene"]]] = 0.0
        cost.append(row)
    total, back = list(cost[0]), []
    for row in cost[1:]:
        best = min(range(len(labels)), key=lambda k: total[k])
        step, new = [], []
        for k in range(len(labels)):
            stay, switch = total[k], total[best] + switch_cost
            step.append(k if stay <= switch else best)
            new.append(min(stay, switch) + row[k])
        back.append(step)
        total = new
    k = min(range(len(labels)), key=lambda j: total[j])
    path = [k]
    for step in reversed(back):
        k = step[k]
        path.append(k)
    path.reverse()
    for c, k in zip(order, path):
        if labels[k] != c["scene"]:
            c["scene"], c["placed_by"] = labels[k], "cut"
    # A recording now used in several scenes is listed under the one most of its clips are in
    votes: dict = {}
    for c in order:
        votes.setdefault(c["recording"], Counter())[c["scene"]] += 1
    for row in scene_map["recordings"]:
        if row["recording"] in votes:
            row["scene"] = votes[row["recording"]].most_common(1)[0][0]
    return scene_map


def look_vector(look) -> tuple[float, float]:
    """(median level, warmth) of a clip's frame, as look-distance units: level in stops,
    warmth = log2(red / blue) of its mid-tones."""
    level, warmth = look[0], look[1]
    return (math.log2(max(level, 1e-3)) / LOOK_LUM_STOP, warmth / LOOK_WARM_UNIT)


def look_features(look) -> list[float]:
    """The look of a clip as one vector for refine_by_look: (level, warmth) as look_vector,
    then, when the look carries it, the frame's layout: 3 x 4 zones of (log2 level, log2
    red / blue, log2 green / magenta), scaled so a one-stop change over the whole frame
    weighs about as much as one stop of median level."""
    vec = list(look_vector(look))
    layout = look[2] if len(look) > 2 else []
    if layout:
        norm = math.sqrt(len(layout))
        vec += [(v if k % 3 == 0 else v / LOOK_WARM_UNIT) / norm for k, v in enumerate(layout)]
    return vec


def refine_by_look(scene_map: dict, looks: dict, clock: dict, switch_cost: float = REFINE_SWITCH_COST,
                   clock_cost: float = REFINE_CLOCK_COST, rounds: int = 5) -> dict:
    """Last pass, in the order of weight Ismael gave (2026-11-12): look first, cut second,
    clock last. Each clip, in cut order, may move to any scene: its cost there is its look
    distance to the scene's centre (look_features), plus `clock_cost` unless that scene holds
    most of the clips of its clock scene (`clock`: {clip id: scene from the clock alone});
    each change of scene along the cut costs `switch_cost`. Solved for the whole cut at once
    (Viterbi), then the centres move and it is solved again, `rounds` times at most.
    Scenes are then renumbered in the order they first appear."""
    clips = scene_map["clips"]
    order = sorted((c for c in clips if c["scene"] and c["id"] in looks), key=lambda c: (c["record_start"], c["track"]))
    if len(order) < 3:
        return scene_map
    vecs = [look_features(looks[c["id"]]) for c in order]
    dims = min(len(v) for v in vecs)
    vecs = [v[:dims] for v in vecs]
    label = [c["scene"] for c in order]
    for _ in range(rounds):
        scenes = sorted(set(label))
        centre = []
        for s in scenes:
            members = [v for v, l in zip(vecs, label) if l == s]
            centre.append([sum(col) / len(members) for col in zip(*members)])
        home: dict = {}
        for c, l in zip(order, label):
            home.setdefault(clock.get(c["id"]), Counter())[l] += 1
        cost = []
        for c, v in zip(order, vecs):
            top = home[clock.get(c["id"])].most_common(1)[0][0]
            cost.append([math.dist(v, m) + (0.0 if s == top else clock_cost) for s, m in zip(scenes, centre)])
        total, back = list(cost[0]), []
        for row in cost[1:]:
            best = min(range(len(scenes)), key=lambda k: total[k])
            step, new = [], []
            for k in range(len(scenes)):
                stay, switch = total[k], total[best] + switch_cost
                step.append(k if stay <= switch else best)
                new.append(min(stay, switch) + row[k])
            back.append(step)
            total = new
        k = min(range(len(scenes)), key=lambda j: total[j])
        path = [k]
        for step in reversed(back):
            k = step[k]
            path.append(k)
        path.reverse()
        new_label = [scenes[k] for k in path]
        if new_label == label:
            break
        label = new_label
    for c, l in zip(order, label):
        if c["scene"] != l:
            c["scene"], c["placed_by"] = l, "look"
    everything = sorted((c for c in clips if c["scene"]), key=lambda c: (c["record_start"], c["track"]))
    number: dict = {0: 0}
    for c in everything:
        number.setdefault(c["scene"], len(number))
    for row in clips:
        row["scene"] = number.get(row["scene"], row["scene"])
    _relabel_recordings(scene_map, everything)
    return scene_map


def _median(xs: list) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _look_cost(vs: list) -> float:
    m0, m1 = _median([v[0] for v in vs]), _median([v[1] for v in vs])
    return sum(min(abs(v[0] - m0) + abs(v[1] - m1), 3.0) for v in vs)   # one odd clip costs at most 3


def _relabel_recordings(scene_map: dict, order: list) -> None:
    votes: dict = {}
    for c in order:
        votes.setdefault(c["recording"], Counter())[c["scene"]] += 1
    for row in scene_map["recordings"]:
        if row["recording"] in votes:
            row["scene"] = votes[row["recording"]].most_common(1)[0][0]


def split_by_look(scene_map: dict, looks: dict, penalty: float = LOOK_PENALTY, min_len: int = LOOK_MIN_LEN) -> dict:
    """Clocks and the cut can join a day street, a shop and a night bar into one scene. Each
    scene's clips, in cut order, are cut into runs of one look (`looks`: {clip id: (median
    level, warmth)}), each run at least `min_len` clips, each new run costing `penalty`
    (optimal partition). Runs after the first become new scenes; clips without a look stay
    with the clip before them in the cut."""
    clips = scene_map["clips"]
    order = sorted((c for c in clips if c["scene"]), key=lambda c: (c["record_start"], c["track"]))
    next_id = max([c["scene"] for c in clips] + [0]) + 1
    splits = []
    for s in sorted({c["scene"] for c in order}):
        members = [c for c in order if c["scene"] == s]
        seen = [c for c in members if c["id"] in looks]
        n = len(seen)
        if n < 2 * min_len:
            continue
        vs = [look_vector(looks[c["id"]]) for c in seen]
        best, back = [0.0] + [math.inf] * n, [0] * (n + 1)
        for j in range(min_len, n + 1):
            for i in [0] + list(range(min_len, j - min_len + 1)):
                if best[i] < math.inf:
                    cost = best[i] + _look_cost(vs[i:j]) + penalty
                    if cost < best[j]:
                        best[j], back[j] = cost, i
        runs, j = [], n
        while j > 0:
            runs.append((back[j], j))
            j = back[j]
        if len(runs) < 2:
            continue
        runs.reverse()
        label = {}
        for k, (i, j) in enumerate(runs):
            for c in seen[i:j]:
                label[c["id"]] = s if k == 0 else next_id + k - 1
        next_id += len(runs) - 1
        splits.append((s, len(runs)))
        last = label[seen[0]["id"]]
        for c in members:
            last = label.get(c["id"], last)
            c["scene"] = last
    _relabel_recordings(scene_map, order)
    scene_map["merges"] = list(scene_map["merges"]) + [("look", s, k, 0) for s, k in splits]
    return scene_map


def cap_scenes(scene_map: dict, limit: int, looks: dict | None = None) -> dict:
    """At most `limit` scenes (scene 0 aside): the smallest scene joins another, again and
    again, then scenes are renumbered in the order they first appear. For a scene map that must
    fit a fixed set of labels, such as Resolve's 16 flag colours plus "no flag". Without
    `looks` it joins the scene it touches most often in the cut; with them ({clip id: (median
    level, warmth)}) the scene that looks most like it, anywhere in the cut, each shared cut
    boundary (up to 3) counting LOOK_TOUCH_BONUS: a night bar never joins a day street just
    because the edit puts them side by side."""
    clips = scene_map["clips"]
    order = sorted((c for c in clips if c["scene"]), key=lambda c: (c["record_start"], c["track"]))
    joined = []
    while True:
        size: dict = {}
        for c in order:
            size[c["scene"]] = size.get(c["scene"], 0) + 1
        if len(size) <= max(1, limit):
            break
        small = min(size, key=lambda s: (size[s], -s))
        touch: dict = {}
        for a, b in zip(order, order[1:]):
            if (a["scene"] == small) != (b["scene"] == small):
                other = b["scene"] if a["scene"] == small else a["scene"]
                touch[other] = touch.get(other, 0) + 1
        if looks:
            centre = {}
            for c in order:
                if c["id"] in looks:
                    centre.setdefault(c["scene"], []).append(look_vector(looks[c["id"]]))
            centre = {k: (_median([v[0] for v in vs]), _median([v[1] for v in vs])) for k, vs in centre.items()}

            def distance(o):
                d = (abs(centre[small][0] - centre[o][0]) + abs(centre[small][1] - centre[o][1])
                     if small in centre and o in centre else 2.0)
                return d - LOOK_TOUCH_BONUS * min(touch.get(o, 0), 3)
            into = min((o for o in size if o != small), key=lambda o: (distance(o), -size[o]))
        else:
            into = max(touch, key=lambda s: (touch[s], size[s]))
        joined.append((small, into, size[small]))
        for row in clips + scene_map["recordings"]:
            if row["scene"] == small:
                row["scene"] = into
    number: dict = {0: 0}
    for c in order:
        number.setdefault(c["scene"], len(number))
    for row in clips + scene_map["recordings"]:
        row["scene"] = number[row["scene"]]
    # merges keep the numbers from before the cap; "renumbered" maps those to the final ones
    scene_map["merges"] = list(scene_map["merges"]) + [("cap", a, b, n) for a, b, n in joined]
    scene_map["renumbered"] = number
    return scene_map


def summary(scene_map: dict) -> str:
    rows = scene_map["recordings"]
    lines = [f"{scene_map['timeline']}: {len(scene_map['clips'])} clips, {len(rows)} recordings, "
             f"{len({c['setup'] for c in scene_map['clips']})} setups, "
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
