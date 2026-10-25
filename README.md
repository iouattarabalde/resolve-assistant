# Resolve Assistant

Scripts that do the assistant's side of a grading job : checking what comes in, and checking
what goes out. Built for one freelance suite, kept generic : no client names, no paths beyond
Resolve's defaults.

## qc_check.py

QC of graded renders against their sources, before anything is sent to a client. Read-only :
it never writes into the render or source folders and never touches Resolve.

```
python qc_check.py RENDERS --sources CAMERA_MEDIA --edl CUT.edl
```

| Check | FAIL when |
|---|---|
| Inventory | a source, or an EDL clip, has no render with the same name |
| Codec | codec or profile differs from `--codec` (default `prores4444`) |
| Format | resolution or frame rate differs from the source, or from `--resolution` / `--fps` |
| Handles | the render does not cover each EDL event plus `--handles` (default 24), or runs outside the source media |
| Frames | a black run of 3 frames or fewer inside the clip. Possible flash frames are a WARN |
| Levels | never : luma outside video range is a WARN, since some deliveries allow it |
| Audio | no audio stream, with `--audio` |

It writes two files to `qc_reports/` : the full report, and a short client summary, written
only when nothing failed. Exit code 0 means no FAIL, 1 means at least one FAIL, 2 means the
check could not run. A check that crashes is reported as a FAIL, never as a pass.

**Limits of v1.** Sources that ffprobe cannot read (R3D, BRAW) are compared to the spec only,
and a short handle against them is a WARN, since the media length is unknown. Drop-frame
timecode is read as non-drop. EDL only, CMX3600 : no XML or AAF yet. The black and flash
thresholds are tuned on test patterns, not yet on real footage.

### Needs

Python 3.9 or later and ffmpeg/ffprobe on the PATH. Standard library only.

### Tests

```
python -m unittest tests.test_qc_check -v
```

Builds a clean delivery and a broken one with ffmpeg (wrong frame rate, a 12-frame head
handle, a renamed clip, one black frame) and checks that the first passes and each fault in
the second is caught.

## shot_match.py

First-pass shot matching on the Color page. Park the playhead on the hero, select the other
clips of the scene, run the script. Each clip is matched to the hero as graded, with Gain
R/G/B only, in the node labelled `Linear` at the head of layer 3.

Install it twice in
`%APPDATA%\Blackmagic Design\DaVinci Resolve\Support\Fusion\Scripts\Utility\`:

| Saved as | Does |
|---|---|
| `Shot Match Preview.py` | prints the values, saves stills, a CSV and a before/after contact sheet to `Documents/ShotMatch/`. Writes nothing into the grade |
| `Shot Match Apply.py` | the same, then writes Gain into each shot's Linear node. The hero is left as it is |

The file decides from its own name : only a name containing "Apply" writes. Output appears in
Workspace > Console; the first line says PREVIEW or APPLY.

What it does to the project while it runs :

- moves the playhead to the middle frame of each clip, and back where it was at the end;
- switches off every node of layers 3 and 4 and of the group Post-Clip graph (the hero's
  Linear node stays on), exports a still, then switches them back on. The API has no way to
  read whether a node was already off, so a node you had switched off yourself comes back on;
- Apply only : `SetCDL` replaces the Linear node's wheels. Lift, Gamma or Gain set by hand
  in that node are lost, and Resolve cannot undo it. Duplicate the timeline first.

A clip is skipped unless it has exactly one node labelled `Linear` in layer 3.

Tested on Resolve Studio 21.1 : `SetCDL` with the Linear node's index lands on layer 3, and
Slope maps one to one onto Gain R/G/B. A CDL Offset does not reach the Offset wheel (it is
folded into Gain), so the script writes Offset 0 and Power 1.

How it matches : exposure from the surfaces both shots share (pixels grouped by colour, each
group votes how many stops apart the shots are), pulled 40 % towards skin when both frames
show skin. Skin comes from MediaPipe's multiclass selfie segmenter; without MediaPipe it
matches on surfaces only. White balance from the same colour groups. Tuned on two scenes
against hand matches : within ⅓ stop on 6 of 7 and 10 of 11 shots.

```
python shot_match.py --stills FOLDER    # offline, on DaVinci Intermediate TIFFs, one named *HERO*
```

### Needs

Resolve Studio 21.0.4 or later (for `Timeline.GetSelectedClips()`), Python 3.9 or later,
and `pip install numpy tifffile pillow mediapipe`. The skin model (16 MB) is downloaded to
`~/.shot_match/` on first run.

### Tests

```
python -m unittest tests.test_shot_match tests.test_shot_match_resolve -v
```

The first builds a hero and a shot one stop brighter and slightly warmer, framed
differently, and checks the solver gives the inverse gain. The second runs the Resolve side
against a fake Resolve : which nodes are off at export, the playhead and nodes restored,
and Gain written to the shot only in apply mode.
