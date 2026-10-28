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

### In MagicScript (where it lives)

One script, **Shot match to the hero** (Colour), with a Mode box: Preview, Apply, Collect or
Collect Final.
The layer, node label, output folder and "keep part of a big light difference" are boxes in
its dialog. Build it from this repo, never edit the copies:

```
python magicscript/build.py      # -> dist/magicscript/Colour/shot_match.py + _shot_match_engine.py
```

Copy both files into
`%APPDATA%\Blackmagic Design\DaVinci Resolve\Support\MagicScript\scripts\Colour\` and press
Reload. MagicScript skips files starting with `_`, so the engine (a copy of `shot_match.py`) is
not listed. The script installs numpy, tifffile, Pillow and MediaPipe into MagicScript's
`pydeps` on first run, then runs the engine in a separate ResolvePython connected to Resolve
(`--resolve MODE`), so nothing compiled loads into Resolve. That needs Preferences > System >
General > External scripting using: Local. MediaPipe goes in without OpenCV, matplotlib and
sounddevice, which the skin segmenter never uses.

The script is flagged destructive, so MagicScript asks for its "I understand" tick on every
run, Preview and Collect included: MagicScript can only make the tick depend on a yes/no box,
not on a choice. Apply passes `--confirmed`, so the engine's own window does not open.
SetCDL still writes into the layer the node editor shows, hence the warning in the dialog.

### From the Scripts menu

Install it twice in
`%APPDATA%\Blackmagic Design\DaVinci Resolve\Support\Fusion\Scripts\Utility\`:

| Saved as | Does |
|---|---|
| `Shot Match Preview.py` | prints the values, saves a CSV and a before/after contact sheet to `Documents/ShotMatch/`. Writes nothing into the grade. |
| `Shot Match Apply.py` | the same, then writes Gain into each shot's Linear node. The hero is left as it is |
| `Shot Match Collect.py` | on a scene you matched by hand: measures what each Linear node does and saves it to `Documents/ShotMatch/collect_*/`. Writes nothing into the grade |
| `Shot Match Collect Final.py` | the same, on a finished scene, plus each clip with its whole grade on (`final/`), to see what the finished grade lines up. Writes nothing into the grade |

The file decides from its own name : only a name containing "Apply" writes. Output appears in
Workspace > Console; the first line says PREVIEW, APPLY or COLLECT.

Collect exports each clip twice, Linear node off then on, with every other node of layers 3
and 4 off both times, so only the Linear node is measured. Its Gain, Gamma and Offset per
channel are fitted to the pair (within 0.01 of the wheels on a real 21.1 scene). One Offset
unit is only 0.0073 linear, so an untouched wheel would read anywhere from 24.9 to 25.3:
Offset is reported as used only when it sits 0.25 unit or more from 25 and clearly explains
the frame better; otherwise it reads "not used". A node with curves or saturation is flagged. It saves the
stills with the node off, the hero with it on (`hero_graded.tif`) and `truth.json`, zips
them, keeps only the zip and opens Explorer on it.

`Documents/ShotMatch/` stays small : Preview and Apply delete their stills (12 MB a clip)
once solved and keep the CSV, the contact sheet and the list of clips written. At the start
of every run, runs older than 30 days are deleted (`KEEP_DAYS`). Only the names the script
creates are touched; anything else in that folder is left alone. Each collected scene is one
more scene to score and tune the solver on:

```
python shot_match.py --bench FOLDER [FOLDER ...]    # solver vs hand, per shot and overall
```

What it does to the project while it runs :

- exports the middle frame of each clip with `ExportFrame` (21.1.2), without moving the playhead, plus two smaller frames (at 20 % and 80 % of the clip) that the solver pools with it; on older versions it moves the playhead to the middle frame only, and back;
- prints its progress and the time left in the Console every 10 s;
- switches off every node of layers 3 and 4 and of the group Post-Clip graph (the hero's
  Linear node stays on), exports a still, then switches them back on. The API has no way to
  read whether a node was already off, so a node you had switched off yourself comes back on;
- Apply only : **show layer 3 in the node editor before running.** `SetCDL` has no layer
  argument and writes into whichever layer is on screen (run from layer 4, it wrote node 1
  of layer 4). The 21.1 API has no call to pick or read that layer, so after the solve a
  window asks you to confirm layer 3 is showing. Cancel, or closing it, writes nothing.
  Without that window (run from a terminal) nothing is written.
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
against hand matches.

Shots of one camera angle get one Gain, as in a hand first pass: shots whose thumbnails
correlate above 0.9 form a group and take its median Gain, or the hero's own Gain when the
hero is in it. A shot solved more than ⅓ stop from its group keeps its own value (the light
moved). The hero is also exported with its Linear node off, to measure its Gain, Gamma and Offset:
shots are solved under the hero's Gamma and Offset. Apply writes all three through
`SetCDL`: the shot's Gain as Slope, the hero's Offset as CDL Offset, its Gamma as Power.
Which wheels Resolve shows them on is up to Resolve (a CDL Offset does not move the Offset
wheel), so after writing, Apply exports each shot with only its Linear node on and checks
on the image that the node does what was intended; a gap over 0.05 stop is reported.

Bench, three scenes and 32 shots matched by hand : 31 within ⅓ stop, exposure error RMS
0.18 stop, white balance RMS 0.07 stop.

```
python shot_match.py --stills FOLDER    # offline, on DaVinci Intermediate TIFFs, one named *HERO*
```

### Needs

Resolve Studio 21.0.4 or later (for `Timeline.GetSelectedClips()`), Python 3.9 or later,
and `pip install numpy tifffile pillow mediapipe`. The skin model (16 MB) is downloaded to
`~/.shot_match/` on first run.

### Tests

```
python -m unittest tests.test_shot_match tests.test_shot_match_resolve tests.test_magicscript -v
```

The first builds a hero and a shot one stop brighter and slightly warmer, framed
differently, and checks the solver gives the inverse gain. The second runs the Resolve side
against a fake Resolve : which nodes are off at export, the playhead and nodes restored,
and Gain written to the shot only in apply mode; Collect measuring a known hand Gain and
the bench scoring it. The third checks the MagicScript wrapper passes the dialog's settings
to the engine and relays its lines and errors.
