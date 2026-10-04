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
