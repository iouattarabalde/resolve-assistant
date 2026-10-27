"""The Resolve side of shot_match.py against a fake Resolve: node toggling, the hero's
Linear node kept on, playhead restored, and Gain written only in apply mode.

Run from the repo root:  python3 -m unittest tests.test_shot_match_resolve -v
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shot_match  # noqa: E402
from tests.test_shot_match import NoSkin, encode, scene  # noqa: E402

EVEN = np.full(6, 1 / 6)


class Graph:
    def __init__(self, labels):
        self.labels, self.enabled = labels, [True] * len(labels)
        self.GetNodeEnabled = None   # like Resolve: unknown methods come back as None

    def GetNumNodes(self):
        return len(self.labels)

    def GetNodeLabel(self, i):
        return self.labels[i - 1]

    def SetNodeEnabled(self, i, on):
        self.enabled[i - 1] = on
        return True


class Item:
    def __init__(self, name, start, light):
        self.name, self.start, self.light = name, start, light
        self.layers = {1: Graph(["Input"]), 3: Graph(["Linear", "Window"]), 4: Graph(["Scene"])}
        self.cdl = None
        self.hand_gain, self.hand_power = np.ones(3), 1.0   # what the Linear node does when on

    def GetName(self): return self.name
    def GetUniqueId(self): return self.name
    def GetStart(self): return self.start
    def GetEnd(self): return self.start + 48
    def GetNodeGraph(self, layer): return self.layers.get(layer)
    def GetColorGroup(self): return None

    def SetCDL(self, cdl):
        self.cdl = cdl
        return True


class Timeline:
    def __init__(self, items, hero):
        self.items, self.hero, self.tc = items, hero, "01:00:00:00"

    def GetSettings(self):
        return {"timelineFrameRate": 24.0, "timelineDropFrameTimecode": "0"}

    def GetCurrentVideoItem(self): return self.hero
    def GetSelectedClips(self): return [i for i in self.items if i is not self.hero]
    def GetCurrentTimecode(self): return self.tc

    def SetCurrentTimecode(self, tc):
        self.tc = tc
        return True

    def item_at_playhead(self):
        h, m, s, f = map(int, self.tc.split(":"))
        frame = ((h * 60 + m) * 60 + s) * 24 + f
        return next(i for i in self.items if i.GetStart() <= frame < i.GetEnd())


class Project:
    def __init__(self, timeline, test):
        self.timeline, self.test = timeline, test

    def GetCurrentTimeline(self): return self.timeline

    def ExportCurrentFrameAsStill(self, path):
        item = self.timeline.item_at_playhead()
        # Record what was switched off at the moment of the export
        self.test.states.setdefault(item.name, []).append({l: list(g.enabled) for l, g in item.layers.items()})
        img = scene(1 if item is self.timeline.hero else 2, item.light, EVEN)
        if item.layers[3].enabled[0]:   # the Linear node: Gain then Gamma, in linear
            lin = shot_match.decode_intermediate(img / 65535.0)
            img = (np.clip(encode(np.maximum(lin * item.hand_gain, 1e-6) ** item.hand_power), 0, 1) * 65535).astype(np.uint16)
        tifffile.imwrite(path, img)
        return True


class Resolve:
    def __init__(self, project):
        self.project = project

    def GetProjectManager(self):
        return mock.Mock(GetCurrentProject=lambda: self.project)

    def OpenPage(self, page): return True


CONFIRM_WRITE = shot_match.confirm_write   # the real one, before setUp patches it


class ResolveFlowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.states = {}
        self.hero = Item("hero", 86400, np.ones(3))
        self.shot = Item("shot", 86448, np.full(3, 2.0))
        self.timeline = Timeline([self.hero, self.shot], self.hero)
        resolve = Resolve(Project(self.timeline, self))
        self.patches = [mock.patch.object(shot_match, "OUT_ROOT", Path(self.tmp.name)),
                        mock.patch.object(shot_match, "get_resolve", lambda: resolve),
                        mock.patch.object(shot_match, "SkinFinder", NoSkin),
                        mock.patch.object(shot_match.time, "sleep", lambda s: None),
                        mock.patch.object(shot_match, "confirm_write", lambda *a: self.confirm)]
        self.confirm = True
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_old_runs_are_cleaned_and_only_ours(self):
        out = Path(self.tmp.name)
        old_run, new_run = out / "20260901_120000", out / "20261020_120000"
        for d in (old_run, new_run):
            (d / "stills").mkdir(parents=True)
            (d / "stills" / "a.tif").write_bytes(b"x" * 10)
        (out / "collect_20260901_120000.zip").write_bytes(b"x")
        (out / "20260901_notes.txt").write_text("mine")      # not a name the script makes
        with mock.patch("sys.stdout", io.StringIO()):
            shot_match.clean_old_runs(now=shot_match.datetime.datetime(2026, 10, 27))
        self.assertFalse(old_run.exists())
        self.assertFalse((out / "collect_20260901_120000.zip").exists())
        self.assertTrue(new_run.exists())
        self.assertTrue((out / "20260901_notes.txt").exists())

    def test_preview_writes_nothing_and_restores(self):
        self.assertEqual(shot_match.run_in_resolve(apply=False), 0)
        self.assertIsNone(self.shot.cdl)
        run = next(Path(self.tmp.name).glob("2*"))
        self.assertTrue((run / "contact_sheet.jpg").exists())
        self.assertFalse((run / "stills").exists(), "stills are deleted once solved")
        self.assertEqual(self.timeline.tc, "01:00:00:00")
        for item in (self.hero, self.shot):
            for graph in item.layers.values():
                self.assertTrue(all(graph.enabled), f"{item.name} left a node off")

    def test_capture_switches_the_right_nodes(self):
        shot_match.run_in_resolve(apply=False)
        self.assertEqual(self.states["shot"], [{1: [True], 3: [False, False], 4: [False]}])
        # The hero as graded (Linear on), then once more with it off to measure its Gamma
        self.assertEqual(self.states["hero"], [{1: [True], 3: [True, False], 4: [False]},
                                               {1: [True], 3: [False, False], 4: [False]}])

    def test_apply_writes_gain_to_the_shot_only(self):
        shot_match.run_in_resolve(apply=True)
        self.assertIsNone(self.hero.cdl)
        self.assertEqual(self.shot.cdl["NodeIndex"], "1")
        self.assertEqual(self.shot.cdl["Offset"], "0.0000 0.0000 0.0000")
        slope = np.array([float(v) for v in self.shot.cdl["Slope"].split()])
        np.testing.assert_allclose(slope, 0.5, atol=0.04)

    def test_apply_carries_the_hero_gamma(self):
        self.hero.hand_gain, self.hero.hand_power = np.array([0.6, 0.6, 0.6]), 1.2
        shot_match.run_in_resolve(apply=True)
        power = np.array([float(v) for v in self.shot.cdl["Power"].split()])
        np.testing.assert_allclose(power, 1.2, atol=0.01)
        # Under that Gamma the shot (twice as bright) needs half the hero's Gain
        slope = np.array([float(v) for v in self.shot.cdl["Slope"].split()])
        np.testing.assert_allclose(slope, 0.3, atol=0.03)

    def test_collect_says_offset_untouched(self):
        self.shot.hand_gain = np.full(3, 0.5)
        with mock.patch.object(shot_match, "show_in_file_browser"):
            shot_match.run_collect()
        archive = next(Path(self.tmp.name).glob("collect_*.zip"))
        shutil.unpack_archive(archive, Path(self.tmp.name) / "u")
        truth = json.loads((Path(self.tmp.name) / "u" / archive.stem / "truth.json").read_text())
        self.assertFalse(any(m["offset_used"] for m in truth["clips"].values()))

    def test_apply_cancelled_writes_nothing(self):
        self.confirm = False
        shot_match.run_in_resolve(apply=True)
        self.assertIsNone(self.shot.cdl)

    def test_no_window_means_no_write(self):
        no_fusion = mock.Mock(**{"Fusion.return_value": None})
        self.assertFalse(CONFIRM_WRITE(no_fusion, 1, "hero"))

    def test_collect_measures_the_hand_gain_and_writes_nothing(self):
        self.hero.hand_gain = np.array([1.04, 1.0, 0.96])
        self.shot.hand_gain = self.hero.hand_gain / 2   # the shot is lit twice as bright
        with mock.patch.object(shot_match, "show_in_file_browser") as shown:
            self.assertEqual(shot_match.run_collect(), 0)
        shown.assert_called_once()
        self.assertIsNone(self.shot.cdl)
        self.assertEqual(self.timeline.tc, "01:00:00:00")
        for item in (self.hero, self.shot):
            for graph in item.layers.values():
                self.assertTrue(all(graph.enabled), f"{item.name} left a node off")
        out = Path(self.tmp.name)
        archive = next(out.glob("collect_*.zip"))
        self.assertFalse(archive.with_suffix("").exists(), "the folder should only live in the zip")
        shutil.unpack_archive(archive, out / "unzipped")
        folder = out / "unzipped" / archive.stem
        truth = json.loads((folder / "truth.json").read_text())
        self.assertEqual(truth["hero"], "01_hero")
        np.testing.assert_allclose(truth["clips"]["02_shot"]["gain"], [0.52, 0.5, 0.48], atol=0.005)
        self.assertLess(truth["clips"]["02_shot"]["misfit_stops"], 0.05)
        # The last export of each clip is the Linear-on one: every other layer 3 and 4 node off
        self.assertEqual(self.states["shot"][-1], {1: [True], 3: [True, False], 4: [False]})
        self.assertTrue((folder / "hero_graded.tif").exists())
        self.assertFalse((folder / "_graded").exists())

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(shot_match.run_bench([folder]), 0)
        rms = float(re.search(r"Exposure error RMS ([\d.]+)", out.getvalue()).group(1))
        self.assertLess(rms, 0.1)


if __name__ == "__main__":
    unittest.main()
