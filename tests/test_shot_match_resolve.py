"""The Resolve side of shot_match.py against a fake Resolve: node toggling, the hero's
Linear node kept on, playhead restored, and Gain written only in apply mode.

Run from the repo root:  python3 -m unittest tests.test_shot_match_resolve -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shot_match  # noqa: E402
from tests.test_shot_match import NoSkin, scene  # noqa: E402

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

    def GetSetting(self, key):
        return {"timelineFrameRate": "24", "timelineDropFrameTimecode": "0"}[key]

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
        self.test.states[item.name] = {l: list(g.enabled) for l, g in item.layers.items()}
        tifffile.imwrite(path, scene(1 if item is self.timeline.hero else 2, item.light, EVEN))
        return True


class Resolve:
    def __init__(self, project):
        self.project = project

    def GetProjectManager(self):
        return mock.Mock(GetCurrentProject=lambda: self.project)

    def OpenPage(self, page): return True


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
                        mock.patch.object(shot_match.time, "sleep", lambda s: None)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_preview_writes_nothing_and_restores(self):
        self.assertEqual(shot_match.run_in_resolve(apply=False), 0)
        self.assertIsNone(self.shot.cdl)
        self.assertEqual(self.timeline.tc, "01:00:00:00")
        for item in (self.hero, self.shot):
            for graph in item.layers.values():
                self.assertTrue(all(graph.enabled), f"{item.name} left a node off")

    def test_capture_switches_the_right_nodes(self):
        shot_match.run_in_resolve(apply=False)
        self.assertEqual(self.states["shot"], {1: [True], 3: [False, False], 4: [False]})
        self.assertEqual(self.states["hero"], {1: [True], 3: [True, False], 4: [False]})

    def test_apply_writes_gain_to_the_shot_only(self):
        shot_match.run_in_resolve(apply=True)
        self.assertIsNone(self.hero.cdl)
        self.assertEqual(self.shot.cdl["NodeIndex"], "1")
        self.assertEqual(self.shot.cdl["Offset"], "0.0000 0.0000 0.0000")
        slope = np.array([float(v) for v in self.shot.cdl["Slope"].split()])
        np.testing.assert_allclose(slope, 0.5, atol=0.04)


if __name__ == "__main__":
    unittest.main()
