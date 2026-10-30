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
        self.hand_gain, self.hand_power, self.hand_offset = np.ones(3), 1.0, 25.0   # Linear node by hand
        self.honours_offset = 1   # 0: a Resolve that drops the CDL Offset
        self.flags, self.colour, self.shares = [], "", EVEN

    def GetName(self): return self.name
    def GetUniqueId(self): return self.name
    def GetStart(self): return self.start
    def GetEnd(self): return self.start + 48
    def GetNodeGraph(self, layer): return self.layers.get(layer)
    def GetColorGroup(self): return None
    def GetFlagList(self): return self.flags
    def GetClipColor(self): return self.colour

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
        return self.render(self.timeline.item_at_playhead(), path)

    def render(self, item, path):
        # Record what was switched off at the moment of the export
        self.test.states.setdefault(item.name, []).append({l: list(g.enabled) for l, g in item.layers.items()})
        img = scene(1 if item is self.timeline.hero else 2, item.light, item.shares)
        if item.layers[3].enabled[0]:   # the Linear node, in linear: written CDL, else the hand values
            lin = shot_match.decode_intermediate(img / 65535.0)
            if item.cdl:
                slope, offset, power = (np.array([float(v) for v in item.cdl[k].split()])
                                        for k in ("Slope", "Offset", "Power"))
                offset = offset * item.honours_offset
            else:
                slope, offset, power = item.hand_gain, (item.hand_offset - 25) * 0.0073, item.hand_power
            img = (np.clip(encode(np.maximum(lin * slope + offset, 1e-6) ** power), 0, 1) * 65535).astype(np.uint16)
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
                        # the synthetic shots differ by exposure only: correct all of it
                        mock.patch.object(shot_match, "KEEP_LIGHT_FROM", float("inf")),
                        mock.patch.object(shot_match, "confirm_write", lambda *a: self.confirm)]
        self.confirm = True
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def two_scenes(self):
        """Timeline of two intercut scenes, heroes flagged: A (hand Gain 1) and B (hand Gain 0.8)."""
        warm, cool = np.array([0.3, 0.3, 0.3, 0.05, 0.05, 0.0]), np.array([0, 0.05, 0.05, 0.3, 0.3, 0.3])
        hero_a, hero_b = Item("hero_a", 86400, np.ones(3)), Item("hero_b", 86448, np.ones(3))
        shot_b, shot_a = Item("shot_b", 86496, np.full(3, 0.5)), Item("shot_a", 86544, np.full(3, 2.0))
        for item, shares in ((hero_a, warm), (shot_a, warm), (hero_b, cool), (shot_b, cool)):
            item.shares = shares
        hero_a.flags = hero_b.flags = [shot_match.HERO_FLAG]
        hero_b.hand_gain = np.full(3, 0.8)
        self.timeline.items, self.timeline.hero = [hero_a, hero_b, shot_b, shot_a], None
        return hero_a, hero_b, shot_a, shot_b

    def test_flagged_heroes_each_match_their_own_scene(self):
        hero_a, hero_b, shot_a, shot_b = self.two_scenes()
        with mock.patch("sys.stdout", io.StringIO()) as out:
            shot_match.run_in_resolve(apply=True)
        self.assertIn("2 heroes flagged", out.getvalue())
        self.assertIsNone(hero_a.cdl)
        self.assertIsNone(hero_b.cdl)
        slope_a = np.array([float(v) for v in shot_a.cdl["Slope"].split()])
        slope_b = np.array([float(v) for v in shot_b.cdl["Slope"].split()])
        np.testing.assert_allclose(slope_a, 0.5, atol=0.05)   # twice as bright as hero A
        np.testing.assert_allclose(slope_b, 1.6, atol=0.15)   # half as bright as hero B, graded 0.8
        run = next(Path(self.tmp.name).glob("2*"))
        self.assertTrue((run / "scene_01" / "match_values.csv").exists())
        self.assertTrue((run / "scene_02" / "match_values.csv").exists())

    def test_clip_colour_overrides_the_automatic_hero(self):
        hero_a, hero_b, shot_a, shot_b = self.two_scenes()
        hero_b.colour = shot_a.colour = "Orange"
        hero_a.colour = "Teal"
        with mock.patch("sys.stdout", io.StringIO()) as out:
            shot_match.run_in_resolve(apply=False)
        self.assertIn("hero by clip colour", out.getvalue())
        run = next(Path(self.tmp.name).glob("2*"))
        csv_text = (run / "scene_02" / "match_values.csv").read_text()
        self.assertIn("shot_a", csv_text)
        self.assertIn("hero by clip colour", csv_text)   # the assignment reaches the CSV too

    def test_unsure_hero_is_not_written(self):
        hero_a, hero_b, shot_a, shot_b = self.two_scenes()
        sure = shot_match.assign_heroes
        def unsure(order, sigs, heroes, colours):
            got = sure(order, sigs, heroes, colours)
            got["04_shot_a"] = (got["04_shot_a"][0], "check its hero: close call with 01_hero_a")
            return got
        with mock.patch.object(shot_match, "assign_heroes", unsure), mock.patch("sys.stdout", io.StringIO()) as out:
            shot_match.run_in_resolve(apply=True)
        self.assertIsNone(shot_a.cdl)
        self.assertIsNotNone(shot_b.cdl)
        self.assertIn("1 shots with an unsure hero will NOT be written", out.getvalue())

    def flag_scenes(self):
        """The two intercut scenes, every clip flagged in its scene's colour, heroes Green too."""
        hero_a, hero_b, shot_a, shot_b = self.two_scenes()
        hero_a.flags, shot_a.flags = ["Blue", shot_match.HERO_FLAG], ["Blue"]
        hero_b.flags, shot_b.flags = [shot_match.HERO_FLAG, "Cyan"], ["Cyan"]
        # The look alone would send each shot to the other hero: the flags must win
        shot_a.shares, shot_b.shares = hero_b.shares, hero_a.shares
        return hero_a, hero_b, shot_a, shot_b

    def test_scenes_by_flag_colour(self):
        hero_a, hero_b, shot_a, shot_b = self.flag_scenes()
        loose = Item("loose", 86592, np.full(3, 2.0))
        self.timeline.items.append(loose)
        with mock.patch("sys.stdout", io.StringIO()) as out:
            shot_match.run_in_resolve(apply=True)
        self.assertIn("2 scenes by flag colour", out.getvalue())
        self.assertIn("no scene flag: loose", out.getvalue())
        self.assertIsNone(loose.cdl)
        self.assertNotIn("close call", out.getvalue())
        run = next(Path(self.tmp.name).glob("2*"))
        self.assertIn("shot_a", (run / "scene_01_Blue" / "match_values.csv").read_text())
        self.assertIn("shot_b", (run / "scene_02_Cyan" / "match_values.csv").read_text())
        np.testing.assert_allclose([float(v) for v in shot_a.cdl["Slope"].split()], 0.5, atol=0.05)

    def test_scene_flags_need_one_hero_each(self):
        hero_a, hero_b, shot_a, shot_b = self.flag_scenes()
        shot_b.flags = ["Blue"]          # Cyan now only holds hero B: fine
        hero_b.flags = [shot_match.HERO_FLAG]
        with self.assertRaisesRegex(shot_match.Stop, "hero_b has no scene flag"):
            shot_match.run_in_resolve()
        hero_b.flags = ["Blue", shot_match.HERO_FLAG]
        with self.assertRaisesRegex(shot_match.Stop, "Scene Blue has 2 clips"):
            shot_match.run_in_resolve()
        hero_b.flags, shot_b.flags = [shot_match.HERO_FLAG, "Cyan"], ["Cyan", "Pink"]
        with self.assertRaisesRegex(shot_match.Stop, "one scene colour a clip"):
            shot_match.run_in_resolve()

    def test_scene_flags_without_green_hero_stop(self):
        hero_a, hero_b, shot_a, shot_b = self.flag_scenes()
        hero_a.flags, hero_b.flags = ["Blue"], ["Cyan"]
        self.timeline.hero = hero_a      # playhead hero, but two scenes
        with self.assertRaisesRegex(shot_match.Stop, "Scene Cyan has no clip"):
            shot_match.run_in_resolve()

    def test_all_clips_flagged_stops(self):
        for item in self.timeline.items:
            item.flags = [shot_match.HERO_FLAG]
        self.timeline.hero = None
        with self.assertRaises(shot_match.Stop):
            shot_match.open_scene()

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
        self.assertEqual(self.shot.cdl["Offset"], "0.00000 0.00000 0.00000")
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

    def test_apply_writes_the_hero_offset_and_checks_the_image(self):
        self.hero.hand_gain, self.hero.hand_offset = np.array([0.6, 0.6, 0.6]), 23.5
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            shot_match.run_in_resolve(apply=True)
        offset = np.array([float(v) for v in self.shot.cdl["Offset"].split()])
        np.testing.assert_allclose(offset, -1.5 * 0.0073, atol=0.002)
        self.assertIn("written nodes do what was intended", out.getvalue())

    def test_apply_check_catches_a_dropped_offset(self):
        self.hero.hand_gain, self.hero.hand_offset = np.array([0.6, 0.6, 0.6]), 22.0
        self.shot.honours_offset = 0
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            shot_match.run_in_resolve(apply=True)
        self.assertIn("CHECK FAILED", out.getvalue())

    def give_export_frame(self):
        """21.1.2: renders the clip as its nodes are switched now, wherever the playhead is;
        maxDimension shrinks it."""
        project = shot_match.get_resolve().GetProjectManager().GetCurrentProject()

        def make(item):
            def export_frame(offset, options):
                path = options["filePath"]
                project.render(item, path)
                if options.get("maxDimension"):
                    img = tifffile.imread(path)
                    step = img.shape[1] // options["maxDimension"]
                    tifffile.imwrite(path, img[::step, ::step])
                return {"success": True}
            return export_frame
        for item in (self.hero, self.shot):
            item.ExportFrame = make(item)

    def test_export_frame_leaves_the_playhead_alone(self):
        moves = []
        self.timeline.SetCurrentTimecode = lambda tc: moves.append(tc) or True
        self.give_export_frame()
        shot_match.run_in_resolve(apply=True)
        self.assertEqual(set(moves), {"01:00:00:00"}, "only the harmless put-back to where it was")
        slope = np.array([float(v) for v in self.shot.cdl["Slope"].split()])
        np.testing.assert_allclose(slope, 0.5, atol=0.04)
        self.assertEqual(self.states["shot"][0], {1: [True], 3: [False, False], 4: [False]})
        for item in (self.hero, self.shot):
            for graph in item.layers.values():
                self.assertTrue(all(graph.enabled))

    def test_collect_saves_extra_frames_and_the_solver_uses_them(self):
        self.give_export_frame()
        self.shot.hand_gain = np.full(3, 0.5)
        with mock.patch.object(shot_match, "show_in_file_browser"):
            shot_match.run_collect()
        out = Path(self.tmp.name)
        archive = next(out.glob("collect_*.zip"))
        shutil.unpack_archive(archive, out / "u")
        folder = out / "u" / archive.stem
        names = sorted(p.name for p in folder.rglob("*.tif"))
        self.assertEqual(names, ["01_hero.tif", "01_hero__f1.tif", "01_hero__f2.tif", "02_shot.tif",
                                 "02_shot__f1.tif", "02_shot__f2.tif", "hero_graded.tif",
                                 "hero_graded__f1.tif", "hero_graded__f2.tif"])
        self.assertEqual(tifffile.imread(folder / "stills" / "02_shot__f1.tif").shape[1], 960)
        stdout = io.StringIO()
        with mock.patch("sys.stdout", stdout):
            shot_match.run_bench([folder])
        rms = float(re.search(r"RMS ([\d.]+)", stdout.getvalue()).group(1))
        self.assertLess(rms, 0.1)

    def test_collect_final_saves_the_whole_grade(self):
        self.give_export_frame()
        with mock.patch.object(shot_match, "show_in_file_browser"):
            shot_match.run_collect(final=True)
        out = Path(self.tmp.name)
        archive = next(out.glob("collect_*.zip"))
        shutil.unpack_archive(archive, out / "u")
        final = out / "u" / archive.stem / "final"
        self.assertEqual(sorted(p.name for p in final.glob("*.tif")), ["01_hero.tif", "02_shot.tif"])
        # Shrunk for upload: analysis size, lossless zlib
        with tifffile.TiffFile(final / "02_shot.tif") as tif:
            self.assertEqual(tif.pages[0].shape[1], 960)
            self.assertEqual(tif.pages[0].compression, 8)   # ADOBE_DEFLATE (zlib)
        # The whole-grade exports switch nothing: every node on, as left by the colourist
        self.assertEqual(self.states["shot"][-1], {1: [True], 3: [True, True], 4: [True]})
        for item in (self.hero, self.shot):
            for graph in item.layers.values():
                self.assertTrue(all(graph.enabled))

    def test_export_frame_failure_falls_back_to_the_playhead(self):
        for item in (self.hero, self.shot):
            item.ExportFrame = lambda offset, options: {"success": False, "error": "nope"}
        shot_match.run_in_resolve(apply=False)
        self.assertEqual(self.timeline.tc, "01:00:00:00")
        self.assertEqual(len(self.states["shot"]), 1)

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
