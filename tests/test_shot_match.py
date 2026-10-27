"""Synthetic checks for shot_match.py's solver: a shot that is the hero with a known
exposure and white-balance change, re-framed, must come back with the inverse gain.

Run from the repo root:  python3 -m unittest tests.test_shot_match -v
Writes 16-bit DaVinci Intermediate TIFFs to a temp folder, then deletes them.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shot_match  # noqa: E402

# Surfaces of a scene: linear DWG reflectances, some neutral, some coloured
MATERIALS = np.array([[0.18, 0.18, 0.18], [0.05, 0.05, 0.05], [0.40, 0.36, 0.30],
                      [0.08, 0.14, 0.04], [0.30, 0.20, 0.12], [0.10, 0.12, 0.20]])


def encode(lin):
    return np.where(lin <= 0.00262409, lin * shot_match.DI_M,
                    shot_match.DI_C * (np.log2(np.maximum(lin + shot_match.DI_A, 1e-10)) + shot_match.DI_B))


def scene(seed, light, shares):
    """1080p frame of material patches with texture, in proportions `shares`, under `light`."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(len(MATERIALS), size=(54, 96), p=shares)
    img = MATERIALS[labels].repeat(20, 0).repeat(20, 1)
    img = img * rng.lognormal(0, 0.15, img.shape[:2])[..., None] * light
    return (np.clip(encode(img), 0, 1) * 65535).astype(np.uint16)


class SolverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        folder = Path(cls.tmp.name)
        even = np.full(len(MATERIALS), 1 / len(MATERIALS))
        tifffile.imwrite(folder / "A_HERO.tif", scene(1, np.array([1.0, 1.0, 1.0]), even))
        # Shot B: one stop brighter, a little warmer, framed on different proportions. A strong
        # cast would be ambiguous: grey under warm light has the chromaticity of beige stone.
        cls.light = np.array([2.1, 2.0, 1.85])
        tifffile.imwrite(folder / "B.tif", scene(2, cls.light, [0.1, 0.3, 0.1, 0.3, 0.1, 0.1]))
        cls.folder = folder
        cls.skin_finder = shot_match.SkinFinder
        shot_match.SkinFinder = NoSkin   # the synthetic frames have no people

    @classmethod
    def tearDownClass(cls):
        shot_match.SkinFinder = cls.skin_finder
        cls.tmp.cleanup()

    def test_recovers_exposure_and_white_balance(self):
        rows = shot_match.match_stills([("A", self.folder / "A_HERO.tif"), ("B", self.folder / "B.tif")],
                                       "A", self.folder / "out")
        gain = rows[1]["gain"]
        expected = 1 / self.light
        exposure_error = np.log2(shot_match.luma(gain) / shot_match.luma(expected))
        self.assertLess(abs(exposure_error), 0.1, f"gain {gain}, expected {expected}")
        wb_error = np.log2((gain[2] / gain[0]) / (expected[2] / expected[0]))
        self.assertLess(abs(wb_error), 0.15, f"gain {gain}, expected {expected}")
        self.assertTrue((self.folder / "out" / "match_values.csv").exists())
        self.assertTrue((self.folder / "out" / "contact_sheet.jpg").exists())

    def test_hero_is_neutral(self):
        rows = shot_match.match_stills([("A", self.folder / "A_HERO.tif"), ("B", self.folder / "B.tif")],
                                       "A", self.folder / "out")
        np.testing.assert_allclose(rows[0]["gain"], [1, 1, 1])

    def test_timecode(self):
        self.assertEqual(shot_match.frames_to_tc(86400 + 25, 24), "01:00:01:01")

    def test_measures_gain_gamma_and_offset_of_a_node(self):
        off = scene(3, np.ones(3), np.full(len(MATERIALS), 1 / len(MATERIALS))) / 65535.0
        lin = shot_match.decode_intermediate(off)
        gain = np.array([0.72, 0.75, 0.81])

        def node(offset, power):   # calibrated wheel law: 0.0073 linear per Offset unit
            out = np.maximum(lin * gain + (offset - 25) * 0.0073, 1e-6) ** power
            return np.round(np.clip(encode(out), 0, 1) * 65535) / 65535
        for offset, power, gamma in ((25, 1, 0), (25, 1.2, -0.05), (24, 1, 0), (25.5, 1.2, -0.05)):
            m = shot_match.measure_node(off, node(offset, power))
            np.testing.assert_allclose(m["gain"], gain, atol=0.01)
            np.testing.assert_allclose(m["gamma"], gamma, atol=0.003)
            np.testing.assert_allclose(m["offset"], offset, atol=0.06)
            self.assertEqual(m["offset_used"], offset != 25)
            self.assertLess(m["misfit_stops"], 0.05)

    def test_same_angle_shots_share_one_gain_unless_the_light_moved(self):
        a, b = np.zeros(48 * 27), np.zeros(48 * 27)
        a[0], b[1] = 1, 1
        rows = [{"clip": "hero", "gain": np.ones(3), "note": "hero"},
                {"clip": "A1", "gain": np.full(3, 0.80), "note": ""},
                {"clip": "A2", "gain": np.full(3, 0.70), "note": ""},
                {"clip": "A3", "gain": np.full(3, 0.30), "note": ""},   # same angle, light moved
                {"clip": "H1", "gain": np.full(3, 0.65), "note": ""}]
        prints = {"hero": b, "A1": a, "A2": a, "A3": a, "H1": b}
        shot_match.group_by_angle(rows, prints, hero_gain=[0.6, 0.6, 0.6])
        gains = {r["clip"]: r["gain"][0] for r in rows}
        self.assertAlmostEqual(gains["A1"], gains["A2"])
        self.assertAlmostEqual(gains["A3"], 0.30)
        self.assertAlmostEqual(gains["H1"], 0.6)   # the hero's angle takes the hero's own Gain
        self.assertEqual(gains["hero"], 1.0)


class NoSkin:
    reason = "disabled in tests"

    def pixels(self, encoded):
        return np.empty((0, 3))

    def close(self):
        pass


if __name__ == "__main__":
    unittest.main()
