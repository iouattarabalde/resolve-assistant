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

    def test_measures_gain_offset_gamma_of_a_node(self):
        off = scene(3, np.ones(3), np.full(len(MATERIALS), 1 / len(MATERIALS)))
        lin = shot_match.decode_intermediate(off / 65535.0)
        gain, offset, gamma = np.array([1.4, 1.5, 1.6]), np.array([24.0, 25.5, 24.5]), -0.06
        on = np.maximum(lin * gain + (offset - 25) * shot_match.OFFSET_STEP, 1e-6) ** shot_match.wheel_power(gamma)
        m = shot_match.measure_node(off / 65535.0, np.round(np.clip(encode(on), 0, 1) * 65535) / 65535)
        np.testing.assert_allclose(m["gain"], gain, atol=0.02)
        np.testing.assert_allclose(m["offset"], offset, atol=0.15)
        np.testing.assert_allclose(m["gamma"], gamma, atol=0.006)


class NoSkin:
    reason = "disabled in tests"

    def pixels(self, encoded):
        return np.empty((0, 3))

    def close(self):
        pass


if __name__ == "__main__":
    unittest.main()
