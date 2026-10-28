"""The MagicScript wrapper: its MANIFEST, and how it starts the engine and relays its lines."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ms_shot_match", ROOT / "magicscript" / "shot_match.py")
ms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ms)

FAKE_ENGINE = r'''
import json, sys
print("args " + json.dumps(sys.argv[1:]))
print("deps first " + str(sys.path[0].endswith("pydeps")))
print("\n  Values and contact sheet: somewhere")
'''


class Ctx:
    def __init__(self, kit, **values):
        self.kit_dir, self.logs = kit, []
        self.vars = {"mode": "Preview", "layer": 3, "node_label": "Linear", "out_dir": "~/Documents/ShotMatch",
                     "keep_light": True, **values}
        self.lib = mock.Mock(expand=lambda p: p.replace("~", "HOME"))

    def log(self, text):
        self.logs.append(text)

    def progress(self, msg=None):
        pass

    def report(self, name, text):
        pass


class WrapperTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.kit = Path(self.tmp.name)
        (self.kit / "Colour").mkdir()
        self.script = self.kit / "Colour" / "shot_match.py"
        (self.kit / "Colour" / ms.ENGINE).write_text(FAKE_ENGINE)
        self.patches = [mock.patch.object(ms, "__file__", str(self.script)),
                        mock.patch.object(ms, "_resolve_python", lambda: sys.executable),
                        mock.patch.object(ms, "_ensure_packages", lambda *a: None)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def args(self, ctx):
        line = next(l for l in ctx.logs if l.startswith("args "))
        return __import__("json").loads(line[5:])

    def test_preview_passes_the_dialog_settings_and_relays_lines(self):
        ctx = Ctx(str(self.kit), layer=4, node_label="Lin", keep_light=False)
        self.assertEqual(ms.run(ctx), "Values and contact sheet: somewhere")
        self.assertEqual(self.args(ctx), ["--resolve", "preview", "--layer", "4", "--label", "Lin",
                                          "--out", "HOME/Documents/ShotMatch", "--no-keep-light"])
        self.assertIn("deps first True", ctx.logs)

    def test_only_apply_skips_the_engine_window_because_magicscript_asked(self):
        apply, collect = Ctx(str(self.kit), mode="Apply"), Ctx(str(self.kit), mode="Collect")
        ms.run(apply)
        ms.run(collect)
        self.assertIn("--confirmed", self.args(apply))
        self.assertEqual(self.args(collect)[:2], ["--resolve", "collect"])
        self.assertNotIn("--confirmed", self.args(collect))
        final = Ctx(str(self.kit), mode="Collect Final")
        ms.run(final)
        self.assertEqual(self.args(final)[:2], ["--resolve", "collect_final"])

    def test_engine_failure_is_raised_with_its_last_line(self):
        (self.kit / "Colour" / ms.ENGINE).write_text('print("Select the clips of the scene."); raise SystemExit(2)')
        with self.assertRaises(RuntimeError) as err:
            ms.run(Ctx(str(self.kit)))
        self.assertEqual(str(err.exception), "Select the clips of the scene.")

    def test_manifest_has_a_mode_choice_and_declares_its_packages(self):
        mode = next(v for v in ms.MANIFEST["variables"] if v["id"] == "mode")
        self.assertEqual(mode["choices"], ["Preview", "Apply", "Collect", "Collect Final"])
        self.assertTrue(ms.MANIFEST["destructive"])
        names = {t["name"] for t in ms.MANIFEST["third_party"]}
        self.assertTrue({"numpy", "tifffile", "pillow", "mediapipe"} <= names)


if __name__ == "__main__":
    unittest.main()
