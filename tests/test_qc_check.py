"""Synthetic fixtures for qc_check.py: one clean delivery, one with four known faults.

Run from the repo root:  python3 -m unittest tests.test_qc_check -v
Builds small ProRes 4444 clips with ffmpeg in a temp folder, then deletes them.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qc_check  # noqa: E402

FPS = 24
CLIP_FRAMES = 240          # 10 s per source clip
EDIT_IN, EDIT_OUT = 48, 192
HANDLES = 24
CLIPS = ["A001C001", "A001C002", "A001C003", "A001C004"]


def tc(frames: int) -> str:
    s, f = divmod(frames, FPS)
    return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}:{f:02d}"


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def source_start(i: int) -> int:
    return (3600 + i * 60) * FPS      # 01:00:00:00, 01:01:00:00, ...


def make_source(path: Path, i: int) -> None:
    ffmpeg("-f", "lavfi", "-i", f"testsrc2=s=320x180:r={FPS}:d={CLIP_FRAMES // FPS}",
           "-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuv444p10le",
           "-timecode", tc(source_start(i)), str(path))


def make_render(src: Path, dst: Path, i: int, first: int, last: int,
                extra_vf: str = "", rate: int | None = None) -> None:
    vf = f"trim=start_frame={first}:end_frame={last},setpts=PTS-STARTPTS"
    if extra_vf:
        vf += "," + extra_vf
    args = ["-i", str(src), "-vf", vf]
    if rate:
        args += ["-r", str(rate)]
    ffmpeg(*args, "-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuv444p10le",
           "-timecode", tc(source_start(i) + first), str(dst))


class QCFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.sources, cls.good, cls.broken = root / "sources", root / "good", root / "broken"
        for d in (cls.sources, cls.good, cls.broken):
            d.mkdir()

        edl = ["TITLE: QC TEST", "FCM: NON-DROP FRAME", ""]
        rec = 3600 * FPS
        for i, name in enumerate(CLIPS):
            make_source(cls.sources / f"{name}.mov", i)
            s_in, s_out = source_start(i) + EDIT_IN, source_start(i) + EDIT_OUT
            length = EDIT_OUT - EDIT_IN
            edl += [f"{i + 1:03d}  {name[:8]:<8} V     C        "
                    f"{tc(s_in)} {tc(s_out)} {tc(rec)} {tc(rec + length)}",
                    f"* FROM CLIP NAME: {name}.mov", ""]
            rec += length
        cls.edl = root / "cut.edl"
        cls.edl.write_text("\n".join(edl))

        first, last = EDIT_IN - HANDLES, EDIT_OUT + HANDLES
        for i, name in enumerate(CLIPS):
            make_render(cls.sources / f"{name}.mov", cls.good / f"{name}.mov", i, first, last)

        src = cls.sources
        # 1. wrong frame rate
        make_render(src / "A001C001.mov", cls.broken / "A001C001.mov", 0, first, last, rate=25)
        # 2. head handle 12 frames instead of 24
        make_render(src / "A001C002.mov", cls.broken / "A001C002.mov", 1, first + 12, last)
        # 3. renamed clip
        make_render(src / "A001C003.mov", cls.broken / "A001C003_v2.mov", 2, first, last)
        # 4. one black frame mid-clip
        make_render(src / "A001C004.mov", cls.broken / "A001C004.mov", 3, first, last,
                    extra_vf="drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='eq(n,96)'")

        cls.out = root / "reports"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_qc(self, renders: Path) -> tuple[int, str]:
        code = qc_check.main([str(renders), "--sources", str(self.sources),
                              "--edl", str(self.edl), "--handles", str(HANDLES),
                              "--out", str(self.out / renders.name)])
        report = next((self.out / renders.name).glob("QC_*[0-9].md")).read_text()
        return code, report

    def test_clean_delivery_passes(self):
        code, report = self.run_qc(self.good)
        self.assertEqual(code, 0, report)
        self.assertNotIn("FAIL", report)
        self.assertTrue(list((self.out / "good").glob("*_client.md")), "client summary missing")

    def test_four_faults_are_caught(self):
        code, report = self.run_qc(self.broken)
        self.assertEqual(code, 1)
        self.assertRegex(report, r"FAIL · fps · 25\.000, expected 24\.000")
        self.assertRegex(report, r"FAIL · handles · .*head short by 12 fr")
        self.assertRegex(report, r"FAIL · A001C003\.mov · no render with this name")
        self.assertRegex(report, r"WARN · A001C003_v2\.mov · render matches no source")
        self.assertRegex(report, r"FAIL · frames · black frame at 96")
        self.assertFalse(list((self.out / "broken").glob("*_client.md")),
                         "no client summary when a FAIL is present")

    def test_missing_ffmpeg_is_not_a_pass(self):
        import shutil
        real = shutil.which
        qc_check.shutil.which = lambda name: None
        try:
            self.assertEqual(qc_check.main([str(self.good)]), 2)
        finally:
            qc_check.shutil.which = real


if __name__ == "__main__":
    unittest.main()
