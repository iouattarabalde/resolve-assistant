"""timeline_inventory.py against a fake Resolve timeline and hand-made inventories.

Run from the repo root:  python3 -m unittest tests.test_timeline_inventory -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import timeline_inventory as ti  # noqa: E402


class Media:
    def __init__(self, uid, props, meta=None):
        self.uid, self.props, self.meta = uid, props, meta or {}

    def GetUniqueId(self): return self.uid
    def GetClipProperty(self): return self.props
    def GetMetadata(self): return self.meta


class Item:
    def __init__(self, name, start, media):
        self.name, self.start, self.media = name, start, media
        self.GetSourceEndFrame = None   # like Resolve: a missing method comes back as None

    def GetName(self): return self.name
    def GetUniqueId(self): return f"ti-{self.name}-{self.start}"
    def GetStart(self): return self.start
    def GetEnd(self): return self.start + 100
    def GetSourceStartFrame(self): return 0
    def GetFlagList(self): return []
    def GetClipColor(self): return ""
    def GetMediaPoolItem(self): return self.media


class Timeline:
    def __init__(self, tracks): self.tracks = tracks
    def GetName(self): return "T"
    def GetSettings(self): return {"timelineFrameRate": "29.97 DF", "timelineDropFrameTimecode": "1"}
    def GetStartFrame(self): return 0
    def GetTrackCount(self, kind): return len(self.tracks)
    def GetItemListInTrack(self, kind, track): return self.tracks[track - 1]


def media(name, start, end, day="2026-08-06", **meta):
    return Media(name, {"File Name": name, "Start TC": start, "End TC": end, "FPS": "25", "Date Created": day}, meta)


def clip(name, start, end, track=1, index=0, day="2026-08-06"):
    return {"track": track, "index": index, "name": name, "id": name + str(index), "media_id": name,
            "props": {"File Name": name, "Start TC": start, "End TC": end, "FPS": "25", "Date Created": day},
            "metadata": {}}


class Reading(unittest.TestCase):
    def test_reads_every_track_and_survives_missing_getters(self):
        a = media("AGRI_CAMA_0043.MXF", "10:00:00:00", "10:05:00:00")
        generator = Item("Solid Color", 300, None)
        inv = ti.read_timeline(Timeline([[Item("a1", 0, a), Item("a2", 100, a)], [generator]]))
        self.assertEqual(inv["total"], 3)
        self.assertEqual([c["track"] for c in inv["clips"]], [1, 1, 2])
        self.assertIsNone(inv["clips"][0]["source_end"])
        self.assertEqual(inv["clips"][2]["props"], {})
        self.assertEqual(inv["clips"][0]["media_id"], "AGRI_CAMA_0043.MXF")

    def test_slices_for_short_calls(self):
        a = media("A.MXF", "10:00:00:00", "10:05:00:00")
        tl = Timeline([[Item(f"c{i}", i * 100, a) for i in range(5)]])
        part = ti.read_timeline(tl, first=2, count=2)
        self.assertEqual([c["name"] for c in part["clips"]], ["c2", "c3"])
        self.assertEqual(part["total"], 5)


class Parsing(unittest.TestCase):
    def test_timecode(self):
        self.assertAlmostEqual(ti.tc_seconds("01:00:10:12", 25), 3610.48)
        self.assertAlmostEqual(ti.tc_seconds("01:00:10;15", 30), 3610.5)
        self.assertIsNone(ti.tc_seconds("", 25))

    def test_camera_from_names_and_metadata(self):
        for name, cam in [("AGRI_20260806_CAMA_0043_S001.MXF", "A"), ("AGRI_20260806_CAM_B0090.MXF", "B"),
                          ("LFC_CAM A0121.MXF", "A"), ("agri_cam_c_24-09-20260002.MP4", "C")]:
            self.assertEqual(ti.camera_of(clip(name, "", "")), cam, name)
        tagged = clip("x.mov", "", "")
        tagged["metadata"]["Camera #"] = "b"
        self.assertEqual(ti.camera_of(tagged), "B")
        self.assertEqual(ti.camera_of(clip("DJI_20260806125241_0012_D.MOV", "", "")), "DJI")
        self.assertEqual(ti.camera_of(clip("C4025.MP4", "", "")), "C")
        self.assertEqual(ti.camera_of(clip("0001.mov", "", "")), "?")


class Scenes(unittest.TestCase):
    def inventory(self):
        return {"timeline": "T", "frame_rate": "25", "clips": [
            clip("CAMA_0001.MXF", "10:00:00:00", "10:04:00:00", index=0),
            clip("CAMB_0001.MXF", "10:02:00:00", "10:20:00:00", index=1),   # overlaps A: same scene
            clip("CAMA_0001.MXF", "10:00:00:00", "10:04:00:00", index=2),   # same recording again
            clip("CAMA_0002.MXF", "10:25:00:00", "10:26:00:00", index=3),   # 5 min after B ended: same scene
            clip("DJI_0001.MOV", "00:00:00:00", "00:00:30:00", index=4),    # no clock: neighbours decide
            clip("CAMA_0003.MXF", "14:00:00:00", "14:01:00:00", index=5),   # hours later: new scene
            clip("CAMA_0009.MXF", "10:01:00:00", "10:02:00:00", index=6, day="2026-08-07"),   # another day
        ]}

    def test_overlap_gap_day_and_neighbours(self):
        scene_map = ti.build_scene_map(self.inventory())
        self.assertEqual([c["scene"] for c in scene_map["clips"]], [1, 1, 1, 1, 1, 2, 3])
        self.assertEqual(scene_map["clips"][4]["placed_by"], "neighbours")
        self.assertEqual(scene_map["clips"][0]["recording"], scene_map["clips"][2]["recording"])
        self.assertEqual(len(scene_map["recordings"]), 6)

    def test_intercut_order_does_not_matter(self):
        inv = self.inventory()
        inv["clips"] = inv["clips"][::-1]
        scenes = {c["name"]: c["scene"] for c in ti.build_scene_map(inv)["clips"]}
        self.assertEqual(scenes["CAMA_0001.MXF"], scenes["CAMB_0001.MXF"])
        self.assertNotEqual(scenes["CAMA_0001.MXF"], scenes["CAMA_0003.MXF"])

    def test_setups_split_a_recording_by_source_distance(self):
        inv = {"timeline": "T", "frame_rate": "25", "clips": [clip("A.MXF", "10:00:00:00", "10:10:00:00", index=i)
                                                              for i in range(4)]}
        for c, (start, end) in zip(inv["clips"], [(0, 100), (120, 200), (5000, 5100), (150, 260)]):
            c["source_start"], c["source_end"] = start, end   # frames at 25 fps
        got = [c["setup"] for c in ti.build_scene_map(inv)["clips"]]
        self.assertEqual(got[0], got[1])        # 0.8 s apart
        self.assertEqual(got[1], got[3])        # overlapping
        self.assertNotEqual(got[0], got[2])     # more than 3 minutes further in the take
        self.assertEqual(len({c["setup"] for c in ti.build_scene_map(inv, setup_gap_s=1e9)["clips"]}), 1)

    def test_clockless_clip_with_no_neighbours_stands_alone(self):
        inv = {"timeline": "T", "frame_rate": "25", "clips": [clip("title.png", "", "")]}
        scene_map = ti.build_scene_map(inv)
        self.assertEqual(scene_map["clips"][0]["placed_by"], "alone")
        self.assertIn("1 scenes", ti.summary(scene_map))


if __name__ == "__main__":
    unittest.main()
