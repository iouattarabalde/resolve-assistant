#!/usr/bin/env python3
"""Shot match, test 2: where does TimelineItem.SetCDL() land, and in what units?

Run ONLY in a throwaway project. It changes the grade of one clip, the one under the
playhead, and Resolve may not let you undo it.

It writes one known CDL (Slope 1.50 1.00 0.50, Offset 0.0100 0 0, Power 1) with NodeIndex set
to the Linear node's index in layer 3. The README gives SetCDL no layer argument, so the
question is which node receives it: the Linear node of layer 3, or the node with the same
index in layer 1. Check both Primaries panels afterwards.
"""
from __future__ import annotations

import sys

LINEAR_LAYER = 3
LINEAR_LABEL = "Linear"
TEST_CDL = {"Slope": "1.5000 1.0000 0.5000", "Offset": "0.0100 0.0000 0.0000",
            "Power": "1.0000 1.0000 1.0000", "Saturation": "1.0000"}


def main() -> int:
    r = globals().get("resolve")
    if not r:
        try:
            import DaVinciResolveScript as dvr
            r = dvr.scriptapp("Resolve")
        except ImportError:
            r = None
    if not r:
        print("Run this from Workspace > Scripts inside Resolve.")
        return 2
    project = r.GetProjectManager().GetCurrentProject()
    timeline = project.GetCurrentTimeline() if project else None
    item = timeline.GetCurrentVideoItem() if timeline else None
    if not item:
        print("Open the throwaway project, park the playhead on a clip, then run this again.")
        return 2
    graph = item.GetNodeGraph(LINEAR_LAYER)
    hits = [i for i in range(1, (graph.GetNumNodes() or 0) + 1) if graph.GetNodeLabel(i) == LINEAR_LABEL] if graph else []
    if len(hits) != 1:
        print(f"{item.GetName()}: needs exactly one node labelled {LINEAR_LABEL} in layer {LINEAR_LAYER}.")
        return 2
    node = hits[0]
    layer1 = item.GetNodeGraph(1)
    print(f"Clip {item.GetName()}: Linear is node {node} of layer {LINEAR_LAYER}. "
          f"Layer 1 node {node} is labelled '{layer1.GetNodeLabel(node) if layer1 else '?'}'.")
    ok = item.SetCDL(dict(TEST_CDL, NodeIndex=str(node)))
    print(f"SetCDL returned {ok}. Wrote Slope {TEST_CDL['Slope']}, Offset {TEST_CDL['Offset']}.")
    print("Now screenshot the Primaries of layer 3's Linear node, and of node "
          f"{node} in layer 1.")
    return 0


if __name__ == "__main__":
    if "resolve" in globals():
        main()
    else:
        sys.exit(main())
