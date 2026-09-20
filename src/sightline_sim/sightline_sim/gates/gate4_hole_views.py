"""Gate 4: can the wrist camera actually see the hole it is going to?

For every screw and every turn angle around the bit, put the tool at its approach
pose and read what the wrist camera has at the hole's own pixel. The camera sits
75 mm off the screw axis, so the turn angle decides what it looks past, and inside
the enclosure it often looks at a wall instead of the hole.

This is the table the look-around layer needs (docs/design.md section 8.4 lists "a clear
line from the wrist camera to the hole" as one of its checks).

Usage: MUJOCO_GL=egl python scripts/gate4_hole_views.py [out_dir]
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from sightline_sim.judge import PerceptionTruth  # noqa: E402
from sightline_planner import B0, Taught, ToolKinematics  # noqa: E402
from sightline_planner.b0 import BIT_DOWN, rot_z  # noqa: E402
from sightline_planner.perceive import find_hole  # noqa: E402
from sightline_sim import pose, script, sensors, station  # noqa: E402

HOLE_D = 0.0046
TURN_STEP = 15


def put(m, d, part: str, where) -> None:
    adr = m.joint(part + "_free").qposadr[0]
    d.qpos[adr:adr + 3] = where
    d.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "results/gate4")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    scene = station.build()
    m, d = scene.model, mujoco.MjData(scene.model)
    mujoco.mj_forward(m, d)
    script.parts_to_start(scene, d)
    points = scene.points
    kin = ToolKinematics(station.robot_only_model())
    base = np.array(points["box_home"], float)
    taught = Taught(part_pose=tuple(base),
                    rail_holes=[tuple(np.array(h, float) - base) for h in points["rail_holes"]],
                    cover_holes=[tuple(np.array(h, float) - base) for h in points["cover_holes"]],
                    feeder_pick=tuple(np.array(points["feeder_pick"], float)))
    b0 = B0.build(kin, taught, park_tip=(-0.45, 0.33, scene.worktop_z + 0.30))
    truth = PerceptionTruth(m, "eyes_cam", 640, 480)
    wrist = sensors.Camera(m, "wrist_cam", 640, 480)
    qadr = pose.robot_qadr(m)

    table = {}
    for screw in b0.screws:
        # the state of the cell when this screw is driven
        put(m, d, "part_base", points["box_home"])
        put(m, d, "part_rail", points["rail_home"])
        if screw.name.startswith("rail"):
            put(m, d, "part_cover", points["pick_cover"])      # still in its tote
        else:
            put(m, d, "part_cover", points["cover_home"])
        mujoco.mj_forward(m, d)
        row = {"reachable": 0, "hole_in_view": [], "detected": [], "blocked_by": {}}
        for psi in range(0, 360, TURN_STEP):
            R = rot_z(np.radians(psi)) @ BIT_DOWN
            above = np.asarray(screw.hole, float) + [0.0, 0.0, taught.screw_length + taught.approach]
            q, err = kin.ik_multi(above, R, np.array([0.0, -1.6, 1.6, -1.6, -1.57, 0.0]))
            if not np.isfinite(err) or err > 1e-4:
                continue
            row["reachable"] += 1
            d.qpos[qadr] = q
            mujoco.mj_forward(m, d)
            owner = truth.hole_owner(d, screw.hole)
            seen = owner.startswith("part_")
            if seen:
                row["hole_in_view"].append(psi)
            else:
                row["blocked_by"][owner] = row["blocked_by"].get(owner, 0) + 1
            spot = truth.hole_pixel(d, screw.hole)
            if spot is not None:
                u, v, dist, f = spot
                if find_hole(wrist.rgb(d), (u, v), HOLE_D * f / dist) is not None:
                    row["detected"].append(psi)
        table[screw.name] = row
        print(f"{screw.name:8s} reachable {row['reachable']:2d}/24, hole in view "
              f"{len(row['hole_in_view']):2d}, detected {len(row['detected']):2d}, "
              f"blocked by {row['blocked_by']}")

    result = {"gate": 4, "date": "2026-09-18",
              "note": "tool at its approach pose, 60 mm above the hole, one entry per 15 degree "
                      "turn around the bit. 'hole in view' means the hole's own pixel shows the "
                      "part, not the arm or the jig.",
              "per_screw": table,
              "wall_time_s": round(time.time() - t0, 1)}
    (out / "hole_views.yaml").write_text(yaml.safe_dump(result, sort_keys=False))
    wrist.close()
    truth.close()
    print(f"wrote {out / 'hole_views.yaml'} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
