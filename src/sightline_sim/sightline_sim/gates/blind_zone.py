"""Where can the robot go that the camera cannot see?

A blind spot only matters if the robot can reach into it. This sweeps the robot
through its joint ranges to find every 5 cm cell it can occupy, then asks the eyes
camera whether it can see each of those cells past the cell furniture. What is left
is the space where a hand could be hidden and the robot could still arrive.

Usage: MUJOCO_GL=egl python scripts/blind_zone.py [out_dir] [samples]
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
from sightline_sim.data import park_pose  # noqa: E402

from sightline_planner.perceive import pack, unpack  # noqa: E402
from sightline_sim import pose, station  # noqa: E402

VOXEL = 0.05


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "results/gate4")
    samples = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    scene = station.build()
    m, d = scene.model, mujoco.MjData(scene.model)
    mujoco.mj_forward(m, d)
    qadr = pose.robot_qadr(m)
    coll = [g for g in range(m.ngeom)
            if m.body(m.geom_bodyid[g]).name.startswith(station.ROBOT_PREFIX)
            and m.geom_group[g] == station.GROUP_COLLISION]
    lo = np.array([m.jnt_range[m.joint(station.ROBOT_PREFIX + n).id][0] for n in
                   ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")])
    hi = np.array([m.jnt_range[m.joint(station.ROBOT_PREFIX + n).id][1] for n in
                   ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")])
    rng = np.random.default_rng(0)
    robot_set = set(coll)

    print(f"sweeping the arm through {samples} poses")
    keys = []
    kept = 0
    for i in range(samples):
        d.qpos[qadr] = lo + rng.random(6) * (hi - lo)
        mujoco.mj_forward(m, d)
        # a pose that drives the arm through the bench is not space the robot can use
        if any(int(d.contact.geom[c][0]) in robot_set or int(d.contact.geom[c][1]) in robot_set
               for c in range(d.ncon)):
            continue
        kept += 1
        for g in coll:
            centre = d.geom_xpos[g]
            r = float(m.geom_rbound[g])
            steps = max(int(np.ceil(2 * r / VOXEL)), 1)
            offs = np.linspace(-r, r, steps + 1)
            grid = np.stack(np.meshgrid(offs, offs, offs, indexing="ij"), -1).reshape(-1, 3)
            grid = grid[np.linalg.norm(grid, axis=1) <= r]
            keys.append(pack(np.floor((centre + grid) / VOXEL).astype(np.int64)))
    print(f"  {kept} of {samples} poses are clear of the cell")
    swept = unpack(np.unique(np.concatenate(keys)))
    centres = (swept + 0.5) * VOXEL
    print(f"  the arm can occupy {len(centres)} cells of {VOXEL * 100:.0f} cm "
          f"({len(centres) * VOXEL ** 3 * 1000:.0f} litres)")

    # park it again before asking what the camera can see: the arm is not furniture
    park = park_pose()
    d.qpos[qadr] = park
    mujoco.mj_forward(m, d)
    cam = np.asarray(scene.points["eyes_pos"], float)
    station_only = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)   # furniture, not the arm or the worker
    gid = np.zeros(1, np.int32)
    hidden, blockers = [], {}
    for point in centres:
        vec = point - cam
        dist = float(np.linalg.norm(vec))
        hit = mujoco.mj_ray(m, d, cam, vec / dist, station_only, 1, m.body("eyes").id, gid)
        if gid[0] >= 0 and hit < dist - 0.004:
            hidden.append(point)
            name = m.body(m.geom_bodyid[int(gid[0])]).name
            blockers[name] = blockers.get(name, 0) + 1
    hidden = np.array(hidden) if hidden else np.zeros((0, 3))
    share = 100.0 * len(hidden) / max(len(centres), 1)
    print(f"  of those, {len(hidden)} are hidden from the eyes camera by the cell itself ({share:.1f} %)")
    for name, count in sorted(blockers.items(), key=lambda kv: -kv[1]):
        print(f"     {name:16s} hides {count:5d} cells")

    result = {
        "date": "2026-09-18",
        "note": "cells the arm or tool can occupy, and which of them the eyes camera "
                "cannot see past the cell furniture. The worker is not in this test: "
                "this is the fixed blind zone, not the shadow a person casts.",
        "voxel_m": VOXEL, "joint_samples": samples, "poses_clear_of_the_cell": kept,
        "reachable_cells": len(centres),
        "reachable_litres": round(len(centres) * VOXEL ** 3 * 1000, 1),
        "hidden_cells": len(hidden),
        "hidden_pct_of_reachable": round(share, 2),
        "hidden_by": blockers,
    }
    if len(hidden):
        result["hidden_extent_m"] = {
            "x": [round(float(hidden[:, 0].min()), 3), round(float(hidden[:, 0].max()), 3)],
            "y": [round(float(hidden[:, 1].min()), 3), round(float(hidden[:, 1].max()), 3)],
            "z": [round(float(hidden[:, 2].min()), 3), round(float(hidden[:, 2].max()), 3)],
        }
    (out / "blind_zone.yaml").write_text(yaml.safe_dump(result, sort_keys=False))
    print(f"wrote {out / 'blind_zone.yaml'} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
