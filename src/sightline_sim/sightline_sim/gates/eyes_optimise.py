"""Pick the eyes camera position with two things in view at once.

1. The shared space: every cell the arm can reach, above the worktop, on the
   worker's side of the bench. The camera should see as much of it as possible past
   the cell furniture. Whatever it cannot see is where a hand could hide.
2. Behind the robot: the space on the far side of the bench where a second person
   could walk up. The camera must keep that in its picture too.

The camera aims at the jig. Candidates are on a pole outside the cell footprint, at
heights a real pole can hold. Everything is a ray cast against the cell furniture,
with the worker absent: this is the fixed blind zone, not a person's shadow.

Usage: MUJOCO_GL=egl python scripts/eyes_optimise.py [out_dir]
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
from sightline_sim import geometry as G, pose, station  # noqa: E402

VOXEL = 0.05
FOVY = station.D455_FOVY
ASPECT = 640 / 480


def swept_cells(scene, samples: int = 4000) -> np.ndarray:
    """Cells the arm can occupy in collision free poses that stay above the worktop."""
    m, d = scene.model, mujoco.MjData(scene.model)
    mujoco.mj_forward(m, d)
    zw = scene.worktop_z
    qadr = pose.robot_qadr(m)
    coll = [g for g in range(m.ngeom)
            if m.body(m.geom_bodyid[g]).name.startswith(station.ROBOT_PREFIX)
            and m.geom_group[g] == station.GROUP_COLLISION]
    robot_set = set(coll)
    names = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
             "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
    lo = np.array([m.jnt_range[m.joint(station.ROBOT_PREFIX + n).id][0] for n in names])
    hi = np.array([m.jnt_range[m.joint(station.ROBOT_PREFIX + n).id][1] for n in names])
    rng = np.random.default_rng(0)
    keys = []
    for _ in range(samples):
        d.qpos[qadr] = lo + rng.random(6) * (hi - lo)
        mujoco.mj_forward(m, d)
        if any(int(d.contact.geom[c][0]) in robot_set or int(d.contact.geom[c][1]) in robot_set
               for c in range(d.ncon)):
            continue
        if min(d.geom_xpos[g][2] - m.geom_rbound[g] for g in coll) < zw:
            continue
        for g in coll:
            centre, r = d.geom_xpos[g], float(m.geom_rbound[g])
            offs = np.linspace(-r, r, max(int(np.ceil(2 * r / VOXEL)), 1) + 1)
            grid = np.stack(np.meshgrid(offs, offs, offs, indexing="ij"), -1).reshape(-1, 3)
            keys.append(pack(np.floor((centre + grid[np.linalg.norm(grid, axis=1) <= r]) / VOXEL).astype(np.int64)))
    return (unpack(np.unique(np.concatenate(keys))) + 0.5) * VOXEL


def in_frustum(points: np.ndarray, cam: np.ndarray, target: np.ndarray) -> np.ndarray:
    x, y = np.split(np.array(G.look_at_xyaxes(cam, target, (0.0, 0.0, 1.0))), 2)
    R = np.column_stack([x, y, np.cross(x, y)])
    rel = (points - cam) @ R
    depth = -rel[:, 2]
    ok = depth > 0.05
    tan_v = np.tan(np.radians(FOVY) / 2)
    ok &= np.abs(rel[:, 1]) <= depth * tan_v
    ok &= np.abs(rel[:, 0]) <= depth * tan_v * ASPECT
    return ok


def visible_share(m, d, cam: np.ndarray, target: np.ndarray, points: np.ndarray) -> float:
    """Share of points the camera has a clear line to, inside its own picture."""
    if len(points) == 0:
        return float("nan")
    framed = in_frustum(points, cam, target)
    station_only = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    gid = np.zeros(1, np.int32)
    seen = 0
    for point, ok in zip(points, framed):
        if not ok:
            continue
        vec = point - cam
        dist = float(np.linalg.norm(vec))
        hit = mujoco.mj_ray(m, d, cam, vec / dist, station_only, 1, m.body("eyes").id, gid)
        if gid[0] < 0 or hit >= dist - 0.004:
            seen += 1
    return 100.0 * seen / len(points)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "results/gate4")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    scene = station.build()
    m, d = scene.model, mujoco.MjData(scene.model)
    park = park_pose()
    d.qpos[pose.robot_qadr(m)] = park
    mujoco.mj_forward(m, d)
    zw = scene.worktop_z
    target = np.array([0.0, -0.06, zw + 0.04])          # the jig, as the pole variant aims

    cells = swept_cells(scene)
    shared = cells[cells[:, 1] < -0.05]
    print(f"arm can reach {len(cells)} cells above the worktop, {len(shared)} on the worker's side")

    # behind the robot: where a second person would stand and reach, at the far side
    xs = np.arange(-0.9, 0.95, 0.1)
    ys = np.arange(0.55, 1.6, 0.1)
    zs = np.arange(0.9, 1.9, 0.1)
    behind = np.array([(x, y, z) for x in xs for y in ys for z in zs])
    print(f"behind the robot: {len(behind)} sample points")

    now = np.asarray(scene.points["eyes_pos"], float)
    rows = []
    candidates = [tuple(now)]
    for x in (0.6, 0.9, 1.15, 1.4, 1.7):
        for y in (-0.6, -0.9, -1.2, -1.5, -1.8):
            for z in (2.9, 3.25, 3.6):
                candidates.append((x, y, z))
    for cand in candidates:
        cam = np.asarray(cand, float)
        shared_seen = visible_share(m, d, cam, target, shared)
        behind_seen = visible_share(m, d, cam, target, behind)
        pitch = float(np.degrees(np.arctan2(cam[2] - target[2], np.linalg.norm((cam - target)[:2]))))
        rows.append({"cam": [round(float(v), 2) for v in cam], "pitch_deg": round(pitch, 1),
                     "shared_seen_pct": round(shared_seen, 1), "behind_seen_pct": round(behind_seen, 1)})
    rows.sort(key=lambda r: -(r["shared_seen_pct"] + 0.5 * r["behind_seen_pct"]))
    print(f"\n{'camera':24s} {'pitch':>6s} {'shared seen':>12s} {'behind seen':>12s}")
    for r in rows[:12]:
        mark = "  <- the pole we have" if r["cam"] == [round(float(v), 2) for v in now] else ""
        print(f"{str(r['cam']):24s} {r['pitch_deg']:6.1f} {r['shared_seen_pct']:11.1f} % {r['behind_seen_pct']:11.1f} %{mark}")
    current = next(r for r in rows if r["cam"] == [round(float(v), 2) for v in now])
    print(f"\ncurrent pole: shared {current['shared_seen_pct']} %, behind {current['behind_seen_pct']} %")
    (out / "eyes_optimise.yaml").write_text(yaml.safe_dump(
        {"date": "2026-09-18", "aim_point": [round(float(v), 3) for v in target],
         "shared_cells": int(len(shared)), "behind_points": int(len(behind)),
         "current": current, "ranked": rows[:20]}, sort_keys=False))
    print(f"wrote {out / 'eyes_optimise.yaml'} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
