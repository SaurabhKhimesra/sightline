"""Gate 1 numbers: camera coverage over the work poses, reach per hole and turn angle, and a park pose.

Writes results/gate1/gate1.yaml and a contact sheet of the work poses.

Usage: MUJOCO_GL=egl python scripts/gate1_numbers.py [out_dir]
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from sightline_sim import geometry as G, measure, pose, station, worker  # noqa: E402

# Bit straight down, screw tip 8 mm above the surface (scene choice), so the tool tip target
# sits one screw length plus that clearance above the hole.
SCREW_LEN = 0.0156
HOVER = 0.008
TURN_STEP_DEG = 15

IK_SEEDS = [np.array([pan, lift, elbow, w1, w2, 0.0])
            for pan in np.linspace(-np.pi, np.pi, 8, endpoint=False)
            for lift in (-2.2, -1.4)
            for elbow in (-2.0, 2.0)
            for w1 in (-2.0, -0.8)
            for w2 in (-1.57, 1.57)]

PARK_TIP_TARGETS = [(-0.45, 0.33), (0.0, 0.36), (0.45, 0.33), (0.62, 0.22), (-0.62, 0.22)]
PARK_TIP_HEIGHTS = (0.30, 0.45)


def bit_down(psi_deg: float) -> np.ndarray:
    base = np.column_stack([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    return G.rot_z(np.radians(psi_deg)) @ base


def reach_at(scene: station.Scene, d: mujoco.MjData, tip, psi_deg: float):
    """Best IK solution for the tool tip at a turn angle, and whether it clears the station."""
    m = scene.model
    res = pose.robot_ik(m, d, "tool_tip", tip, bit_down(psi_deg), IK_SEEDS, iters=220)
    qadr = pose.robot_qadr(m)
    for err, q in res[:6]:
        if err > 1e-5:
            break
        d.qpos[qadr] = q
        hits = measure.station_collisions(m, d)
        if not hits:
            return True, True, q
    if res[0][0] <= 1e-5:
        return True, False, res[0][1]
    return False, False, None


def reach_table(scene: station.Scene, d: mujoco.MjData, targets: dict) -> dict:
    out = {}
    for name, point in targets.items():
        tip = np.asarray(point, float) + [0.0, 0.0, SCREW_LEN + HOVER]
        angles_reach, angles_clear = [], []
        for psi in range(0, 360, TURN_STEP_DEG):
            ok, clear, _ = reach_at(scene, d, tip, psi)
            if ok:
                angles_reach.append(psi)
            if clear:
                angles_clear.append(psi)
        n = 360 // TURN_STEP_DEG
        out[name] = {"turn_angles_tested": n, "reachable": len(angles_reach), "clear_of_station": len(angles_clear),
                     "clear_angles_deg": angles_clear}
        print(f"  {name:16s} reachable {len(angles_reach):2d}/{n}, clear of the station {len(angles_clear):2d}/{n}")
    return out


def park_candidates(scene: station.Scene, d: mujoco.MjData) -> list:
    """Collision-free robot poses parked away from the work, with the tool pointing down."""
    zw = scene.worktop_z
    out = []
    for (x, y) in PARK_TIP_TARGETS:
        for dz in PARK_TIP_HEIGHTS:
            for psi in (0, 90, 180, 270):
                ok, clear, q = reach_at(scene, d, (x, y, zw + dz), psi)
                if ok and clear:
                    out.append(((x, y, dz, psi), q))
                    break
    return out


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/gate1")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = {"date": "2026-09-18", "note": "gate 1, station. One worker per pose, robot parked unless stated."}
    # every RAY_STRIDE'th mesh vertex of the worker's clothing. A coarser grid once
    # reported 0.0 % for a hand that is 6.5 % visible, so the stride is part of the record.
    ray_stride, block_stride = 5, 9

    scene = station.build()
    m = scene.model
    d = mujoco.MjData(m)
    zw = scene.worktop_z
    cam = np.asarray(scene.points["eyes_pos"], float)
    results["heights_m"] = {"worker_stature": worker.WORKER_STATURE_M, "elbow": round(scene.points["elbow_height"], 4),
                            "worktop": round(zw, 4), "eyes_camera": round(float(cam[2]), 3),
                            "eyes_pitch_deg": round(scene.points["eyes_pitch_deg"], 1)}
    print(f"worktop {zw:.3f} m, elbow {scene.points['elbow_height']:.3f} m, "
          f"camera {cam[2]:.2f} m at {scene.points['eyes_pitch_deg']:.1f} deg")
    mujoco.mj_forward(m, d)
    frame = measure.camera_frame(m, d)  # a point counts as seen only if it is in shot

    # park the robot first: coverage should not depend on where the arm happens to be
    print("park candidates")
    parks = park_candidates(scene, d)
    print(f"  {len(parks)} collision free candidates")
    qadr = pose.robot_qadr(m)
    if parks:
        d.qpos[qadr] = parks[0][1]

    # worker poses: coverage from the eyes camera, robot parked
    poser = pose.WorkerPoser(m, d)
    coverage, worker_qpos = {}, {}
    print("work poses, robot parked at the first candidate")
    for wp in measure.work_poses(scene):
        errs = measure.apply_work_pose(scene, d, wp, poser)
        worker_qpos[wp.name] = d.qpos.copy()
        seen = measure.worker_seen(m, d, cam, stride=ray_stride, frame=frame)
        coverage[wp.name] = {
            "reach_err_mm": {k: round(v * 1000, 2) for k, v in errs.items()},
            "seen_pct": {g: round(seen[g]["seen_pct"], 1) for g in measure.BODY_GROUPS},
            "hands": {side: {"seen_pct": round(seen[f"{k}hand"]["seen_pct"], 1),
                             "points": seen[f"{k}hand"]["points"],
                             "hidden_by_own_body": seen[f"{k}hand"]["own_body"],
                             "hidden_by_station": seen[f"{k}hand"]["station"],
                             "out_of_frame": seen[f"{k}hand"]["out_of_frame"]}
                      for side, k in (("left", "l"), ("right", "r"))},
        }
        print(f"  {wp.name:34s} hands {seen['hands']['seen_pct']:5.1f} % "
              f"(left {seen['lhand_seen_pct']:5.1f}, right {seen['rhand_seen_pct']:5.1f}), "
              f"head {seen['head']['seen_pct']:5.1f} %, arms {seen['arms']['seen_pct']:5.1f} %")
    results["camera_coverage_pct"] = coverage

    # park pose: a candidate that hides nothing of the worker in any pose
    print("park pose against every work pose")
    park_scores = []
    for (spec, q) in parks:
        d.qpos[qadr] = q
        worst = 0.0
        for name, qpos in worker_qpos.items():
            d.qpos[:] = qpos
            d.qpos[qadr] = q
            mujoco.mj_forward(m, d)
            worst = max(worst, measure.robot_blocks(m, d, cam, stride=block_stride, frame=frame)["robot_pct"])
        park_scores.append((worst, spec, q))
        print(f"  tip at x {spec[0]:+.2f} y {spec[1]:+.2f} z+{spec[2]:.2f} turn {spec[3]:3d} deg: "
              f"worst blocked {worst:.2f} %")
    park_scores.sort(key=lambda s: s[0])
    if park_scores:
        best = park_scores[0]
        results["park_pose"] = {"tip_x": best[1][0], "tip_y": best[1][1], "tip_above_worktop": best[1][2],
                                "turn_deg": best[1][3], "worst_worker_blocked_pct": round(best[0], 3),
                                "joints_rad": [round(float(v), 4) for v in best[2]]}
        d.qpos[qadr] = best[2]

    # how much the robot hides while it works, for comparison with the park pose
    holes = scene.points["cover_holes"]
    tip = np.asarray(holes[3], float) + [0.0, 0.0, SCREW_LEN + HOVER]
    ok, clear, q_work = reach_at(scene, d, tip, 330)
    blocked = {}
    if q_work is not None:
        for name, qpos in worker_qpos.items():
            d.qpos[:] = qpos
            d.qpos[qadr] = q_work
            mujoco.mj_forward(m, d)
            blocked[name] = round(measure.robot_blocks(m, d, cam, stride=block_stride, frame=frame)["robot_pct"], 2)
        print("robot at the rear right cover hole hides, per pose:", blocked)
    results["robot_blocking_pct_at_one_hole"] = blocked

    # reach per hole and turn angle
    print("reach, cover on")
    targets = {f"cover_hole_{i + 1}": p for i, p in enumerate(scene.points["cover_holes"])}
    results["reach_cover"] = reach_table(scene, d, targets)

    # contact sheet of the work poses from the eyes camera
    opt = mujoco.MjvOption()
    for g in range(6):
        opt.geomgroup[g] = 1 if g in (0, 1, 2) else 0
    tiles = []
    if park_scores:
        d.qpos[qadr] = park_scores[0][2]
    with mujoco.Renderer(m, 360, 640) as r:
        for name, qpos in worker_qpos.items():
            d.qpos[:] = qpos
            if park_scores:
                d.qpos[qadr] = park_scores[0][2]
            mujoco.mj_forward(m, d)
            r.update_scene(d, camera="eyes_cam", scene_option=opt)
            tiles.append(r.render())
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    imageio.imwrite(out_dir / "work_poses.png", np.vstack(rows))
    del scene, m, d

    # rail screws are driven before the cover goes on, so measure those on the open box
    print("reach, cover off (rail screws)")
    scene2 = station.build(station.Layout(with_cover=False))
    d2 = mujoco.MjData(scene2.model)
    mujoco.mj_forward(scene2.model, d2)
    targets2 = {f"rail_hole_{i + 1}": p for i, p in enumerate(scene2.points["rail_holes"])}
    targets2["feeder_pick"] = tuple(np.asarray(scene2.points["feeder_pick"], float) + [0, 0, 0.004])
    results["reach_open_box"] = reach_table(scene2, d2, targets2)

    results["ray_stride"] = ray_stride
    results["robot_block_stride"] = block_stride
    (out_dir / "gate1.yaml").write_text(yaml.safe_dump(results, sort_keys=False))
    print(f"wrote {out_dir / 'gate1.yaml'} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
