"""Gate 1 stills: build the station, pose worker and robot, render the film and robot cameras.

Usage: MUJOCO_GL=egl python scripts/gate1_stills.py [out_dir]
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

from sightline_sim import geometry as G, pose, station  # noqa: E402

SHOTS = {  # camera: (width, height)
    "hero": (1920, 1080),
    "wide": (1920, 1080),
    "side": (1600, 900),
    "top": (1600, 900),
    "closeup": (1600, 900),
    "eyes_cam": (1280, 720),
    "wrist_cam": (1280, 720),
}


def pose_scene(scene: station.Scene, d: mujoco.MjData) -> dict:
    m, L, zw = scene.model, scene.layout, scene.worktop_z
    info = {}
    P = pose.WorkerPoser(m, d)
    P.stand(L.worker_xy, np.pi / 2)
    P.lean(0.24)
    P.settle_on_floor()
    jx, jy = L.jig_xy
    holes = scene.points["cover_holes"]
    cover_top = holes[0][2]
    info["reach_left_err_m"] = P.reach("l", (jx - 0.062, jy - 0.050, cover_top + 0.024), finger_dir=(0.35, 1.0, 0.0))
    # the right arm's joint ranges are not mirrored from the left (CobotSafe human_motion.py), so the
    # right reach starts from CobotSafe's measured forward reach pose
    from sightline_sim.human import RIGHT_ARM_REACH
    P.seed_arm(RIGHT_ARM_REACH)
    info["reach_right_err_m"] = P.reach("r", (0.20, -0.30, zw + 0.024), finger_dir=(-0.3, 1.0, 0.0))
    target_hole = np.array(holes[3])
    tip = target_hole + [0.0, 0.0, 0.008 + 0.0156]
    info["look_err"] = P.look_at(tip)
    P.settle_on_floor()

    # Bit straight down. Turn angle psi about the bit: psi = 0 puts the flange toward the robot base,
    # which the arm cannot reach here; keep the reachable solutions and pick the one whose elbow and
    # wrist stay furthest from the worker's head.
    R0 = np.column_stack([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    seeds = [np.array([pan, lift, elbow, w1, w2, 0.0])
             for pan in np.linspace(-np.pi, np.pi, 8, endpoint=False)
             for lift in (-2.2, -1.4)
             for elbow in (-2.0, 2.0)
             for w1 in (-2.0, -0.8)
             for w2 in (-1.57, 1.57)]
    qadr = pose.robot_qadr(m)
    head = d.xpos[m.body("worker/head").id].copy()
    candidates = []
    for psi in np.radians(np.arange(0, 360, 30)):
        for err, q in pose.robot_ik(m, d, "tool_tip", tip, G.rot_z(psi) @ R0, seeds, iters=250)[:6]:
            if err > 1e-5 or np.any(np.abs(q[3:]) > 2 * np.pi):
                continue
            d.qpos[qadr] = q
            mujoco.mj_kinematics(m, d)
            elbow_p = d.xpos[m.body("ur5e/forearm_link").id]
            wrist_p = d.xpos[m.body("ur5e/wrist_2_link").id]
            shoulder_p = d.xpos[m.body("ur5e/upper_arm_link").id]
            clearance = min(np.linalg.norm(elbow_p - head), np.linalg.norm(wrist_p - head))
            elbow_up = elbow_p[2] > shoulder_p[2]
            candidates.append((not elbow_up, -clearance, float(np.degrees(psi)), q.copy()))
    candidates.sort(key=lambda c: (c[0], c[1]))
    info["robot_candidates"] = len(candidates)
    if candidates:
        d.qpos[qadr] = candidates[0][3]
        info["robot_psi_deg"] = candidates[0][2]
        info["robot_q"] = np.round(candidates[0][3], 3).tolist()
    mujoco.mj_forward(m, d)
    return info


def render_all(scene: station.Scene, d: mujoco.MjData, out: Path) -> None:
    m = scene.model
    opt = mujoco.MjvOption()
    for g in range(6):
        opt.geomgroup[g] = 1 if g in (0, 1, 2) else 0
    opt.sitegroup[:] = 0
    for cam, (w, h) in SHOTS.items():
        with mujoco.Renderer(m, h, w) as r:
            r.update_scene(d, camera=cam, scene_option=opt)
            r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = cam in ("hero", "wide", "side")
            img = r.render()
        imageio.imwrite(out / f"{cam}.png", img)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "results/gate1")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # the gate 1 stills pose the cell mid work, with two cover screws already in
    scene = station.build(station.Layout(driven_screws=2))
    t1 = time.time()
    d = mujoco.MjData(scene.model)
    info = pose_scene(scene, d)
    t2 = time.time()
    render_all(scene, d, out)
    t3 = time.time()
    print(f"worktop {scene.worktop_z:.3f} m, elbow {scene.points['elbow_height']:.3f} m, worker scale {scene.worker_scale:.4f}")
    print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in info.items()})
    print(f"build {t1 - t0:.1f} s, pose {t2 - t1:.1f} s, render {t3 - t2:.1f} s; ngeom {scene.model.ngeom}, nmesh {scene.model.nmesh}")


if __name__ == "__main__":
    main()
