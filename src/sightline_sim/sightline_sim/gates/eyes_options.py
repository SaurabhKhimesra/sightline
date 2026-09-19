"""Render the eyes camera options side by side, with the numbers that matter for the choice.

Usage: MUJOCO_GL=egl python scripts/eyes_options.py [out_dir]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from sightline_sim import station  # noqa: E402
from scripts.gate1_stills import pose_scene  # noqa: E402


BODY_GROUPS = {
    "hands": ("lhand", "rhand", "lfingers", "rfingers", "lthumb"),
    "arms": ("lhumerus", "rhumerus", "lradius", "rradius", "lwrist", "rwrist"),
    "head": ("head", "upperneck"),
    "torso": ("thorax", "lowerback", "root"),
}


def _surface_points(m: mujoco.MjModel, d: mujoco.MjData, bodies, stride: int = 7) -> np.ndarray:
    """Points on the worker's visible surface (clothing meshes), world frame."""
    pts = []
    for name in bodies:
        bid = m.body("worker/" + name).id
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != bid or m.geom_group[g] != station.GROUP_WORKER:
                continue
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = m.geom_dataid[g]
                adr, n = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
                v = m.mesh_vert[adr:adr + n][::stride]
                pts.append(d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T)
            else:
                pts.append(d.geom_xpos[g][None, :])
    return np.vstack(pts) if pts else np.zeros((0, 3))


def visibility(scene: station.Scene, d: mujoco.MjData, bodies) -> dict:
    """Share of the worker's surface the camera sees, and what hides the rest.

    A ray runs from the camera to each surface point. If it stops early, the geom
    it hits is either the robot, the worker's own body, or the station.
    """
    m = scene.model
    cam = np.asarray(scene.points["eyes_pos"], float)
    pts = _surface_points(m, d, bodies)
    groups = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)
    eyes_body = m.body("eyes").id  # the camera housing sits on the ray start
    gid = np.zeros(1, np.int32)
    seen = robot = self_hidden = station_hidden = 0
    for p in pts:
        vec = p - cam
        dist = float(np.linalg.norm(vec))
        hit = mujoco.mj_ray(m, d, cam, vec / dist, groups, 1, eyes_body, gid)
        if gid[0] < 0 or hit >= dist - 0.004:
            seen += 1
            continue
        name = m.body(m.geom_bodyid[int(gid[0])]).name
        if name.startswith(station.ROBOT_PREFIX):
            robot += 1
        elif name.startswith("worker/"):
            self_hidden += 1
        else:
            station_hidden += 1
    n = max(len(pts), 1)
    return {"points": len(pts), "seen_pct": 100.0 * seen / n, "robot_pct": 100.0 * robot / n,
            "self_pct": 100.0 * self_hidden / n, "station_pct": 100.0 * station_hidden / n}


def hole_visibility(scene: station.Scene, d: mujoco.MjData) -> str:
    """How many of the four cover holes the camera can see, and what hides the rest."""
    m = scene.model
    cam = np.asarray(scene.points["eyes_pos"], float)
    groups = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)
    eyes_body = m.body("eyes").id
    gid = np.zeros(1, np.int32)
    seen, by = 0, []
    for hole in scene.points["cover_holes"]:
        vec = np.asarray(hole, float) - cam
        dist = float(np.linalg.norm(vec))
        hit = mujoco.mj_ray(m, d, cam, vec / dist, groups, 1, eyes_body, gid)
        if gid[0] < 0 or hit >= dist - 0.004:
            seen += 1
        else:
            name = m.body(m.geom_bodyid[int(gid[0])]).name
            by.append("robot" if name.startswith(station.ROBOT_PREFIX) else
                      "worker" if name.startswith("worker/") else "station")
    return f"{seen}/4" + (" hidden by " + ",".join(sorted(set(by))) if by else "")


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "results/gate1/eyes_options")
    out.mkdir(parents=True, exist_ok=True)
    names = sys.argv[2].split(",") if len(sys.argv) > 2 else ["top", "corner", "pole"]
    for name in names:
        layout = station.Layout(eyes_variant=name)
        scene = station.build(layout)
        d = mujoco.MjData(scene.model)
        pose_scene(scene, d)
        all_bodies = tuple(b for group in BODY_GROUPS.values() for b in group)
        overall = visibility(scene, d, all_bodies)
        hands = visibility(scene, d, BODY_GROUPS["hands"])
        holes = hole_visibility(scene, d)
        opt = mujoco.MjvOption()
        for g in range(6):
            opt.geomgroup[g] = 1 if g in (0, 1, 2) else 0
        with mujoco.Renderer(scene.model, 720, 1280) as r:
            r.update_scene(d, camera="eyes_cam", scene_option=opt)
            imageio.imwrite(out / f"eyes_{name}.png", r.render())
        with mujoco.Renderer(scene.model, 900, 1600) as r:
            r.update_scene(d, camera="hero", scene_option=opt)
            imageio.imwrite(out / f"mount_{name}.png", r.render())
        print(f"{name:7s} pitch {scene.points['eyes_pitch_deg']:4.1f} deg | camera {scene.points['eyes_pos'][2]:.2f} m up | "
              f"worker seen {overall['seen_pct']:5.1f} % "
              f"(robot hides {overall['robot_pct']:.1f} %, own body {overall['self_pct']:.1f} %, station {overall['station_pct']:.1f} %) | "
              f"hands seen {hands['seen_pct']:5.1f} % (robot hides {hands['robot_pct']:.1f} %) | "
              f"cover holes {holes}")


if __name__ == "__main__":
    main()
