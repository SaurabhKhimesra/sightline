"""Measurements on a posed scene: what the eyes camera sees, what the robot blocks, reach and collisions.

Visibility is measured by ray casting: a ray from the camera to a point on the
worker's visible surface either arrives, or the geom it hits first says what is
in the way. That is cheaper and clearer than counting pixels, and it names the
blocker.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import station

BODY_GROUPS = {
    "hands": ("lhand", "rhand", "lfingers", "rfingers", "lthumb", "rthumb"),
    "arms": ("lhumerus", "rhumerus", "lradius", "rradius", "lwrist", "rwrist"),
    "head": ("head", "upperneck"),
    "torso": ("thorax", "lowerback", "root"),
}
ALL_BODIES = tuple(b for group in BODY_GROUPS.values() for b in group)

VISIBLE_GROUPS = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)  # env, worker, robot


def surface_points(m: mujoco.MjModel, d: mujoco.MjData, bodies, stride: int = 7) -> np.ndarray:
    """Points on the worker's visible surface (the clothing meshes), world frame."""
    pts = []
    for name in bodies:
        bid = m.body(station.HUMAN_PREFIX + name).id
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


def camera_frame(m: mujoco.MjModel, d: mujoco.MjData, name: str = "eyes_cam", aspect: float = 16 / 9):
    """(position, rotation, half angles) of a camera. MuJoCo cameras look along -z of their frame."""
    cid = m.cam(name).id
    fovy = float(m.cam_fovy[cid])
    half_v = np.radians(fovy / 2)
    half_h = np.arctan(np.tan(half_v) * aspect)
    return d.cam_xpos[cid].copy(), d.cam_xmat[cid].reshape(3, 3).copy(), (half_h, half_v)


def in_frame(point, cam_pos, cam_rot, half_angles) -> bool:
    """Is the point inside the camera's field of view?"""
    local = cam_rot.T @ (np.asarray(point, float) - cam_pos)
    depth = -local[2]
    if depth <= 1e-6:
        return False
    return abs(np.arctan2(local[0], depth)) <= half_angles[0] and abs(np.arctan2(local[1], depth)) <= half_angles[1]


def visibility(m: mujoco.MjModel, d: mujoco.MjData, cam, pts, exclude_body: int = -1, frame=None) -> dict:
    """Share of the given points the camera sees: inside its field of view and not blocked.

    A clear line of sight is not enough, so points outside the field of view are
    counted separately rather than as seen.
    """
    cam = np.asarray(cam, float)
    gid = np.zeros(1, np.int32)
    seen = robot = own = env = out = 0
    for p in pts:
        if frame is not None and not in_frame(p, *frame):
            out += 1
            continue
        vec = p - cam
        dist = float(np.linalg.norm(vec))
        hit = mujoco.mj_ray(m, d, cam, vec / dist, VISIBLE_GROUPS, 1, exclude_body, gid)
        if gid[0] < 0 or hit >= dist - 0.004:
            seen += 1
            continue
        name = m.body(m.geom_bodyid[int(gid[0])]).name
        if name.startswith(station.ROBOT_PREFIX):
            robot += 1
        elif name.startswith(station.HUMAN_PREFIX):
            own += 1
        else:
            env += 1
    n = max(len(pts), 1)
    return {"points": int(len(pts)), "seen": seen, "robot": robot, "own_body": own, "station": env,
            "out_of_frame": out, "seen_pct": 100.0 * seen / n, "robot_pct": 100.0 * robot / n,
            "out_of_frame_pct": 100.0 * out / n}


def eyes_body(m: mujoco.MjModel) -> int:
    return m.body("eyes").id


def worker_seen(m: mujoco.MjModel, d: mujoco.MjData, cam, stride: int = 7, frame=None) -> dict:
    """Visibility per body group, plus each hand on its own."""
    out = {}
    for group, bodies in BODY_GROUPS.items():
        out[group] = visibility(m, d, cam, surface_points(m, d, bodies, stride), eyes_body(m), frame)
    for side in ("l", "r"):
        # the thumb belongs to the hand. Leaving it out drops a hand from 201 sampled
        # points to 168 and moves every per-hand number by a few points.
        pts = surface_points(m, d, (f"{side}hand", f"{side}fingers", f"{side}thumb"), stride)
        out[f"{side}hand"] = visibility(m, d, cam, pts, eyes_body(m), frame)
        out[f"{side}hand_seen_pct"] = out[f"{side}hand"]["seen_pct"]
    return out


def robot_blocks(m: mujoco.MjModel, d: mujoco.MjData, cam, stride: int = 7, frame=None) -> dict:
    """How much of the worker the robot hides from the camera, over all body groups."""
    pts = surface_points(m, d, ALL_BODIES, stride)
    return visibility(m, d, cam, pts, eyes_body(m), frame)


def robot_geoms(m: mujoco.MjModel) -> set:
    return {g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(station.ROBOT_PREFIX)}


def station_collisions(m: mujoco.MjModel, d: mujoco.MjData) -> list:
    """Contacts between the robot (with its tool) and the fixed station, by geom name."""
    mujoco.mj_forward(m, d)
    robot = robot_geoms(m)
    out = []
    for i in range(d.ncon):
        g1, g2 = int(d.contact.geom[i][0]), int(d.contact.geom[i][1])
        n1 = m.body(m.geom_bodyid[g1]).name
        n2 = m.body(m.geom_bodyid[g2]).name
        in1, in2 = g1 in robot, g2 in robot
        if in1 == in2:
            continue
        other = n2 if in1 else n1
        if other.startswith(station.HUMAN_PREFIX):
            continue
        out.append((other, float(d.contact.dist[i])))
    return out


# ---------------------------------------------------------------- worker work poses

@dataclass
class WorkPose:
    """One thing the worker is doing: hand targets in world coordinates and how far he leans."""
    name: str
    left: tuple | None
    right: tuple | None
    lean: float = 0.24
    gaze: tuple | None = None
    left_fingers: tuple = (0.35, 1.0, 0.0)
    right_fingers: tuple = (-0.3, 1.0, 0.0)
    seeds: dict = field(default_factory=dict)


def work_poses(scene: station.Scene) -> list[WorkPose]:
    """A set of poses the worker holds during a cycle. Scene choices, meant to cover the station."""
    L = scene.layout
    zw = scene.worktop_z
    jx, jy = L.jig_xy
    holes = scene.points["cover_holes"]
    cover = holes[0][2]
    tray = (L.tray_xy[0], L.tray_xy[1], zw + 0.10)
    prep = (L.prep_xy[0], L.prep_xy[1], zw + 0.04)
    rack = (-0.92, -0.30, zw + 0.10)
    return [
        WorkPose("both hands on the cover", (jx - 0.062, jy - 0.050, cover + 0.024),
                 (jx + 0.070, jy - 0.055, cover + 0.024), gaze=(jx, jy, cover)),
        WorkPose("left hand on the cover", (jx - 0.062, jy - 0.050, cover + 0.024),
                 (0.20, -0.30, zw + 0.024), gaze=(jx + 0.09, jy + 0.07, cover)),
        WorkPose("hand beside the next hole", (jx + 0.058, jy + 0.030, cover + 0.024),
                 (0.20, -0.30, zw + 0.024), gaze=(jx + 0.09, jy + 0.07, cover)),
        WorkPose("reaching across the jig", (jx + 0.12, jy + 0.10, zw + 0.10),
                 (0.22, -0.28, zw + 0.024), lean=0.34, gaze=(jx + 0.12, jy + 0.12, zw + 0.08)),
        WorkPose("leaning in to look", (jx - 0.10, jy - 0.11, zw + 0.03),
                 (jx + 0.12, jy - 0.11, zw + 0.03), lean=0.46, gaze=(jx, jy, cover)),
        WorkPose("taking a part from the rack", rack, (0.16, -0.30, zw + 0.024), lean=0.18,
                 left_fingers=(-1.0, 0.2, 0.0), gaze=rack),
        WorkPose("putting a box in the tray", (jx - 0.10, jy - 0.06, zw + 0.12), tray, lean=0.22,
                 right_fingers=(0.6, 0.6, 0.0), gaze=tray),
        WorkPose("clipping blocks in the prep area", (prep[0] - 0.09, prep[1] - 0.02, prep[2] + 0.02),
                 (prep[0] + 0.07, prep[1] + 0.01, prep[2] + 0.03), lean=0.30, gaze=prep),
    ]


def apply_work_pose(scene: station.Scene, d: mujoco.MjData, pose: WorkPose, poser) -> dict:
    """Stand the worker, lean, reach both hands, aim the head. Returns the reach errors."""
    L = scene.layout
    poser.stand(L.worker_xy, np.pi / 2)
    poser.lean(pose.lean)
    poser.settle_on_floor()
    out = {}
    if pose.left is not None:
        out["left_err_m"] = poser.reach("l", pose.left, finger_dir=pose.left_fingers)
    if pose.right is not None:
        from sightline_sim.human import RIGHT_ARM_REACH
        poser.seed_arm(RIGHT_ARM_REACH)
        out["right_err_m"] = poser.reach("r", pose.right, finger_dir=pose.right_fingers)
    if pose.gaze is not None:
        poser.look_at(pose.gaze)
    poser.settle_on_floor()
    mujoco.mj_forward(scene.model, d)
    return out
