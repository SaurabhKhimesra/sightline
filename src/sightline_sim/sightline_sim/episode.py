"""What an episode of the cell needs beside the simulator: the static blind cells of
the furniture, the words of the robot's states for captions, the caption band, and
the calibration error draw. Shared by the gate scripts, the cell node and the videos.
"""
import mujoco
import numpy as np

from sightline_planner.perceive import pack, unpack
from . import pose, station, textures
from .data import park_pose

TITLES = {"B0": "B0: IGNORES HIM", "B2": "B2: KEEPS 10 CM FROM HIM", "B3": "B3: 10 CM AND OUT OF HIS VIEW",
          "B4": "B4: PLANS ON THE CAMERA GRID"}


def blind_cells(scene, cam_pos, voxel: float = 0.05, samples: int = 3000, jig=None) -> np.ndarray:
    """Cells the arm can reach, above the worktop, on the worker's side, that the eyes
    camera cannot see past the cell furniture. Static knowledge of the cell: a hand
    can hide there and the camera would not know, so while a person is in the cell
    the arm treats them as occupied. The worker is not in this sweep.

    jig is (centre, half size, top) of the jig and the box in it. The 8 cells inside it
    are left out: fingers in the open box come with a hand and wrist above it in plain
    view, and the grid covers 15 cm behind every seen surface plus the margin. Left
    in, they were live whenever he worked at the jig, and 22 cm from the root of the
    upper arm, so the arm could not turn toward its own feeder."""
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
            offs = np.linspace(-r, r, max(int(np.ceil(2 * r / voxel)), 1) + 1)
            grid = np.stack(np.meshgrid(offs, offs, offs, indexing="ij"), -1).reshape(-1, 3)
            keys.append(pack(np.floor((centre + grid[np.linalg.norm(grid, axis=1) <= r]) / voxel).astype(np.int64)))
    cells = (unpack(np.unique(np.concatenate(keys))) + 0.5) * voxel
    shared = cells[cells[:, 1] < -0.05]
    if jig is not None:
        centre, half, top = jig
        inside = (np.abs(shared[:, 0] - centre[0]) <= half[0]) & (np.abs(shared[:, 1] - centre[1]) <= half[1]) & \
                 (shared[:, 2] <= top)
        shared = shared[~inside]
    park = park_pose()
    d.qpos[qadr] = park
    mujoco.mj_forward(m, d)
    station_only = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    gid = np.zeros(1, np.int32)
    hidden = []
    for point in shared:
        vec = point - cam_pos
        dist = float(np.linalg.norm(vec))
        mujoco.mj_ray(m, d, cam_pos, vec / dist, station_only, 1, m.body("eyes").id, gid)
        hit = mujoco.mj_ray(m, d, cam_pos, vec / dist, station_only, 1, m.body("eyes").id, gid)
        if gid[0] >= 0 and hit < dist - 0.004:
            hidden.append(point)
    return np.array(hidden) if hidden else np.zeros((0, 3))


ROBOT_WORDS = {
    "wait": "WAITING FOR THE NEXT PART", "to_feeder": "GOING TO THE FEEDER",
    "pick": "AT THE FEEDER WITH A SCREW", "to_hole": "MOVING TO {s}", "servo": "LINING UP ON {s}",
    "descend": "GOING INTO {s}", "drive": "DRIVING {s}", "retract": "COMING OUT OF {s}",
    "yield": "WAITING CLEAR OF HIM", "park": "PARKED",
}


def captioned(frame: np.ndarray, lines, px: int = 3, alert: str = "") -> np.ndarray:
    """Dark band with lines of text over the top of a frame; a red border and a word
    at the bottom while something is wrong. What the judge saw, not a decoration."""
    img = frame.astype(np.float32) / 255.0
    band = len(lines) * 10 * px + 20
    img[:band] *= 0.30
    for i, (text, color) in enumerate(lines):
        textures.draw_text(img, text, 16, 12 + i * 10 * px, color, px=px)
    if alert:
        edge = 8
        img[:edge], img[-edge:], img[:, :edge], img[:, -edge:] = (0.85, 0.08, 0.06), (0.85, 0.08, 0.06), \
            (0.85, 0.08, 0.06), (0.85, 0.08, 0.06)
        h = 10 * px + 16
        img[-h - edge:-edge, edge:-edge] *= 0.30
        textures.draw_text(img, alert, 24, img.shape[0] - h - edge + 8, (1.0, 0.35, 0.3), px=px)
    return textures.to_uint8(img)


def rotation_error(rng, degrees: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.radians(degrees)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
