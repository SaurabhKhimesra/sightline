"""Ground truth for gate 4, taken straight from the simulator.

What the camera could have seen of the person is the fair yardstick: perception is
not expected to find a hand that is behind the worker's own back. So the truth is
built from a perfect depth image labelled by segmentation, which gives, for every
pixel the camera can see, whether it is the person, the robot or the cell.

The planner never imports this.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .. import sensors, station
from .rules import SegmentationCamera

OTHER, PERSON, ROBOT = 0, 1, 2


class PerceptionTruth:
    """Per pixel labels, true person voxels, and the true pose of the box."""

    def __init__(self, m: mujoco.MjModel, camera: str = "eyes_cam", width: int = 640, height: int = 480,
                 voxel: float = 0.02):
        self.m = m
        self.voxel = voxel
        self.seg = SegmentationCamera(m, camera, width, height)
        self.depth = sensors.DepthCamera(m, camera, width, height, delay_frames=0)
        self.width, self.height = width, height
        f, cx, cy = self.depth.intrinsics()
        u = np.arange(width) - cx
        v = np.arange(height) - cy
        uu, vv = np.meshgrid(u, v)
        self.local = np.stack([uu / f, -vv / f, -np.ones_like(uu)], axis=-1).reshape(-1, 3)
        self.range_scale = np.linalg.norm(self.local, axis=1)
        self.local /= self.range_scale[:, None]
        self.wrist = None

    def labels(self, d: mujoco.MjData) -> np.ndarray:
        return self.seg.labels(d, hide_robot=False)

    def cloud(self, d: mujoco.MjData, camera: str = "eyes_cam"):
        """Every visible point with its label, from a perfect depth image."""
        depth = self.depth.raw(d).reshape(-1)
        label = self.labels(d).reshape(-1)
        keep = depth > 0
        pos, R = sensors.camera_pose(self.m, d, camera)
        dirs = self.local[keep] @ R.T
        pts = pos + dirs * (depth[keep] * self.range_scale[keep])[:, None]
        return pts, label[keep]

    def person_voxels(self, d: mujoco.MjData, camera: str = "eyes_cam") -> np.ndarray:
        """Only the person's pixels are unprojected: the other 290 000 cost 250 ms."""
        label = self.labels(d).reshape(-1)
        mine = np.nonzero(label == PERSON)[0]
        if mine.size == 0:
            return np.zeros((0, 3))
        depth = self.depth.raw(d).reshape(-1)[mine]
        good = depth > 0
        mine, depth = mine[good], depth[good]
        pos, R = sensors.camera_pose(self.m, d, camera)
        person = pos + (self.local[mine] @ R.T) * (depth * self.range_scale[mine])[:, None]
        if person.shape[0] == 0:
            return np.zeros((0, 3))
        keys = _unpack(np.unique(_pack(np.floor(person / self.voxel).astype(np.int64))))
        return (keys + 0.5) * self.voxel

    def box_pose(self, d: mujoco.MjData, jig_xy, reach: float = 0.15) -> tuple | None:
        """The assembly sitting in the jig: its centre, its top face and its yaw.

        Whatever is in the jig is what the camera sees, and during a cycle that is the
        base alone, then the base with a rail, then the box with its cover on. Reading
        the cover's own body instead follows it into its tote and reports the box as a
        metre away from where it is.
        """
        jig = np.asarray(jig_xy, float)
        base = self.m.body("part_base").id
        if float(np.linalg.norm(d.xpos[base][:2] - jig)) > reach:
            return None
        top = -np.inf
        for name in ("part_base", "part_rail", "part_cover"):
            try:
                body = self.m.body(name).id
            except KeyError:
                continue
            if float(np.linalg.norm(d.xpos[body][:2] - jig)) > reach:
                continue
            for g in range(self.m.ngeom):
                if self.m.geom_bodyid[g] == body and self.m.geom_group[g] == station.GROUP_COLLISION:
                    top = max(top, float(d.geom_xpos[g][2] + self.m.geom_size[g][2]))
        if not np.isfinite(top):
            return None
        quat = d.xquat[base]
        yaw = float(np.arctan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                               1 - 2 * (quat[2] ** 2 + quat[3] ** 2)))
        return float(d.xpos[base][0]), float(d.xpos[base][1]), top, yaw

    def hole_owner(self, d: mujoco.MjData, hole, camera: str = "wrist_cam") -> str:
        """What the wrist camera actually has at the hole's pixel.

        A ray from the camera is no use here: it starts on the tool and hits the
        tool's own shapes straight away. The picture is the ground truth, so this
        renders the segmentation and reads the label where the hole should be.
        Returns the body name, "out of frame", or "behind the camera".
        """
        if self.wrist is None:
            self.wrist = SegmentationCamera(self.m, camera, self.width, self.height)
            self.wrist_ids = np.arange(self.m.ngeom + 1)
        pos, R = sensors.camera_pose(self.m, d, camera)
        rel = R.T @ (np.asarray(hole, float) - pos)
        if rel[2] >= -1e-6:
            return "behind the camera"
        f = (self.height / 2) / np.tan(np.radians(float(self.m.cam_fovy[self.m.cam(camera).id])) / 2)
        u = int(round(self.width / 2 - 0.5 + f * (rel[0] / -rel[2])))
        v = int(round(self.height / 2 - 0.5 - f * (rel[1] / -rel[2])))
        if not (0 <= u < self.width and 0 <= v < self.height):
            return "out of frame"
        self.wrist.renderer.update_scene(d, camera=camera, scene_option=self.wrist.with_robot)
        for i in range(self.wrist.renderer.scene.ngeom):
            self.wrist.renderer.scene.geoms[i].segid = i
        seg = self.wrist.renderer.render()
        gidx = int(seg[v, u, 0])
        return self.m.body(self.m.geom_bodyid[gidx]).name if gidx >= 0 else "nothing"

    def hole_pixel(self, d: mujoco.MjData, hole, camera: str = "wrist_cam"):
        """Where the hole should appear in the wrist image, and how far away it is."""
        pos, R = sensors.camera_pose(self.m, d, camera)
        rel = R.T @ (np.asarray(hole, float) - pos)
        if rel[2] >= -1e-6:
            return None
        f = (self.height / 2) / np.tan(np.radians(float(self.m.cam_fovy[self.m.cam(camera).id])) / 2)
        return (self.width / 2 - 0.5 + f * (rel[0] / -rel[2]),
                self.height / 2 - 0.5 - f * (rel[1] / -rel[2]),
                float(np.linalg.norm(rel)), f)

    def close(self) -> None:
        self.seg.close()
        self.depth.close()
        if self.wrist is not None:
            self.wrist.close()


SHIFT = 1 << 20
MASK = (1 << 21) - 1


def _pack(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, np.int64) + SHIFT
    return (g[:, 0] << 42) | (g[:, 1] << 21) | g[:, 2]


def _unpack(keys: np.ndarray) -> np.ndarray:
    keys = np.asarray(keys, np.int64)
    return np.stack([(keys >> 42) - SHIFT, ((keys >> 21) & MASK) - SHIFT, (keys & MASK) - SHIFT], axis=-1)


def _grown(keys: np.ndarray) -> np.ndarray:
    """Every key and its 26 neighbours, packed and sorted."""
    grid = _unpack(keys)
    out = [grid + (dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    return np.unique(_pack(np.concatenate(out)))


def _share(probe: np.ndarray, sorted_keys: np.ndarray) -> float:
    if probe.size == 0:
        return float("nan")
    where = np.searchsorted(sorted_keys, probe)
    hit = where < sorted_keys.size
    hit[hit] = sorted_keys[where[hit]] == probe[hit]
    return float(hit.mean())


def match(found: np.ndarray, truth: np.ndarray, voxel: float) -> tuple[float, float]:
    """Recall and precision, counting a voxel as matched if it is within one voxel.

    One voxel of slack, because a 2 cm grid built from noisy depth will not land on
    the same cell as a grid built from a perfect one. Packed integer keys, because
    sets of tuples over thousands of voxels cost more than the perception they judge.
    """
    if truth.shape[0] == 0:
        return float("nan"), float("nan") if found.shape[0] else 1.0
    if found.shape[0] == 0:
        return 0.0, float("nan")
    fk = np.unique(_pack(np.floor(found / voxel).astype(np.int64)))
    tk = np.unique(_pack(np.floor(truth / voxel).astype(np.int64)))
    recall = _share(tk, _grown(fk))
    precision = _share(fk, _grown(tk))
    return recall, precision
