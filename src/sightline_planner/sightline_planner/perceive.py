"""What the robot works out about the cell from its two cameras.

SPEC.md section 8.3. Depth to points, background out, its own arm out, the box out,
and whatever is left is treated as the person. Behind the person, the space the
camera cannot see is treated as occupied too, which is the depth space idea of
Flacco et al. 2012.

Nothing here knows there is a person. It knows there is something that is not the
station, not the robot and not the box, and that is enough: a loose part left on the
bench counts as person, which is the conservative way round.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

VOXEL_M = 0.02           # person voxel size, SPEC.md section 8.3 (scene choice)
UNSEEN_DEPTH_M = 0.30    # how far behind a person point counts as occupied (scene choice)
BACKGROUND_MARGIN_M = 0.04   # least a point must beat the empty station by (scene choice)
BACKGROUND_SIGMAS = 3.0      # and at least this many standard deviations of depth noise
ROBOT_MARGIN_M = 0.05    # points within this of the robot's own shape are its own (scene choice)
MIN_POINTS_PER_VOXEL = 2  # one stray point is noise (scene choice)
JIG_YAW_DEG = 10.0       # how far out of square the jig can hold a part (scene choice)
# The arm and the parts are grown by this many pixels before subtracting (scene choice).
# Two was not enough: the lateral noise hands a pixel 3 to 4 px outside the box's edge
# the box's own depth, 100 mm nearer than the worktop behind it
SILHOUETTE_PX = 4
LATERAL_REACH_PX = 3     # how far sideways the camera's lateral noise can fetch a sample from (scene choice)
# A return this far in front of the robot's own rendered surface is still the robot: at
# a link's edge the camera's lateral noise samples the link's front face, up to a link
# radius (60 mm) nearer than the edge, and the depth noise adds 12 mm at 3 m. Two such
# pixels at the tool's edge, 70 mm in front of the render, read as a hand touching the
# arm and sent it backing away from itself (scene choice)
ROBOT_DEPTH_TOL_M = 0.08


@dataclass
class Calibration:
    """Where the planner believes the eyes camera is, and what it believes it sees.

    A cell knows this from its own calibration, never perfectly. The error is a
    scene choice and is stated in MODEL_NOTES.
    """
    pos: np.ndarray
    R: np.ndarray
    width: int
    height: int
    fovy_deg: float

    def focal(self) -> float:
        return (self.height / 2) / np.tan(np.radians(self.fovy_deg) / 2)

    def rays(self) -> np.ndarray:
        """Unit direction of every pixel, in the world frame. Built once."""
        f = self.focal()
        u = np.arange(self.width) + 0.5 - self.width / 2
        v = np.arange(self.height) + 0.5 - self.height / 2
        uu, vv = np.meshgrid(u, v)
        dirs = np.stack([uu / f, -vv / f, -np.ones_like(uu)], axis=-1)
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
        return dirs @ self.R.T

    def axis_cos(self) -> np.ndarray:
        """Cosine between each pixel ray and the camera axis, to turn depth into range."""
        f = self.focal()
        u = np.arange(self.width) + 0.5 - self.width / 2
        v = np.arange(self.height) + 0.5 - self.height / 2
        uu, vv = np.meshgrid(u, v)
        return 1.0 / np.sqrt(1.0 + (uu / f) ** 2 + (vv / f) ** 2)



class KnownWorld:
    """A depth image of what the planner already knows is there: its own arm at its
    current joint angles and the parts at their fitted pose, seen from where it
    believes the eyes camera is. SPEC.md section 8.3, steps 3 and 4.

    Anything clearly in front of this, and of the empty cell, is not explained by
    anything the planner knows, so it is the person. Deleting every point within
    5 cm of the arm and everything in the jig volume instead left the planner blind
    exactly where the arm meets a hand: 36 to 55 real person voxels within 10 cm of
    the arm, none perceived.
    """

    PARTS = ("wp_base", "wp_rail", "wp_cover")

    def __init__(self, model: mujoco.MjModel, calib: Calibration):
        self.m = model
        self.d = mujoco.MjData(model)
        model.vis.global_.fovy = calib.fovy_deg          # the free camera takes its field of view here
        self.renderer = mujoco.Renderer(model, calib.height, calib.width)
        self.renderer.enable_depth_rendering()
        self.opt = mujoco.MjvOption()
        for g in range(6):
            self.opt.geomgroup[g] = 1 if g in (0, 1, 2) else 0
        forward = -np.asarray(calib.R, float)[:, 2]
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance = 1.0
        self.cam.lookat[:] = np.asarray(calib.pos, float) + forward
        self.cam.azimuth = float(np.degrees(np.arctan2(forward[1], forward[0])))
        self.cam.elevation = float(np.degrees(np.arcsin(np.clip(forward[2], -1.0, 1.0))))
        self.mocap = {name: int(model.body_mocapid[model.body(name).id]) for name in self.PARTS}

    def depth(self, q: np.ndarray, parts: dict) -> np.ndarray:
        """parts maps a part name to its origin in the cell, or None if it is not there."""
        self.d.qpos[:6] = q
        for name, slot in self.mocap.items():
            where = parts.get(name)
            self.d.mocap_pos[slot] = (0.0, 0.0, -10.0) if where is None else where
            self.d.mocap_quat[slot] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_kinematics(self.m, self.d)
        self.renderer.update_scene(self.d, camera=self.cam, scene_option=self.opt)
        z = np.asarray(self.renderer.render(), dtype=np.float64)
        z[z > 20.0] = np.inf                              # nothing of ours on this pixel
        return z

    def close(self) -> None:
        self.renderer.close()

@dataclass
class PersonModel:
    """Everything the planner thinks is in the way, at one moment."""
    t: float
    voxels: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))      # centres, m
    velocity: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))    # m/s
    unseen: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))      # behind the person
    points: int = 0
    box_pose: tuple | None = None      # x, y, top z, yaw, this frame
    box_pose_held: tuple | None = None  # the last one that passed, which the planner uses
    box_age_s: float = 0.0
    speed_max: float = 0.0
    person_px: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))  # pixel index
    person_z: np.ndarray = field(default_factory=lambda: np.zeros(0))                  # depth along the axis, m


class Perception:
    """The eyes pipeline, one frame at a time."""

    def __init__(self, calib: Calibration, kin, taught, voxel: float = VOXEL_M,
                 unseen_depth: float = UNSEEN_DEPTH_M, known: "KnownWorld | None" = None):
        self.calib = calib
        self.kin = kin
        self.taught = taught
        self.known = known
        self.surface: np.ndarray | None = None   # nearest known surface per pixel, this frame
        self.voxel = voxel
        self.unseen_depth = unseen_depth
        self.background: np.ndarray | None = None
        self.background_gate: np.ndarray | None = None
        self.rays = calib.rays().reshape(-1, 3)
        self.range_scale = (1.0 / calib.axis_cos()).reshape(-1)
        self.prev_grid: np.ndarray | None = None
        self.prev_t: float | None = None
        self.last_box: tuple | None = None
        self.last_box_t: float | None = None
        self.robot_shapes = self._robot_shapes()

    # ------------------------------------------------------------------ setup
    def _robot_shapes(self) -> list:
        """The robot's own collision shapes, from its own model, for self filtering."""
        m = self.kin.m
        out = []
        for g in range(m.ngeom):
            if m.geom_group[g] != 3:
                continue
            out.append((g, int(m.geom_type[g]), m.geom_size[g].copy()))
        return out

    def set_background(self, depth: np.ndarray) -> None:
        """A depth map of the empty station, taken once at the start.

        How far in front of it a point has to be to count is set per pixel: three
        standard deviations of the camera's noise at that distance, plus, at a depth
        edge, how much nearer anything within 3 px is. The lateral noise hands a pixel a
        neighbour's depth, so beside the jig plate the worktop reads 16 mm nearer, and
        with two sigma of axial noise on top it crossed a fixed 40 mm gate once in a few
        hundred frames: two such pixels made a person cell, and the robot backed away
        from a corner of the jig.
        """
        self.background = np.where(depth > 0, depth, np.inf)
        sigma = 0.0012 + 0.0019 * (np.clip(self.background, 0.4, 2.75) - 0.4) ** 2
        lateral = self.background - _min_filter(self.background, LATERAL_REACH_PX)
        gate = np.maximum(BACKGROUND_MARGIN_M, BACKGROUND_SIGMAS * sigma + np.where(np.isfinite(lateral), lateral, 0.0))
        self.background_gate = (self.background - gate).reshape(-1)

    # ------------------------------------------------------------------ the frame
    def part_origins(self, present) -> dict:
        """Where each part the planner believes is in the jig sits, from the last box
        fit if there is one and the jig's nominal pose if not. Height from the jig."""
        base = np.asarray(self.taught.part_pose, float).copy()
        if self.last_box is not None:
            base[0], base[1] = self.last_box[0], self.last_box[1]
        dz = {"wp_base": 0.0, "wp_rail": self.taught.rail_origin_dz, "wp_cover": self.taught.cover_origin_dz}
        return {name: base + [0.0, 0.0, dz[name]] for name in present if name in dz}

    def update(self, depth: np.ndarray, q: np.ndarray, t: float, parts_present=(),
               q_at_capture: np.ndarray | None = None) -> PersonModel:
        """parts_present names the parts the planner's own task state says are in the jig.

        q_at_capture is where the arm was when the picture was taken. The camera is a
        frame late, so subtracting the arm where it is now leaves a band along every
        moving link that reads as a person hugging the arm: the robot froze on its own
        ghost for 18 s. The planner knows its own joint history, so it compares like
        with like.
        """
        q_seen = np.asarray(q if q_at_capture is None else q_at_capture, float)
        flat = depth.reshape(-1)
        keep = flat > 0
        if self.background_gate is not None:
            keep &= flat < self.background_gate
        idx = np.nonzero(keep)[0]
        if idx.size == 0:
            self.prev_grid, self.prev_t = None, t
            return PersonModel(t=t)
        rng = flat[idx] * self.range_scale[idx]
        pts = self.calib.pos + self.rays[idx] * rng[:, None]

        box_pose, box_mask = self._box(pts)
        if box_pose is not None:
            self.last_box, self.last_box_t = box_pose, t
        if self.known is not None:
            # what the planner knows is there: its own arm and the parts in the jig
            expected = self.known.depth(q_seen, self.part_origins(parts_present))
            expected = grow_near(expected, SILHOUETTE_PX).reshape(-1)
            ours = expected[idx]
            gate = np.maximum(BACKGROUND_MARGIN_M, BACKGROUND_SIGMAS * (
                0.0012 + 0.0019 * (np.clip(ours, 0.4, 2.75) - 0.4) ** 2))
            explained = flat[idx] >= ours - gate
            # the arm alone, with the looser tolerance its own edges need
            arm = grow_near(self.known.depth(q_seen, {}), SILHOUETTE_PX).reshape(-1)[idx]
            explained |= flat[idx] >= arm - ROBOT_DEPTH_TOL_M
            pts = pts[~explained]
            fg_px, fg_z = idx[~explained], flat[idx][~explained]
            self.surface = np.minimum(self.background.reshape(-1) if self.background is not None
                                      else np.full_like(expected, np.inf), expected)
        else:
            # the older filters: kept for comparison, blind next to the arm and over the box
            fg_px, fg_z = np.zeros(0, dtype=np.int64), np.zeros(0)
            pts = pts[~box_mask]
            if pts.size:
                pts = pts[~self._is_robot(pts, q)]

        model = PersonModel(t=t, points=int(pts.shape[0]), box_pose=box_pose,
                            person_px=fg_px, person_z=fg_z,
                            box_pose_held=self.last_box,
                            box_age_s=0.0 if self.last_box_t is None else t - self.last_box_t)
        if pts.shape[0] == 0:
            self.prev_grid, self.prev_t = None, t
            return model

        grid, centres = self._voxelize(pts)
        model.voxels = centres
        model.velocity, model.speed_max = self._track(grid, centres, t)
        model.unseen = self._unseen(centres)
        self.prev_grid, self.prev_t = grid, t
        return model

    # ------------------------------------------------------------------ pieces
    def _box(self, pts: np.ndarray):
        """Points inside the taught jig volume are the workpiece. Fit its top face.

        The fit looks for the flat face with the most points, not the highest ones:
        the worker's hands are usually on top of the cover, and taking the topmost
        points put the fit 50 mm out and the yaw 80 degrees out. It is coarse on
        purpose, a few mm, and the wrist camera refines it hole by hole.
        """
        base = np.asarray(self.taught.part_pose, float)
        half = np.array([0.13, 0.11])
        inside = (np.abs(pts[:, 0] - base[0]) < half[0]) & (np.abs(pts[:, 1] - base[1]) < half[1]) \
            & (pts[:, 2] > base[2] + 0.02) & (pts[:, 2] < base[2] + 0.25)
        if inside.sum() < 80:
            return None, inside
        box = pts[inside]
        bins = np.arange(base[2] + 0.02, base[2] + 0.25, 0.005)
        counts, _ = np.histogram(box[:, 2], bins=bins)
        level = float(bins[int(np.argmax(counts))] + 0.0025)
        # the jig holds the part at a known height, so a face well above it is something
        # lying on the cover, usually a hand
        if abs(level - (base[2] + self.taught.cover_top)) > 0.025:
            return None, inside
        face = box[np.abs(box[:, 2] - level) < 0.008]
        if face.shape[0] < 60:
            return None, inside
        rel = face[:, :2] - np.median(face[:, :2], axis=0)
        want = sorted(self.taught.cover_size, reverse=True)
        # Turn the known rectangle until it fits what is on show. Taking the principal
        # axis instead flips by up to 35 degrees as soon as part of the face is hidden.
        # The search covers the jig's own tolerance only: from 3.25 m the cover is about
        # 34 by 26 pixels, which does not pin a rotation, and a free search runs to its
        # own limit. What this fit contributes is the centre and the height.
        best = None
        for angle in np.radians(np.arange(-JIG_YAW_DEG, JIG_YAW_DEG + 0.1, 1.0)):
            axis = np.array([np.cos(angle), np.sin(angle)])
            across_axis = np.array([-axis[1], axis[0]])
            lo_a, hi_a = np.percentile(rel @ axis, [2, 98])
            lo_b, hi_b = np.percentile(rel @ across_axis, [2, 98])
            score = abs((hi_a - lo_a) - want[0]) + abs((hi_b - lo_b) - want[1])
            if best is None or score < best[0]:
                best = (score, angle, axis, across_axis, (lo_a + hi_a) / 2, (lo_b + hi_b) / 2)
        score, angle, axis, across_axis, mid_a, mid_b = best
        if score > 0.030:                      # nothing that size is on show
            return None, inside
        mid = np.median(face[:, :2], axis=0) + axis * mid_a + across_axis * mid_b
        # The angle the search lands on carries no information at this range: it runs to
        # whichever end of the jig's tolerance fits the noise. The jig sets the yaw.
        return (float(mid[0]), float(mid[1]), float(np.median(face[:, 2])), float(angle)), inside

    def _is_robot(self, pts: np.ndarray, q: np.ndarray) -> np.ndarray:
        """True for points that are the robot's own arm or tool.

        A real cell does this from its URDF and joint angles. Here the same model
        the planner plans with says where its shapes are.
        """
        m, d = self.kin.m, self.kin.d
        d.qpos[:self.kin.nq] = q
        mujoco.mj_kinematics(m, d)
        mine = np.zeros(pts.shape[0], dtype=bool)
        for g, gtype, size in self.robot_shapes:
            p, R = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
            local = (pts - p) @ R
            if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                axial = np.clip(local[:, 2], -size[1], size[1])
                near = np.linalg.norm(local - np.stack([np.zeros_like(axial), np.zeros_like(axial), axial], -1), axis=1)
                mine |= near < size[0] + ROBOT_MARGIN_M
            elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                radial = np.hypot(local[:, 0], local[:, 1]) - size[0]
                axial = np.abs(local[:, 2]) - size[1]
                mine |= np.hypot(np.maximum(radial, 0), np.maximum(axial, 0)) < ROBOT_MARGIN_M
            elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
                out = np.maximum(np.abs(local) - size[:3], 0.0)
                mine |= np.linalg.norm(out, axis=1) < ROBOT_MARGIN_M
            else:
                mine |= np.linalg.norm(local, axis=1) < float(size[0]) + ROBOT_MARGIN_M
        return mine

    def _voxelize(self, pts: np.ndarray):
        grid = np.floor(pts / self.voxel).astype(np.int64)
        packed, counts = np.unique(pack(grid), return_counts=True)
        keys = unpack(packed[counts >= MIN_POINTS_PER_VOXEL])
        return keys, (keys + 0.5) * self.voxel

    @staticmethod
    def _pack(grid: np.ndarray) -> np.ndarray:
        """Three voxel indices into one sortable integer, so lookups can be vectorised."""
        return pack(grid)

    def _track(self, grid: np.ndarray, centres: np.ndarray, t: float):
        """Velocity from the nearest voxel of the frame before, within one cell."""
        if self.prev_grid is None or self.prev_t is None or t <= self.prev_t or self.prev_grid.size == 0:
            return np.zeros_like(centres), 0.0
        dt = t - self.prev_t
        packed = np.sort(self._pack(self.prev_grid))
        best = np.full(centres.shape[0], np.inf)
        match = np.zeros_like(centres)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    probe = self._pack(grid + np.array([dx, dy, dz]))
                    where = np.searchsorted(packed, probe)
                    hit = (where < packed.size)
                    hit[hit] = packed[where[hit]] == probe[hit]
                    if not hit.any():
                        continue
                    other = centres[hit] - np.array([dx, dy, dz]) * self.voxel
                    dist = np.linalg.norm(other - centres[hit], axis=1)
                    better = dist < best[hit]
                    rows = np.nonzero(hit)[0][better]
                    best[rows] = dist[better]
                    match[rows] = other[better]
        moved = np.isfinite(best)
        vel = np.zeros_like(centres)
        vel[moved] = (centres[moved] - match[moved]) / dt
        speeds = np.linalg.norm(vel, axis=1)
        return vel, float(speeds.max()) if speeds.size else 0.0

    def _unseen(self, centres: np.ndarray) -> np.ndarray:
        """Voxels behind the person along the camera ray, which the camera cannot see.

        They count as occupied for R1: a hand could be there and the camera would not
        know. Flacco et al. 2012 call this depth space.
        """
        if centres.shape[0] == 0:
            return np.zeros((0, 3))
        rays = centres - self.calib.pos
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        steps = np.arange(1, int(self.unseen_depth / self.voxel) + 1) * self.voxel
        behind = (centres[:, None, :] + rays[:, None, :] * steps[None, :, None]).reshape(-1, 3)
        probe = np.unique(pack(np.floor(behind / self.voxel).astype(np.int64)))
        seen = np.sort(pack(np.floor(centres / self.voxel).astype(np.int64)))
        where = np.searchsorted(seen, probe)
        same = (where < seen.size)
        same[same] = seen[where[same]] == probe[same]
        keys = unpack(probe[~same])
        if not keys.size:
            return np.zeros((0, 3))
        out = (keys + 0.5) * self.voxel
        if self.surface is not None:
            # A hand can hide behind the person, but not inside the bench or behind the
            # box: keep only what lies in front of the nearest known surface on its ray.
            rel = (out - self.calib.pos) @ self.calib.R
            axis_depth = -rel[:, 2]
            ahead = axis_depth > 1e-3
            f = self.calib.focal()
            u = np.full(len(out), -1, dtype=np.int64)
            v = np.full(len(out), -1, dtype=np.int64)
            u[ahead] = np.rint(self.calib.width / 2 - 0.5 + f * rel[ahead, 0] / axis_depth[ahead]).astype(np.int64)
            v[ahead] = np.rint(self.calib.height / 2 - 0.5 - f * rel[ahead, 1] / axis_depth[ahead]).astype(np.int64)
            inside = ahead & (u >= 0) & (u < self.calib.width) & (v >= 0) & (v < self.calib.height)
            free = np.zeros(len(out), dtype=bool)
            pix = v[inside] * self.calib.width + u[inside]
            free[inside] = axis_depth[inside] < self.surface[pix] - 0.5 * self.voxel
            out = out[free]
        return out



def _min_filter(a: np.ndarray, r: int) -> np.ndarray:
    """Square minimum filter of half width r, as two passes of shifted minima."""
    out = a.copy()
    for axis in (0, 1):
        src = out.copy()
        for k in range(1, r + 1):
            if axis == 0:
                np.minimum(out[k:, :], src[:-k, :], out=out[k:, :])
                np.minimum(out[:-k, :], src[k:, :], out=out[:-k, :])
            else:
                np.minimum(out[:, k:], src[:, :-k], out=out[:, k:])
                np.minimum(out[:, :-k], src[:, k:], out=out[:, :-k])
    return out


def grow_near(depth: np.ndarray, px: int) -> np.ndarray:
    """Each pixel takes the nearest depth within px pixels: near objects grow sideways.

    The depth camera's lateral noise moves samples up to a few pixels, so a pixel just
    off the arm's outline can read the arm's depth. Growing the arm by the same amount
    explains it. The price is a rim this wide around the arm where a person touching
    it at the same depth is explained away too: about 1.5 cm at 3 m, against a 10 cm
    safety distance.
    """
    if px <= 0:
        return depth
    out = depth.copy()
    h, w = depth.shape
    for dv in range(-px, px + 1):
        for du in range(-px, px + 1):
            if dv == 0 and du == 0:
                continue
            shifted = np.full_like(depth, np.inf)
            ys = slice(max(dv, 0), h + min(dv, 0))
            yd = slice(max(-dv, 0), h + min(-dv, 0))
            xs = slice(max(du, 0), w + min(du, 0))
            xd = slice(max(-du, 0), w + min(-du, 0))
            shifted[yd, xd] = depth[ys, xs]
            np.minimum(out, shifted, out=out)
    return out


# ---------------------------------------------------------------- voxel keys

SHIFT = 1 << 20     # voxel indices run from -2^20 to 2^20, which is +-20 km at 2 cm
MASK = (1 << 21) - 1


def pack(grid: np.ndarray) -> np.ndarray:
    """Three voxel indices into one int64.

    numpy's unique on rows sorts the array as records and costs 12 ms on 13 000
    points. On packed integers the same answer takes 0.5 ms, and every lookup
    becomes a searchsorted instead of a set of tuples.
    """
    g = np.asarray(grid, np.int64) + SHIFT
    return (g[:, 0] << 42) | (g[:, 1] << 21) | g[:, 2]


def unpack(keys: np.ndarray) -> np.ndarray:
    keys = np.asarray(keys, np.int64)
    return np.stack([(keys >> 42) - SHIFT, ((keys >> 21) & MASK) - SHIFT, (keys & MASK) - SHIFT], axis=-1)


def blocks_line(voxels: np.ndarray, start: np.ndarray, end: np.ndarray, radius: float) -> bool:
    """Does anything the planner thinks is in the way sit on this line of sight?"""
    if voxels.shape[0] == 0:
        return False
    seg = np.asarray(end, float) - np.asarray(start, float)
    length = float(np.linalg.norm(seg))
    if length < 1e-9:
        return False
    direction = seg / length
    rel = voxels - np.asarray(start, float)
    along = np.clip(rel @ direction, 0.0, length)
    off = np.linalg.norm(rel - along[:, None] * direction, axis=1)
    return bool((off < radius).any())


def find_hole(image: np.ndarray, predicted_uv: tuple, expect_px: float,
              window: int = 40, near_px: float = 18.0) -> tuple | None:
    """The dark blob that is the hole, in image coordinates, or None.

    Thresholding a window and taking the centroid of everything dark lands 7 to 16 mm
    off, because the shadow under the tool and the neighbouring hole are dark too.
    This takes the connected blob nearest the prediction and checks its size against
    the hole's own diameter at the distance the tool is at.
    """
    h, w = image.shape[:2]
    u, v = predicted_uv
    u0, u1 = int(max(u - window, 0)), int(min(u + window, w))
    v0, v1 = int(max(v - window, 0)), int(min(v + window, h))
    if u1 - u0 < 12 or v1 - v0 < 12:
        return None
    win = image[v0:v1, u0:u1].astype(np.float32).mean(axis=2)
    dark = win < 0.55 * float(np.median(win))
    if not dark.any():
        return None
    labels = _label(dark)
    want_area = np.pi * (expect_px / 2) ** 2
    best = None
    for tag in range(1, labels.max() + 1):
        ys, xs = np.nonzero(labels == tag)
        area = len(xs)
        if area < max(6, 0.25 * want_area) or area > 4.0 * want_area:
            continue
        cu, cv = u0 + xs.mean(), v0 + ys.mean()
        off = float(np.hypot(cu - u, cv - v))
        if off > near_px:
            continue
        # round enough to be a hole seen from above
        spread = np.hypot(xs.std(), ys.std()) + 1e-6
        roundness = min(xs.std(), ys.std()) / max(xs.std(), ys.std(), 1e-6)
        if roundness < 0.45:
            continue
        score = off + abs(area - want_area) / max(want_area, 1.0)
        if best is None or score < best[0]:
            best = (score, cu, cv, area)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _label(mask: np.ndarray) -> np.ndarray:
    """Connected components of a small boolean image, four way. No scipy in this project."""
    labels = np.zeros(mask.shape, dtype=np.int32)
    tag = 0
    stack = []
    for sv in range(mask.shape[0]):
        for su in range(mask.shape[1]):
            if not mask[sv, su] or labels[sv, su]:
                continue
            tag += 1
            stack.append((sv, su))
            labels[sv, su] = tag
            while stack:
                cv_, cu_ = stack.pop()
                for dv, du in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nv, nu = cv_ + dv, cu_ + du
                    if 0 <= nv < mask.shape[0] and 0 <= nu < mask.shape[1] \
                            and mask[nv, nu] and not labels[nv, nu]:
                        labels[nv, nu] = tag
                        stack.append((nv, nu))
    return labels
