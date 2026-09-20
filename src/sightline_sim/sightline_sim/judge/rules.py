"""R1 and R2, measured the way docs/design.md section 4 defines them.

R1: at every step, the distance between every robot geom and every human geom is
above zero. Every contact is labelled with who moved in.

R2: for every eyes camera frame, one segmentation image with the robot and one
without. A pixel that shows the person without the robot and the robot with it is
blocked.

Nothing here may be imported by the planner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .. import station

# who moved in, per docs/design.md section 4: put the robot back one motion cycle and keep
# the person where they are. The cycle is 10 ms.
MOTION_CYCLE_S = 0.010
GEOM_TYPE = int(mujoco.mjtObj.mjOBJ_GEOM)


@dataclass
class Contact:
    t: float
    human_body: str
    depth_mm: float
    who_moved_in: str
    person_speed_m_s: float


@dataclass
class BlockedFrame:
    t: float
    pixels: int
    person_pixels: int
    who_moved_in: str


@dataclass
class BlockEvent:
    """A run of blocked frames. Who moved in is decided once, when it starts.

    Asking the question on every frame answers something else: once the view has
    been blocked for a while, rewinding the robot 10 ms never clears it, so every
    frame reads "the person moved in" whatever actually happened.
    """
    t_start: float
    t_end: float
    frames: int
    worst_pixels: int
    who_moved_in: str          # docs/design.md section 4: the robot back one motion cycle, 10 ms
    who_moved_in_frame: str    # the robot back one camera frame, 40 ms at 25 Hz


def _geoms(m: mujoco.MjModel, prefix: str) -> np.ndarray:
    return np.array([g for g in range(m.ngeom)
                     if m.body(m.geom_bodyid[g]).name.startswith(prefix)
                     and m.geom_group[g] == station.GROUP_COLLISION], dtype=int)


class R1Judge:
    """Contacts and the closest approach between the robot and the person."""

    def __init__(self, m: mujoco.MjModel, robot_qadr: np.ndarray):
        self.m = m
        self.robot_qadr = np.asarray(robot_qadr, int)
        self.robot = _geoms(m, station.ROBOT_PREFIX)
        self.human = _geoms(m, station.HUMAN_PREFIX)
        if len(self.robot) == 0 or len(self.human) == 0:
            raise ValueError("no collision geoms for the robot or the person")
        self.robot_bound = m.geom_rbound[self.robot]
        self.human_bound = m.geom_rbound[self.human]
        self.robot_set = set(int(g) for g in self.robot)
        self.human_set = set(int(g) for g in self.human)
        self.min_distance = float("inf")
        self.min_distance_t = 0.0
        self.contacts: list[Contact] = []
        self._open: dict = {}

    def distance(self, d: mujoco.MjData, near: float = 0.20) -> float:
        """Smallest robot to person distance, m. Negative means overlap.

        A bounding sphere test first: the exact call is only worth making for pairs
        that could be close, and there are 600 pairs.
        """
        rp = d.geom_xpos[self.robot]
        hp = d.geom_xpos[self.human]
        gap = np.linalg.norm(rp[:, None, :] - hp[None, :, :], axis=2)
        gap -= self.robot_bound[:, None] + self.human_bound[None, :]
        best = float("inf")
        for i, j in zip(*np.where(gap < near)):
            dist = mujoco.mj_geomDistance(self.m, d, int(self.robot[i]), int(self.human[j]), near, None)
            best = min(best, dist)
        return best

    def step(self, d: mujoco.MjData, t: float, robot_q_before: np.ndarray | None,
             person_speed: float = 0.0, measure_distance: bool = False) -> None:
        if measure_distance:
            dist = self.distance(d)
            if dist < self.min_distance:
                self.min_distance, self.min_distance_t = dist, t
        touching = {}
        for i in range(d.ncon):
            g1, g2 = int(d.contact.geom[i][0]), int(d.contact.geom[i][1])
            if g1 in self.robot_set and g2 in self.human_set:
                rg, hg = g1, g2
            elif g2 in self.robot_set and g1 in self.human_set:
                rg, hg = g2, g1
            else:
                continue
            body = self.m.body(self.m.geom_bodyid[hg]).name
            depth = -float(d.contact.dist[i])
            touching[(rg, hg)] = max(touching.get((rg, hg), 0.0), depth)
        for key, depth in touching.items():
            if key in self._open:
                self._open[key].depth_mm = max(self._open[key].depth_mm, depth * 1000.0)
                continue
            c = Contact(t=round(t, 3), human_body=self.m.body(self.m.geom_bodyid[key[1]]).name,
                        depth_mm=depth * 1000.0,
                        who_moved_in=self._who(d, key, robot_q_before),
                        person_speed_m_s=round(person_speed, 3))
            self._open[key] = c
            self.contacts.append(c)
        for key in list(self._open):
            if key not in touching:
                del self._open[key]

    def _who(self, d: mujoco.MjData, pair, robot_q_before) -> str:
        """Rewind the robot one motion cycle, keep the person, and look again."""
        if robot_q_before is None:
            return "unknown"
        keep = d.qpos[self.robot_qadr].copy()
        d.qpos[self.robot_qadr] = robot_q_before
        mujoco.mj_forward(self.m, d)
        still = any((int(d.contact.geom[i][0]), int(d.contact.geom[i][1])) in (pair, pair[::-1])
                    for i in range(d.ncon))
        d.qpos[self.robot_qadr] = keep
        mujoco.mj_forward(self.m, d)
        return "person" if still else "robot"

    def results(self) -> dict:
        moved_in = {"robot": 0, "person": 0, "unknown": 0}
        for c in self.contacts:
            moved_in[c.who_moved_in] += 1
        return {
            "contacts": len(self.contacts),
            "contacts_by_who_moved_in": moved_in,
            "min_distance_mm": None if self.min_distance == float("inf") else round(self.min_distance * 1000, 1),
            "min_distance_t": round(self.min_distance_t, 2),
            "worst_contact_depth_mm": round(max((c.depth_mm for c in self.contacts), default=0.0), 1),
            "contact_list": [vars(c) | {"depth_mm": round(c.depth_mm, 1)} for c in self.contacts[:40]],
        }


class SegmentationCamera:
    """Segmentation renders that name the owner of every pixel.

    Anti-aliasing blends neighbouring segment ids into ids that belong to no geom,
    and the renderer then raises on its own lookup table, so the context is built
    with offsamples off. The colour renderers keep theirs.
    """

    def __init__(self, m: mujoco.MjModel, camera: str = "eyes_cam", width: int = 640, height: int = 480):
        self.m = m
        self.camera = camera
        samples = m.vis.quality.offsamples
        m.vis.quality.offsamples = 0
        try:
            self.renderer = mujoco.Renderer(m, height, width)
        finally:
            m.vis.quality.offsamples = samples
        self.renderer.enable_segmentation_rendering()
        self.with_robot = mujoco.MjvOption()
        self.without_robot = mujoco.MjvOption()
        for g in range(6):
            self.with_robot.geomgroup[g] = 1 if g in (station.GROUP_ENV, station.GROUP_WORKER, station.GROUP_ROBOT) else 0
            self.without_robot.geomgroup[g] = 1 if g in (station.GROUP_ENV, station.GROUP_WORKER) else 0
        self.owner = np.full(m.ngeom + 1, 0, dtype=np.uint8)  # 0 other, 1 person, 2 robot
        for g in range(m.ngeom):
            name = m.body(m.geom_bodyid[g]).name
            if name.startswith(station.HUMAN_PREFIX):
                self.owner[g] = 1
            elif name.startswith(station.ROBOT_PREFIX):
                self.owner[g] = 2

    def labels(self, d: mujoco.MjData, hide_robot: bool = False) -> np.ndarray:
        opt = self.without_robot if hide_robot else self.with_robot
        self.renderer.update_scene(d, camera=self.camera, scene_option=opt)
        seg = self.renderer.render()
        objid, objtype = seg[:, :, 0], seg[:, :, 1]
        out = np.zeros(objid.shape, dtype=np.uint8)
        # against the enum itself this comparison goes through Python for every pixel
        # and costs 160 ms. Against a plain int it is a numpy compare: under 2 ms.
        hit = objtype == GEOM_TYPE
        out[hit] = self.owner[objid[hit]]
        return out

    def close(self) -> None:
        self.renderer.close()


class R2Judge:
    """Pixels of the person that the robot takes away from the eyes camera."""

    def __init__(self, m: mujoco.MjModel, robot_qadr: np.ndarray, camera: str = "eyes_cam",
                 width: int = 640, height: int = 480):
        self.m = m
        self.robot_qadr = np.asarray(robot_qadr, int)
        self.cam = SegmentationCamera(m, camera, width, height)
        self.frames = 0
        self.blocked: list[BlockedFrame] = []
        self.events: list[BlockEvent] = []
        self.worst_pixels = 0
        self._open: BlockEvent | None = None

    def _blocked_with(self, d: mujoco.MjData, person: np.ndarray, robot_q: np.ndarray) -> bool:
        """Was the view blocked with the robot at these joints and the person as it is now?"""
        keep = d.qpos[self.robot_qadr].copy()
        d.qpos[self.robot_qadr] = robot_q
        mujoco.mj_forward(self.m, d)
        before = self.cam.labels(d, hide_robot=False)
        d.qpos[self.robot_qadr] = keep
        mujoco.mj_forward(self.m, d)
        return bool((person & (before == 2)).any())

    def frame(self, d: mujoco.MjData, t: float, robot_q_before: np.ndarray | None,
              robot_q_last_frame: np.ndarray | None = None) -> BlockedFrame | None:
        self.frames += 1
        clear = self.cam.labels(d, hide_robot=True)
        shown = self.cam.labels(d, hide_robot=False)
        person = clear == 1
        taken = person & (shown == 2)
        n = int(taken.sum())
        if n == 0:
            self._open = None
            return None
        if self._open is not None:                 # the same blockage, still there
            self._open.t_end = round(t, 3)
            self._open.frames += 1
            self._open.worst_pixels = max(self._open.worst_pixels, n)
            out = BlockedFrame(t=round(t, 3), pixels=n, person_pixels=int(person.sum()),
                               who_moved_in=self._open.who_moved_in)
            self.blocked.append(out)
            self.worst_pixels = max(self.worst_pixels, n)
            return out
        who = "unknown"
        if robot_q_before is not None:
            who = "person" if self._blocked_with(d, person, robot_q_before) else "robot"
        # One motion cycle is 2.5 mm of arm travel. By the time a 25 Hz camera catches
        # the overlap it is wider than that, so the 10 ms answer is almost always
        # "the person moved in". The same question one camera frame back is the one the
        # picture can actually answer, so both are reported.
        who_frame = "unknown"
        if robot_q_last_frame is not None:
            who_frame = "person" if self._blocked_with(d, person, robot_q_last_frame) else "robot"
        out = BlockedFrame(t=round(t, 3), pixels=n, person_pixels=int(person.sum()), who_moved_in=who)
        self.blocked.append(out)
        self.worst_pixels = max(self.worst_pixels, n)
        self._open = BlockEvent(t_start=round(t, 3), t_end=round(t, 3), frames=1, worst_pixels=n,
                                who_moved_in=who, who_moved_in_frame=who_frame)
        self.events.append(self._open)
        return out

    def results(self) -> dict:
        by_frame = {"robot": 0, "person": 0, "unknown": 0}
        by_event = {"robot": 0, "person": 0, "unknown": 0}
        by_event_frame = {"robot": 0, "person": 0, "unknown": 0}
        for f in self.blocked:
            by_frame[f.who_moved_in] += 1
        for e in self.events:
            by_event[e.who_moved_in] += 1
            by_event_frame[e.who_moved_in_frame] += 1
        longest = max((e.frames for e in self.events), default=0)
        return {
            "frames": self.frames,
            "blocked_frames": len(self.blocked),
            "blocked_frames_pct": round(100.0 * len(self.blocked) / max(self.frames, 1), 1),
            "blocked_events": len(self.events),
            "blocked_events_by_who_moved_in": by_event,
            "blocked_events_by_who_moved_in_one_frame_back": by_event_frame,
            "blocked_frames_by_who_moved_in": by_frame,
            "longest_block_s": round(longest / 30.0, 2),
            "worst_blocked_pixels": self.worst_pixels,
            "event_list": [vars(e) for e in self.events[:40]],
        }

    def close(self) -> None:
        self.cam.close()
