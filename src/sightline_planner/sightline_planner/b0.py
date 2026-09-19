"""B0: visual servoing, no avoidance. The baseline that shows the problem.

It drives every screw as a taught cell would: go to the feeder, take a screw, go to
the hole, line up on the hole with the wrist camera, descend, drive, retract. It
never looks for the person, because nothing in a plain cell does. Gate 3 measures
what that costs.

Speeds, dwell times and the stand-off are scene choices and are stated in
MODEL_NOTES. The robot's own limits come from the UR5e datasheet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .frame import SensorFrame, Taught
from .kin import ToolKinematics, shape_line

TURN_STEP_DEG = 15
MAX_JOINT_SPEED = np.deg2rad(90.0)   # scene choice, below the datasheet's 180 deg/s
MAX_TOOL_SPEED = 0.25                # m/s, scene choice
DRIVE_S = 1.2                        # how long a screw takes to go in, scene choice
PICK_S = 0.4                         # dwell at the feeder, scene choice
REACHED_M = 0.0015                   # close enough to call a move done, scene choice
# A correction bigger than this is not believed. The wrist detector is a centroid of
# dark pixels and lands 5 to 12 mm off the hole, which is worse than the taught
# position, so in gate 3 every correction is refused. Gate 4 builds the detector
# properly and this is where its output will be trusted.
MAX_CORRECTION_M = 0.004
JOINT_ACCEL = np.deg2rad(400.0)      # the working ramp, the same number the motion layer enforces
YIELD_AFTER_S = 0.3                  # held this long by the rules near the work: step back (scene choice)
CLEAR_MARGIN_M = 0.02                # clear: the arm at the hole would stay one voxel beyond R1's distance (scene choice)
CLEAR_FOR_S = 0.5                    # and has stayed clear this long (scene choice)
STANDOFF_BACK_M = 0.25               # a screw's waiting pose: this far back toward the robot (scene choice)
STANDOFF_UP_M = 0.30                 # and this far above the hole (scene choice)


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


BIT_DOWN = np.column_stack([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


@dataclass
class Screw:
    name: str
    hole: np.ndarray        # world position of the hole mouth
    needs: str              # the jig signal that has to be showing
    turn_deg: float = 0.0
    driven: bool = False


@dataclass
class B0:
    """State machine and motion for the no-avoidance baseline."""
    kin: ToolKinematics
    taught: Taught
    park_tip: tuple
    screws: list = field(default_factory=list)
    state: str = "wait"
    index: int = 0
    until: float = 0.0
    log: list = field(default_factory=list)
    hole_seen: int = 0
    hole_missed: int = 0
    unreachable: list = field(default_factory=list)
    hole_error_mm: list = field(default_factory=list)
    hole_rejected: int = 0
    _corrected: np.ndarray | None = None
    _goals: dict = field(default_factory=dict)
    _leg: dict = field(default_factory=dict)
    motion: object = None                    # a MotionLayer, or None for the baseline
    reports: list = field(default_factory=list)
    _speed: dict = field(default_factory=dict)
    _last_t: float | None = None
    _now: float = 0.0
    poses: dict = field(default_factory=dict)       # the taught joint poses, solved once
    resume_state: str | None = None
    yield_key: str = "park:park"
    yield_since: float = 0.0
    clear_since: float | None = None
    yields: list = field(default_factory=list)      # (from, to, screw) for every step back
    missed: list = field(default_factory=list)      # (time, screw, why) for every screw given up

    @classmethod
    def build(cls, kin: ToolKinematics, taught: Taught, park_tip, home_q=None) -> "B0":
        screws = [Screw(f"rail_{i + 1}", taught.world(h), "rail_ready") for i, h in enumerate(taught.rail_holes)]
        screws += [Screw(f"cover_{i + 1}", taught.world(h), "cover_ready") for i, h in enumerate(taught.cover_holes)]
        out = cls(kin=kin, taught=taught, park_tip=tuple(park_tip), screws=screws)
        for s in out.screws:
            s.turn_deg = out._pick_turn(s.hole)
        out._teach(np.array([0.0, -1.6, 1.6, -1.6, -1.57, 0.0]) if home_q is None else np.asarray(home_q, float))
        return out

    def _teach(self, home: np.ndarray) -> None:
        """Solve every pose the cycle visits once, as one chain from home, and keep them.

        A programmer teaches a cell this way: each pose a short joint move from the
        last, all in one arm configuration. Solving each target afresh, nearest to
        wherever the arm happens to be, let a 15 mm drift while backing away from the
        worker flip the solver onto another branch, and the next move dived 40 cm under
        its own path to get there.
        """
        self.poses = {}
        q = home.copy()
        park, err = self.kin.ik_multi(np.asarray(self.park_tip, float), BIT_DOWN, q)
        if np.isfinite(err):
            self.poses["park:park"] = park
            q = park
        for screw in self.screws:
            R = rot_z(np.radians(screw.turn_deg)) @ BIT_DOWN
            feeder = np.asarray(self.taught.feeder_pick, float) + [0.0, 0.0, self.taught.approach]
            q_feed, err = self.kin.ik_multi(feeder, R, q)
            if np.isfinite(err):
                self.poses[f"to_feeder:{screw.name}"] = q_feed
                q = q_feed
            hole, _ = self._pose_for(screw, self.taught.screw_length + self.taught.approach)
            q_hole, err = self.kin.ik_multi(hole, R, q)
            if np.isfinite(err):
                self.poses[f"to_hole:{screw.name}"] = q_hole
                # where to wait for this screw if he is in the way: back toward the robot
                # and up, a short move from the hole, out of his space and behind the jig
                # as the camera sees it
                back = np.asarray(screw.hole, float) + [0.0, STANDOFF_BACK_M, STANDOFF_UP_M]
                q_wait, err = self.kin.ik_multi(back, R, q_hole)
                if np.isfinite(err):
                    self.poses[f"standoff:{screw.name}"] = q_wait
                q = q_hole

    # ------------------------------------------------------------------ geometry
    def _pick_turn(self, hole) -> float:
        """The turn angle around the bit that gives the smoothest insertion.

        The screwdriver mounts from the side, so the turn angle moves the whole arm
        and some angles have no solution at all (gate 1). Taking the first angle that
        solves is not enough: at one of them the arm sits next to a configuration
        change and swings 37 mm sideways on the way into the hole. So each angle is
        scored by how far the joints have to move between the approach pose and the
        hole, and the quietest one wins. A programmer teaching this cell would do the
        same by eye.
        """
        hole = np.asarray(hole, float)
        above = hole + [0.0, 0.0, self.taught.screw_length + self.taught.approach]
        down = hole + [0.0, 0.0, self.taught.screw_length]
        home = np.array([0.0, -1.6, 1.6, -1.6, -1.57, 0.0])
        best, best_score = 0.0, np.inf
        for psi in range(0, 360, TURN_STEP_DEG):
            R = rot_z(np.radians(psi)) @ BIT_DOWN
            q_up, err = self.kin.ik_multi(above, R, home)
            if not np.isfinite(err):
                continue
            q_dn, err = self.kin.ik(down, R, q_up, iters=200)
            if err > 1e-4:
                continue
            score = float(np.max(np.abs(q_dn - q_up)))
            if score < best_score:
                best, best_score = float(psi), score
        return best

    def _pose_for(self, screw: Screw, height: float):
        return (np.asarray(screw.hole, float) + [0.0, 0.0, height],
                rot_z(np.radians(screw.turn_deg)) @ BIT_DOWN)

    # ------------------------------------------------------------------ the wrist camera
    def find_hole(self, frame: SensorFrame, screw: Screw) -> np.ndarray | None:
        """Where the wrist camera says the hole is, in the cell frame.

        Predict the hole in the image from the taught pose, look for the dark blob
        around that prediction, and turn its centroid back into a point on the part's
        top face. Nothing here needs the true pose of anything.
        """
        img = frame.wrist_rgb
        if img is None:
            return None
        cam_pos, cam_R = self.kin.camera_pose(frame.q)
        rel = cam_R.T @ (np.asarray(screw.hole, float) - cam_pos)
        if rel[2] >= -1e-6:                       # the camera looks along its own -z
            return None
        h, w = img.shape[:2]
        f = (h / 2) / np.tan(np.radians(self.kin.camera_fovy()) / 2)
        u = w / 2 + f * (rel[0] / -rel[2])
        v = h / 2 - f * (rel[1] / -rel[2])
        r = 45
        u0, u1 = int(max(u - r, 0)), int(min(u + r, w))
        v0, v1 = int(max(v - r, 0)), int(min(v + r, h))
        if u1 - u0 < 10 or v1 - v0 < 10:
            self.hole_missed += 1
            return None
        win = img[v0:v1, u0:u1].astype(np.float32).mean(axis=2)
        dark = win < 0.55 * float(np.median(win))
        n = int(dark.sum())
        if n < 12 or n > 0.5 * dark.size:         # too small to be a hole, or the view is dark
            self.hole_missed += 1
            return None
        ys, xs = np.nonzero(dark)
        cu, cv = u0 + xs.mean(), v0 + ys.mean()
        # back along the ray to the plane the hole sits in
        ray = cam_R @ np.array([(cu - w / 2) / f, -(cv - h / 2) / f, -1.0])
        if abs(ray[2]) < 1e-6:
            self.hole_missed += 1
            return None
        s = (screw.hole[2] - cam_pos[2]) / ray[2]
        self.hole_seen += 1
        return cam_pos + s * ray

    # ------------------------------------------------------------------ the cycle
    def update(self, frame: SensorFrame, obstacles=None) -> np.ndarray:
        """One motion cycle. Returns the joint velocity command.

        With a motion layer attached, the command it wants goes through the layer,
        which keeps the rules. Without one this is the baseline, B0.
        """
        qd = self._task(frame)
        if self.motion is None or obstacles is None:
            return qd
        report = self.motion.solve_joint(np.asarray(frame.q, float), qd, obstacles)
        self.reports.append(report)
        return report.qd

    def _task(self, frame: SensorFrame) -> np.ndarray:
        self._now = frame.t
        qd = self._task_inner(frame)
        self._last_t = frame.t
        return qd

    # what the jig's switches can say, in the order they happen: rail pressed in, cover
    # placed (the rail screws are under it now), clamps closed, clamps opened again
    STAGES = ("empty", "rail_ready", "cover_on", "cover_ready", "done")

    def _drop_passed(self, frame: SensorFrame) -> None:
        """A screw whose stage has passed cannot be driven any more: once the cover is on,
        a rail screw under it is out of reach. Record it as missed, with the reason, and
        get on with the next one. Without this the arm kept trying the first rail screw
        for the rest of the cycle and never started the four cover screws."""
        while self.index < len(self.screws):
            screw = self.screws[self.index]
            if self.STAGES.index(frame.jig_signal) > self.STAGES.index(screw.needs) and not screw.driven \
                    and self.state not in ("descend", "drive"):
                why = "the box was taken before it was driven" if frame.jig_signal == "done" \
                    else "the next part went on before it was driven"
                self.missed.append((round(frame.t, 2), screw.name, why))
                self.index += 1
                self.state = "wait"
                self.resume_state = None
                continue
            break

    def _task_inner(self, frame: SensorFrame) -> np.ndarray:
        q = np.asarray(frame.q, float)
        self._drop_passed(frame)
        if self.index >= len(self.screws):
            self.state = "park"
        screw = self.screws[self.index] if self.index < len(self.screws) else None

        if self.state == "wait":
            if screw is not None and frame.jig_signal == screw.needs:
                self._go("to_feeder", frame.t)
            return np.zeros(6)

        if self.state == "to_feeder":
            target = np.asarray(self.taught.feeder_pick, float) + [0.0, 0.0, self.taught.approach]
            qd, left = self._goto(q, target, rot_z(np.radians(screw.turn_deg)) @ BIT_DOWN, self._key("to_feeder"))
            if left < 0.004:
                self._go("pick", frame.t, PICK_S)
            return qd

        if self.state == "pick":
            target = np.asarray(self.taught.feeder_pick, float) + [0.0, 0.0, self.taught.screw_length]
            qd, err = self._move(q, target, rot_z(np.radians(screw.turn_deg)) @ BIT_DOWN, tool_speed=0.10)
            if frame.t >= self.until and err < 0.004:
                self._go("to_hole", frame.t)
            return qd

        if self.state == "to_hole":
            pos, R = self._pose_for(screw, self.taught.screw_length + self.taught.approach)
            qd, left = self._goto(q, pos, R, self._key("to_hole"))
            if left < 0.004:
                self._corrected = None
                self._go("servo", frame.t, 0.4)
            return qd

        if self.state == "servo":
            seen = self.find_hole(frame, screw)
            if seen is not None:
                off = float(np.linalg.norm(seen[:2] - np.asarray(screw.hole, float)[:2]))
                self.hole_error_mm.append(round(off * 1000, 2))
                if off <= MAX_CORRECTION_M:
                    self._corrected = seen
                else:
                    self.hole_rejected += 1
            pos, R = self._pose_for(screw, self.taught.screw_length + self.taught.approach)
            if self._corrected is not None:
                pos = np.array([self._corrected[0], self._corrected[1], pos[2]])
            qd, err = self._move(q, pos, R)
            if frame.t >= self.until and err < 0.002:
                self._go("descend", frame.t)
            return qd

        if self.state == "descend":
            pos, R = self._pose_for(screw, self.taught.screw_length)
            if self._corrected is not None:
                pos = np.array([self._corrected[0], self._corrected[1], pos[2]])
            above = np.array([pos[0], pos[1], pos[2] + self.taught.approach])
            qd, left = self._straight(q, above, pos, R, self._key("descend"), speed=np.deg2rad(25.0))
            if left < 0.003:
                self._go("drive", frame.t, DRIVE_S)
            return qd

        if self.state == "drive":
            if frame.t >= self.until:
                screw.driven = True
                self._go("retract", frame.t)
            return np.zeros(6)

        if self.state == "retract":
            pos, R = self._pose_for(screw, self.taught.screw_length + self.taught.approach)
            below = np.array([pos[0], pos[1], pos[2] - self.taught.approach])
            qd, left = self._straight(q, below, pos, R, self._key("retract"), speed=np.deg2rad(45.0))
            if left < 0.004:
                self.index += 1
                self.state = "wait"
                self.log.append((round(frame.t, 2), "done", screw.name))
            return qd

        if self.state == "yield":
            key = self.yield_key if self.yield_key in self.poses else "park:park"
            qd, left = self._goto(q, np.asarray(self.park_tip, float), BIT_DOWN, key)
            if left < 0.004 and self.clear_since is not None and frame.t - self.clear_since >= CLEAR_FOR_S:
                resume = self.resume_state or "to_hole"
                self.yields.append((round(self.yield_since, 2), round(frame.t, 2), self.screws[self.index].name))
                # back into the cycle from the approach: whatever was half done is redone
                self._go("to_hole" if resume in ("servo", "descend", "to_hole") else resume, frame.t)
                self.resume_state = None
            return qd

        qd, _ = self._goto(q, np.asarray(self.park_tip, float), BIT_DOWN, "park:park")
        return qd

    def step_back(self, t: float) -> None:
        """The rules have held the arm near the work: retract to park and wait there.

        Waiting in place, over the box, is the worst place to wait: in gate 5 the worker
        walked into the held arm 54 times and it blocked the camera 44 % of the time.
        The park pose hides none of him from the camera (gate 1).
        """
        if self.state in ("wait", "park", "drive") or self.index >= len(self.screws):
            return
        if self.state == "yield":
            # held on the way to the stand-off as well: that is in his way too, go to park
            if self.yield_key != "park:park":
                self._goals.pop(self.yield_key, None)
                self._speed.pop(self.yield_key, None)
                self.yield_key = "park:park"
            return
        self.resume_state = self.state
        self.yield_since = t
        self.clear_since = None
        self.yield_key = f"standoff:{self.screws[self.index].name}"
        self._go("yield", t)
        self._goals.pop(self.yield_key, None)
        self._speed.pop(self.yield_key, None)

    def note_clear(self, t: float, person: np.ndarray, safe: float = 0.10) -> None:
        """Called every frame with what the camera sees of him: could the arm be at the
        current hole right now and keep R1's distance with a margin?

        Only the voxels actually seen go into this test. It decides when to try again,
        not whether the move is safe: the motion layer still holds R1 against everything,
        unseen space included, and if that holds the arm back it steps back again. A
        30 cm keep-out round the hole, unseen space included, was so strict that the arm
        waited at park until the cover went on and every rail screw was lost.
        """
        if self.index >= len(self.screws):
            return
        key = f"to_hole:{self.screws[self.index].name}"
        pose = self.poses.get(key)
        if pose is None or person.shape[0] == 0:
            gap = np.inf
        else:
            gap = self._arm_gap(pose, person)
        if gap < safe + CLEAR_MARGIN_M:
            self.clear_since = None
        elif self.clear_since is None:
            self.clear_since = t

    def _arm_gap(self, q: np.ndarray, points: np.ndarray) -> float:
        """Closest distance from the arm at these joints to any of the points."""
        m, d = self.kin.m, self.kin.d
        d.qpos[:6] = q
        mujoco.mj_kinematics(m, d)
        best = np.inf
        for g in range(m.ngeom):
            if m.geom_group[g] != 3:
                continue
            centre = d.geom_xpos[g]
            half, radius = shape_line(m, g)
            axis = d.geom_xmat[g].reshape(3, 3)[:, 2] * half
            a, b = centre - axis, centre + axis
            ab = b - a
            denom = max(float(ab @ ab), 1e-12)
            s = np.clip((points - a) @ ab / denom, 0.0, 1.0)
            gap = np.linalg.norm(points - (a + s[:, None] * ab), axis=1) - radius
            best = min(best, float(gap.min()))
        return best

    def _move(self, q, pos, R, tool_speed: float = MAX_TOOL_SPEED):
        """Fine Cartesian motion, for lining up and going in."""
        return self.kin.velocity_to(q, pos, R, max_tool_speed=tool_speed, max_joint_speed=MAX_JOINT_SPEED)

    def _straight(self, q, pos_from, pos_to, R, key: str, speed: float, steps: int = 5):
        """A straight line for the tool tip: solve waypoints along it and step through them.

        Interpolating the joints between the two ends instead bows the bit 1.8 mm
        sideways over a 60 mm descent, which would catch the screw on the hole edge.
        """
        plan = self._goals.get(key)
        if plan is None:
            seed = np.asarray(q, float)
            plan = []
            for i in range(1, steps + 1):
                point = np.asarray(pos_from, float) + (i / steps) * (np.asarray(pos_to, float) - np.asarray(pos_from, float))
                # stay on the arm configuration we are already in: searching seeds again
                # can hop to another branch, and interpolating across a hop throws the
                # bit 37 mm sideways.
                nxt, err = self.kin.ik(point, R, seed, iters=200)
                if err > 1e-4:
                    nxt, err = self.kin.ik_multi(point, R, seed)
                if not np.isfinite(err) or err > 1e-4:
                    self.unreachable.append(key)
                    break
                seed = nxt
                plan.append(seed.copy())
            self._goals[key] = plan
            self._leg[key] = 0
        if not plan:
            return np.zeros(6), 0.0
        leg = self._leg.get(key, 0)
        left = plan[leg] - np.asarray(q, float)
        # only the last point has to be hit exactly, the ones on the way just shape the path
        if float(np.max(np.abs(left))) < 0.03 and leg < len(plan) - 1:
            self._leg[key] = leg + 1
            left = plan[leg + 1] - np.asarray(q, float)
        done = float(np.max(np.abs(plan[-1] - np.asarray(q, float))))
        # the speed is profiled on the distance to the last point, the direction on this leg
        direction = left / max(float(np.max(np.abs(left))), 1e-9)
        whole = self._profiled(direction * done, key, speed, self._now)
        return whole, done

    def _profiled(self, left: np.ndarray, key: str, speed: float, t: float) -> np.ndarray:
        """One speed for all joints, along the straight joint line to the goal.

        The fastest joint ramps at the acceleration limit, cruises, and slows so it can
        stop exactly at the goal; every other joint keeps its share of that speed.
        Letting each joint ramp on its own instead made the small wrist joints finish
        while the shoulder was still starting its swing, which reached the tool out and
        down 40 cm below the path before it turned.
        """
        dt = 0.01 if self._last_t is None else float(np.clip(t - self._last_t, 1e-3, 0.05))
        dist = float(np.max(np.abs(left)))
        if dist < 1e-9:
            self._speed[key] = 0.0
            return np.zeros(6)
        stop = np.sqrt(2.0 * JOINT_ACCEL * dist)
        target = min(speed, stop, dist / dt)
        now = self._speed.get(key, 0.0)
        now = min(target, now + JOINT_ACCEL * dt) if target >= now else max(target, now - 3 * JOINT_ACCEL * dt)
        self._speed[key] = now
        return (left / dist) * now

    def _goto(self, q, pos, R, key: str, speed: float = MAX_JOINT_SPEED, t: float | None = None):
        """A joint move to a solved pose, the way a taught cell gets from A to B.

        Driving the tool along a straight line instead walks joints into their limits
        and stops 17 mm short, which is where the first run of this baseline stuck.
        """
        goal = self._goals.get(key)
        if goal is None:
            taught = self.poses.get(key)
            if taught is not None:
                # exactly as taught. The teaching chain already decides which way each
                # joint turns; "fewest turns" would swing the shoulder through the front,
                # over the worker, where the taught route goes round the back.
                goal = np.asarray(taught, float).copy()
            else:
                goal, err = self.kin.ik_multi(pos, R, np.asarray(q, float))
                if not np.isfinite(err):
                    self.unreachable.append(key)
                    self._goals[key] = np.asarray(q, float)
                    return np.zeros(6), 0.0
            self._goals[key] = goal
        left = goal - np.asarray(q, float)
        qd = self._profiled(left, key, speed, self._now if t is None else t)
        return qd, float(np.max(np.abs(left)))

    def _go(self, state: str, t: float, dwell: float = 0.0) -> None:
        self._goals.pop(self._key(state), None)   # solve the next move fresh
        self._leg.pop(self._key(state), None)
        self._speed.pop(self._key(state), None)
        self.state = state
        self.until = t + dwell
        self.log.append((round(t, 2), state, self.screws[self.index].name if self.index < len(self.screws) else "-"))

    def _key(self, state: str) -> str:
        if state in ("park", "yield"):
            return "park:park"
        name = self.screws[self.index].name if self.index < len(self.screws) else "park"
        return f"{state}:{name}"

    def parts_in_jig(self, signal: str, box_seen: bool) -> tuple:
        """Which parts the planner's own task state says are in the jig. It gets no
        signal for the base alone, so the base counts once the box fit has seen it."""
        if self.index >= len(self.screws) or signal == "done":
            return ()
        if signal in ("cover_on", "cover_ready"):
            return ("wp_base", "wp_rail", "wp_cover")
        if signal == "rail_ready":
            return ("wp_base", "wp_rail")
        return ("wp_base",) if box_seen else ()

    def summary(self) -> dict:
        return {
            "screws_driven": sum(1 for s in self.screws if s.driven),
            "screws_total": len(self.screws),
            "turn_angles_deg": {s.name: s.turn_deg for s in self.screws},
            "hole_seen_frames": self.hole_seen,
            "hole_missed_frames": self.hole_missed,
            "unreachable": self.unreachable,
            "hole_rejected_corrections": self.hole_rejected,
            "yields": [list(y) for y in self.yields],
            "missed": [list(x) for x in self.missed],
            "hole_error_mm": self.hole_error_mm,
            "state_log": [list(x) for x in self.log],
        }
