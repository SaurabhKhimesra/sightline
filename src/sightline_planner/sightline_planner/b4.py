"""B4: the rules, and a robot that looks for a way to keep working.

B2 and B3 keep the rules and then do the screws in the order they were written down.
In gate 5 that order sent the arm to the hole 3.2 cm from where the worker's hand held
the cover, and it waited there until the cycle ended while three holes it could have
driven sat 12 to 19 cm from his hand.

B4 plans against the camera's grid (viewgrid.py), the user's design: every direction
in which the camera sees him is off limits from the camera out to just behind him,
with a margin that grows with the picture's age at his measured speed. The grid is
rebuilt with every picture, every 40 ms. Every 10 ms the guard (guard.py) checks that
the arm can keep its command one more cycle and still stop outside it, and slows,
brakes or backs the arm out if not.

On top of that, B4 does what SPEC.md section 8.4 lists, in its order. It keeps
several ways of driving every screw, one per usable turn angle around the bit, each
solved once when the cell is taught. The feeder trip is the same whichever hole comes
next, since all screws are alike and the bit is symmetric, so the choice is made when
the arm leaves the feeder, with the latest picture:

1. a hole and turn angle whose approach pose, screwing pose and the joint path there
   are all outside the grid's off-limits space;
2. of those, the shortest joint move, with a hole the wrist camera can see past the
   robot's own links preferred;
3. if none is clear, wait at the nearest clear stand-off and choose again with every
   picture.

On the way, the rest of the path is checked again with every picture, and a hole
whose path has closed is swapped for another before the arm gets there. At the hole,
if the guard has to brake or move the arm, the screw is given up for now and the arm
steps back to wait.

Nothing here reads the simulator. It uses the planner's own model, its own picture of
the person, and the calibrated position of the eyes camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .b0 import (B0, BIT_DOWN, MAX_JOINT_SPEED, STANDOFF_BACK_M, STANDOFF_UP_M, YIELD_AFTER_S,
                 rot_z)
from .guard import TrajectoryGuard
from .kin import shape_line

OPTION_STEP_DEG = 15          # turn angles tried per screw when teaching (scene choice)
OPTIONS_PER_SCREW = 24        # how many ways of driving each screw are kept (scene choice)
SMOOTH_LIMIT_RAD = 0.35       # approach to hole joint move above this bows the bit: not kept
LOOK_AHEAD_S = 0.5            # a block this close along the path is acted on now (scene choice)
# Planning asks the grid as it will be this far ahead, the guard's own stopping horizon.
# Asked about the grid as it is, a hole passed the choice and failed the guard's next
# check, the arm was held, gave the hole up and chose it again: twelve times in 12 s.
PLAN_AHEAD_S = 0.15
COOL_DOWN_S = 1.0             # a hole given up is not chosen again for this long (scene choice)
AT_THE_HOLE = ("servo", "descend", "drive", "retract")


@dataclass
class Option:
    turn_deg: float
    q_hole: np.ndarray            # the approach pose above the hole
    q_down: np.ndarray            # the screwing pose, bit in the hole
    q_standoff: np.ndarray | None
    view_clear: bool              # the wrist camera sees the hole past the robot's own links
    smooth: float                 # joint move from the approach to the hole, rad


@dataclass
class B4(B0):
    eye: np.ndarray | None = None                 # where the planner believes the eyes camera is
    options: dict = field(default_factory=dict)   # screw name -> list of Option
    choices: list = field(default_factory=list)   # (time, screw, turn, why) for every choice made
    grid: object = None                           # the camera's grid, a ViewGrid
    guard: TrajectoryGuard | None = None
    waiting_since: float | None = None
    blocked_since: float | None = None
    gave_up: list = field(default_factory=list)   # (time, screw, state) for every screw left for now
    swaps: list = field(default_factory=list)     # (time, from screw, why) for every change of hole on the way
    _planned_on: float | None = None              # the picture the last path check used
    _q: np.ndarray | None = None
    feeder_turn: float = 0.0                      # the turn angle the feeder pose is taught at
    _closed_pictures: int = 0                     # pictures in a row the way to the chosen hole looked closed
    _dropped: dict = field(default_factory=dict)  # (screw, turn) -> when it was last given up

    @classmethod
    def build(cls, kin, taught, park_tip, home_q=None, eye=None, grid=None, planes=()) -> "B4":
        base = B0.build(kin, taught, park_tip, home_q=home_q)
        out = cls(kin=base.kin, taught=base.taught, park_tip=base.park_tip, screws=base.screws)
        out.poses = base.poses
        out.eye = None if eye is None else np.asarray(eye, float)
        out.guard = TrajectoryGuard(kin, grid, planes)
        out.grid = grid
        out._teach_feeder()
        out._teach_options()
        return out

    def _teach_feeder(self) -> None:
        """The feeder pose that keeps the whole arm behind the jig's back edge.

        The screws wait at the feeder with the arm there, so it is where the arm spends
        most of the cycle. B0's taught pose reached it with the elbow hanging out over
        the right of the jig, 1.5 m up, where his hands and the parts go: he pushed the
        waiting arm off the feeder every time he worked on that side. The bit is round
        and the pick is straight down, so any turn angle will do; of every angle and
        arm configuration that reaches the feeder, this keeps the one whose arm stays
        furthest behind the jig's back edge, as a programmer placing it by eye would.
        """
        park = self.poses.get("park:park")
        if park is None:
            return
        above = np.asarray(self.taught.feeder_pick, float) + [0.0, 0.0, self.taught.approach]
        pick = np.asarray(self.taught.feeder_pick, float) + [0.0, 0.0, self.taught.screw_length]
        back_edge = float(self.taught.part_pose[1]) + self.taught.cover_size[1] / 2
        old = [v for k, v in self.poses.items() if k.startswith("to_feeder:")]
        best = None
        for psi in range(0, 360, OPTION_STEP_DEG):
            R = rot_z(np.radians(psi)) @ BIT_DOWN
            for q in self.kin.ik_all(above, R, park, extra_seeds=old[:1]):
                q_dn, err = self.kin.ik(pick, R, q, iters=200)
                if err > 1e-4 or float(np.max(np.abs(q_dn - q))) > SMOOTH_LIMIT_RAD:
                    continue
                if not (self._above_worktop(q) and self._above_worktop(q_dn)):
                    continue
                g = self.guard
                pts = g.points(q)[g.moves]
                behind = float(np.min(pts[:, 1] - g.radii[g.moves])) - back_edge
                score = (round(behind, 2), -float(np.max(np.abs(q - park))))
                if best is None or score > best[0]:
                    best = (score, psi, q)
        if best is None:
            return
        self.feeder_turn = float(best[1])
        for key in [k for k in self.poses if k.startswith("to_feeder:")]:
            self.poses[key] = best[2]
        self.feeder_clearance = best[0][0]

    # ------------------------------------------------------------------ teaching
    def _teach_options(self) -> None:
        """One way of driving each screw at every turn angle around the bit, 15 degrees
        apart, solved once.

        Which side of the hole the wrist ends up on depends on the turn angle alone: at
        the first rail screw, angles 0 to 75 and 270 to 330 degrees keep both wrist joints
        on the robot's side, the rest put them 5 to 19 cm beyond the hole toward the
        worker. Each angle is solved from several starting poses and the configuration
        nearest the feeder pose is kept, if it goes down into the hole smoothly and no
        part of the arm is under the worktop. The first version walked round the bit
        from B0's pose and kept 3 of the 23 angles that reach, all with the wrist toward
        him.
        """
        ref = self.poses.get("park:park")
        for screw in self.screws:
            q_seed = self.poses.get(f"to_hole:{screw.name}")
            if q_seed is None:
                continue
            start = self.poses.get(f"to_feeder:{screw.name}", q_seed)
            above = np.asarray(screw.hole, float) + [0.0, 0.0, self.taught.screw_length + self.taught.approach]
            down = np.asarray(screw.hole, float) + [0.0, 0.0, self.taught.screw_length]
            back = np.asarray(screw.hole, float) + [0.0, STANDOFF_BACK_M, STANDOFF_UP_M]
            found = []
            for psi in range(0, 360, OPTION_STEP_DEG):
                R = rot_z(np.radians(psi)) @ BIT_DOWN
                seeds = [q_seed] + ([ref] if ref is not None else [])
                for q_up in self.kin.ik_all(above, R, start, extra_seeds=seeds):
                    q_dn, err = self.kin.ik(down, R, q_up, iters=200)
                    if err > 1e-4:
                        continue
                    smooth = float(np.max(np.abs(q_dn - q_up)))
                    if smooth > SMOOTH_LIMIT_RAD or not (self._above_worktop(q_up) and self._above_worktop(q_dn)):
                        continue
                    q_wait, err = self.kin.ik(back, R, q_up, iters=300)
                    if err > 1e-4:
                        q_wait, err = self.kin.ik_multi(back, R, q_up)
                    found.append(Option(turn_deg=float(psi), q_hole=q_up, q_down=q_dn,
                                        q_standoff=q_wait if np.isfinite(err) and self._above_worktop(q_wait) else None,
                                        view_clear=self._wrist_sees(q_up, screw.hole), smooth=smooth))
                    break
            found.sort(key=lambda o: float(np.max(np.abs(o.q_hole - start))))
            self.options[screw.name] = found[:OPTIONS_PER_SCREW]

    def _above_worktop(self, q: np.ndarray) -> bool:
        """No part of the arm under the worktop, the shoulder aside."""
        g = self.guard
        if not g.planes:
            return True
        pts = g.points(q)
        for point, normal in g.planes:
            if (((pts - point) @ normal < g.radii) & g.above_worktop).any():
                return False
        return True

    def _wrist_sees(self, q: np.ndarray, hole) -> bool:
        """Is the line from the wrist camera to the hole clear of the robot's own links?

        Gate 4 found the upper arm and forearm in that line at some turn angles, never
        the tool itself, which sits beside the camera on purpose.
        """
        cam, _ = self.kin.camera_pose(q)
        m, d = self.kin.m, self.kin.d
        d.qpos[:6] = q
        mujoco.mj_kinematics(m, d)
        wrist = m.body("ur5e/wrist_3_link").id
        a, b = cam, np.asarray(hole, float)
        for g in range(m.ngeom):
            if m.geom_group[g] != 3 or m.geom_bodyid[g] == wrist:
                continue
            centre = d.geom_xpos[g]
            half, radius = shape_line(m, g)
            axis = d.geom_xmat[g].reshape(3, 3)[:, 2] * half
            for s in np.linspace(0.0, 1.0, 12):
                p = a + s * (b - a)
                c0, c1 = centre - axis, centre + axis
                ab = c1 - c0
                t = 0.0 if float(ab @ ab) < 1e-12 else float(np.clip((p - c0) @ ab / float(ab @ ab), 0.0, 1.0))
                if float(np.linalg.norm(p - (c0 + t * ab))) < radius:
                    return False
        return True

    # ------------------------------------------------------------------ what the camera says
    def see(self, person_px, person_z, taken_at: float, q_capture, blind=None) -> None:
        """A new picture: rebuild the grid. q_capture is where the arm was when it was
        taken, which is what hid part of the view."""
        self.grid.update(person_px, person_z, taken_at, self.guard.shadow(np.asarray(q_capture, float)))
        if blind is not None:
            self.guard.blind = blind

    def note_clear(self, t: float, person: np.ndarray, safe: float = 0.10) -> None:
        """B2 and B3 decide when to try again from the voxels; B4 asks the grid instead."""

    def _first_block(self, q: np.ndarray, goal: np.ndarray, now: float) -> float:
        """Travel time to the first pose on the joint line to goal that is off limits in
        the grid as it is now, or inf. Planning asks about the world as last seen; the
        guard's own check every cycle covers what he does next."""
        configs, arrive = self.guard.line(q, goal, MAX_JOINT_SPEED, now)
        for qk, tk in zip(configs, arrive):
            if not self.guard.pose_ok(qk, now + PLAN_AHEAD_S, q):
                return tk - now
        return np.inf

    # ------------------------------------------------------------------ choosing
    def choose(self, frame, why: str) -> bool:
        """Pick the hole to drive next and how. Returns False if nothing is clear."""
        q = np.asarray(frame.q, float)
        now = frame.t
        stage = self.STAGES.index(frame.jig_signal)
        candidates = []
        for k in range(self.index, len(self.screws)):
            screw = self.screws[k]
            if screw.driven or self.STAGES.index(screw.needs) != stage:
                continue
            for opt in self.options.get(screw.name, []):
                cost = float(np.max(np.abs(opt.q_hole - q))) + (0.0 if opt.view_clear else 1.0)
                candidates.append((cost, k, opt))
        candidates.sort(key=lambda c: c[0])
        for _, k, opt in candidates:
            if now - self._dropped.get((self.screws[k].name, opt.turn_deg), -np.inf) < COOL_DOWN_S:
                continue
            ahead = now + PLAN_AHEAD_S
            if not (self.guard.pose_ok(opt.q_hole, ahead, q) and self.guard.pose_ok(opt.q_down, ahead, q)):
                continue
            if np.isfinite(self._first_block(q, opt.q_hole, now)):
                continue
            # bring the chosen screw to the front of the queue and drive it this way
            self.screws[self.index], self.screws[k] = self.screws[k], self.screws[self.index]
            screw = self.screws[self.index]
            screw.turn_deg = opt.turn_deg
            self.poses[f"to_hole:{screw.name}"] = opt.q_hole
            if opt.q_standoff is not None:
                self.poses[f"standoff:{screw.name}"] = opt.q_standoff
            self.choices.append((round(now, 2), screw.name, opt.turn_deg, why))
            return True
        return False

    def _wait_pose(self, q: np.ndarray, now: float) -> str:
        """The nearest stand-off, of any screw still to do, that is clear now; else park."""
        found = []
        for screw in self.screws[self.index:]:
            if screw.driven:
                continue
            for opt in self.options.get(screw.name, []):
                if opt.q_standoff is not None:
                    found.append((float(np.max(np.abs(opt.q_standoff - q))), screw.name, opt.q_standoff))
        found.sort(key=lambda f: f[0])
        for _, name, pose in found:
            if self.guard.pose_ok(pose, now, q):
                key = f"standoff:{name}:wait"
                self.poses[key] = pose
                return key
        return "park:park"

    # ------------------------------------------------------------------ the cycle
    def _task_inner(self, frame):
        q = np.asarray(frame.q, float)
        self._q = q
        now = frame.t
        self._drop_passed(frame)
        # pushed off the feeder by the guard: go back with a joint move, not the 10 cm/s
        # final approach, which took 7 s to come back from an escape
        if self.state == "pick" and self.index < len(self.screws):
            low = np.asarray(self.taught.feeder_pick, float) + [0.0, 0.0, self.taught.screw_length]
            high = np.asarray(self.taught.feeder_pick, float) + [0.0, 0.0, self.taught.approach]
            pos, _ = self.kin.fk(q)
            # off the short vertical line the pick runs along
            along = float(np.clip((pos - low) @ (high - low) / max(float((high - low) @ (high - low)), 1e-9), 0.0, 1.0))
            if float(np.linalg.norm(pos - (low + along * (high - low)))) > 0.03:
                self._go("to_feeder", now)
        new_picture = self.grid.taken_at is not None and self.grid.taken_at != self._planned_on
        if new_picture:
            self._planned_on = self.grid.taken_at
        # at the feeder: take a screw, and wait with it until its part is in. Then choose
        # the hole, not before
        if self.state == "pick" and self.index < len(self.screws):
            target = np.asarray(self.taught.feeder_pick, float) + [0.0, 0.0, self.taught.screw_length]
            qd, err = self._move(q, target, rot_z(np.radians(self.feeder_turn)) @ BIT_DOWN, tool_speed=0.10)
            if now >= self.until and err < 0.004 and frame.jig_signal == self.screws[self.index].needs:
                if self.choose(frame, "left the feeder"):
                    self._go("to_hole", now)
                else:
                    self._wait_somewhere(frame)
                return np.zeros(6)
            return qd
        if new_picture and self.index < len(self.screws):
            # waiting: choose again with every picture, go as soon as any hole is clear
            if self.state == "yield":
                if self.choose(frame, "clear again"):
                    self.yields.append((round(self.yield_since, 2), round(now, 2), self.screws[self.index].name))
                    self.waiting_since = None
                    self._go("to_hole", now)
                    return np.zeros(6)
                if self.yield_key != "park:park" and not self.guard.pose_ok(self.poses[self.yield_key], now, q):
                    self._move_wait(q, now)
            # on the way to a hole: is the rest of the way still clear in this picture?
            elif self.state == "to_hole":
                # Only when it stays closed for two pictures: a path that grazes the
                # margin flips between clear and closed with the noise, and B4 chose
                # and dropped the same hole twelve times in 12 s. The guard's own check
                # every 10 ms still holds the arm the moment he really comes close.
                goal = self.poses.get(self._key("to_hole"))
                if goal is not None and self._first_block(q, goal, now) < LOOK_AHEAD_S:
                    self._closed_pictures += 1
                    if self._closed_pictures >= 2:
                        self._closed_pictures = 0
                        self._swap(frame, "the way closed")
                else:
                    self._closed_pictures = 0
        # waiting for the next part: fetch its screw now and wait at the feeder with it.
        # Waiting at park instead meant a 180 degree swing round the back of the base
        # once the part was in, while he was still placing it 24 cm in front of the
        # base, and the swing was stopped for most of the 14 s the rail screws allow.
        if self.state == "wait" and self.index < len(self.screws):
            self._go("to_feeder", now)
            return np.zeros(6)
        return super()._task_inner(frame)

    def _swap(self, frame, why: str) -> None:
        was = self.screws[self.index].name
        self._dropped[(was, float(self.screws[self.index].turn_deg))] = frame.t
        if self.choose(frame, why):
            self.swaps.append((round(frame.t, 2), was, why))
            self._go("to_hole", frame.t)
        else:
            self._wait_somewhere(frame)

    def _move_wait(self, q: np.ndarray, now: float) -> None:
        key = self._wait_pose(q, now)
        if key != self.yield_key:
            self.yield_key = key
            self._goals.pop(key, None)
            self._speed.pop(key, None)

    def _wait_somewhere(self, frame) -> None:
        self.resume_state = "to_hole"
        self.yield_since = frame.t
        self.waiting_since = frame.t
        self.yield_key = self._wait_pose(np.asarray(frame.q, float), frame.t)
        self._go("yield", frame.t)
        self._goals.pop(self.yield_key, None)
        self._speed.pop(self.yield_key, None)

    def step_back(self, t: float) -> None:
        """Held or moved by the guard at the hole: leave this screw for now and wait."""
        if self.index >= len(self.screws):
            return
        screw = self.screws[self.index]
        if self.state == "retract":
            # the screw is in; only the way out was cut short, and the guard is seeing to that
            self.index += 1
            self.state = "wait"
            self.log.append((round(t, 2), "done", screw.name))
            return
        if self.state not in ("to_hole",) + AT_THE_HOLE:
            return
        self.gave_up.append((round(t, 2), screw.name, self.state))
        self._dropped[(screw.name, float(screw.turn_deg))] = t
        q = self._q if self._q is not None else np.asarray(self.poses["park:park"], float)
        self.resume_state = "to_hole"
        self.yield_since = t
        self.waiting_since = t
        self.yield_key = self._wait_pose(q, t)
        self._go("yield", t)
        self._goals.pop(self.yield_key, None)
        self._speed.pop(self.yield_key, None)

    def safe_command(self, frame, qd_task: np.ndarray) -> np.ndarray:
        """The guard's say on this cycle's command, and what the task does about it."""
        # at the hole the way out is the screw's stand-off; anywhere else, back and up
        # are nearer than park, which is on the far side of the base
        targets = [("park", self.poses.get("park:park"))]
        if self.state in ("to_hole",) + AT_THE_HOLE and self.index < len(self.screws):
            targets.insert(0, ("standoff", self.poses.get(f"standoff:{self.screws[self.index].name}")))
        elif self.state == "yield":
            targets.insert(0, ("standoff", self.poses.get(self.yield_key)))
        qd, verdict = self.guard.command(frame.q, frame.qd, qd_task, frame.t, targets)
        blocked = verdict == "brake" or verdict.startswith("escape")
        if not blocked:
            self.blocked_since = None
            return qd
        if self.blocked_since is None:
            self.blocked_since = frame.t
        if self.state in AT_THE_HOLE:
            self.step_back(frame.t)
        elif self.state == "to_hole" and frame.t - self.blocked_since >= YIELD_AFTER_S:
            self._swap(frame, "held on the way")
            self.blocked_since = None
        return qd

    def summary(self) -> dict:
        out = super().summary()
        out["choices"] = [list(c) for c in self.choices]
        out["options_per_screw"] = {k: len(v) for k, v in self.options.items()}
        out["options_view_clear"] = {k: sum(o.view_clear for o in v) for k, v in self.options.items()}
        out["gave_up"] = [list(g) for g in self.gave_up]
        out["swaps"] = [list(s) for s in self.swaps]
        if self.guard is not None:
            ms = np.array(self.guard.ms) if self.guard.ms else np.zeros(1)
            out["guard"] = {"verdicts": dict(self.guard.verdicts),
                            "ms_median": round(float(np.median(ms)), 3),
                            "ms_p99": round(float(np.percentile(ms, 99)), 3),
                            "ms_max": round(float(ms.max()), 3),
                            "points_on_the_arm": int(len(self.guard.radii))}
        out["feeder_turn_deg"] = self.feeder_turn
        out["feeder_arm_behind_jig_m"] = getattr(self, "feeder_clearance", None)
        return out
