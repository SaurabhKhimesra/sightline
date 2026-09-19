"""Every 10 ms: would the arm's next moves put any part of it where it must not be?

A trajectory is checked by where it puts the arm over the next moments: each joint,
points along every link, the tool tip and the wrist camera, each projected into the
camera's grid at the time the arm would be there. The grid grows with time by how
far he could have moved, so a move that is clear now but would meet him before the
arm could stop is caught now.

Two questions are asked:
- can the arm keep this command for one more cycle and still stop, with no point of
  it deeper inside the off-limits space than holding still would leave it? That is
  the hard one: a clear point must stay clear, and a point he has come too close to
  must not get closer. If not, a slower command, then braking, then a way out is
  sent instead. When a way out clearly helps, it goes first.
- does the path ahead stay clear? That one is for planning: the planner asks it
  before it commits to a move, and again with every new picture. A point that hardly
  moves on the way may stay where it is.

A first version allowed nothing but a way out while any point was inside the margin.
He works at the jig 20 cm from the root of the upper arm, which no way out moves
far, so the arm froze at park for the last 28 s of the cycle.

Every point carries its own margin: how far the robot's surface reaches beyond it.
The bit is 8.5 mm round, the upper arm 50 mm, and a point between two samples on a
link is up to half the spacing away from one. Checking every point with the
thickest link's radius kept the tool 6 cm further from the worker than it needs to
be.
"""

from __future__ import annotations

import time
from collections import Counter, deque

import mujoco
import numpy as np

from .kin import shape_line

SAMPLE_M = 0.05              # a point at least every 5 cm along each link (scene choice)
CYCLE_S = 0.01               # the motion cycle
STOP_DECEL = np.deg2rad(1200.0)   # the protective stop's deceleration per joint, 3 x the working ramp (scene choice)
ESCAPE_SPEED = np.deg2rad(90.0)   # the joint speed limit the motion layer keeps
STEP_S = 0.04                # one check point per camera frame along a trajectory
ESCAPE_SAMPLES = (0.10, 0.20)   # where an escape is scored, s ahead (scene choice)
# A point inside the margin may not get any deeper. Both sides of the comparison use the
# same grid at the same moment, so there is no noise to allow for; a 2 mm slack let a
# slow move creep down onto his hand 2 mm a cycle.
TOL_M = 1e-9
ESCAPE_GAIN_M = 0.01         # a way out goes first when it takes this much depth off the arm in 0.2 s (scene choice)
STAYS_M = 0.03               # in planning, a point that moves less than this may stay where it is (scene choice)
HISTORY_S = 2.0              # clear poses remembered for backing out the way the arm came
UP_SPEED = 0.25              # the tool straight up, toward the camera, m/s: the way out from anything below (scene choice)
BLIND_CELL_M = 0.05          # the blind cells' size: a hand can be anywhere in one


class TrajectoryGuard:
    def __init__(self, kin, grid, planes=()):
        self.kin = kin
        self.grid = grid
        self.planes = [(np.asarray(p, float), np.asarray(n, float) / np.linalg.norm(n)) for p, n in planes]
        self.blind = np.zeros((0, 3))           # blind cells that are live right now
        self.blind_pad = 0.5 * np.sqrt(3.0) * BLIND_CELL_M
        m = kin.m
        self.shapes = []
        radii = []
        # each joint, by the body that turns on it, with that body's thickest shape
        self.joint_bodies = [m.body(name).id for name in (
            "ur5e/shoulder_link", "ur5e/upper_arm_link", "ur5e/forearm_link",
            "ur5e/wrist_1_link", "ur5e/wrist_2_link", "ur5e/wrist_3_link")]
        bodies = []
        for b in self.joint_bodies:
            own = [shape_line(m, g)[1] for g in range(m.ngeom) if m.geom_group[g] == 3 and m.geom_bodyid[g] == b]
            radii.append(max(own) if own else 0.06)
            bodies.append(b)
        self.tip = m.site("tool_tip").id
        wrist = m.body("ur5e/wrist_3_link").id
        radii += [0.0085, 0.032]                 # the bit, whose tip this is; the wrist camera's box corner
        bodies += [wrist, wrist]
        for g in range(m.ngeom):
            if m.geom_group[g] != 3:
                continue
            half, radius = shape_line(m, g)
            n = int(np.ceil(2 * half / SAMPLE_M)) + 1 if half > 0 else 1
            ts = np.linspace(-1.0, 1.0, n) if n > 1 else np.zeros(1)
            spacing = 2 * half / (n - 1) if n > 1 else 0.0
            self.shapes.append((g, half, ts))
            radii += [radius + 0.5 * spacing] * n
            bodies += [int(m.geom_bodyid[g])] * n
        self.radii = np.array(radii)
        # the same points as arrays, so the arm is placed with a few numpy operations
        self._gid = np.array([g for g, half, ts in self.shapes for _ in ts], dtype=np.int64)
        self._off = np.array([t * half for g, half, ts in self.shapes for t in ts])
        self._cam_body = int(m.cam_bodyid[kin.cam])
        self._cam_local = m.cam_pos[kin.cam].copy()
        # the shoulder is bolted on at worktop height and only turns about the vertical,
        # so the worktop is not a limit for it
        self.above_worktop = np.array([b != self.joint_bodies[0] for b in bodies])
        # Points no joint can move, the shoulder on its own axis, are left out of every
        # decision: nothing the arm does changes them. He can still walk up to the base,
        # and the judge will say who moved in. Left in, they froze the arm whenever he
        # put a part into the jig, which sits 18 cm from the shoulder.
        rng = np.random.default_rng(0)
        spread = np.zeros(len(self.radii))
        ref = self.points(np.zeros(6))
        axis = ref[0]                           # the shoulder joint sits on the base's own axis
        off_axis = np.zeros(len(self.radii))
        for _ in range(8):
            pts = self.points(rng.uniform(-np.pi, np.pi, 6))
            spread = np.maximum(spread, np.linalg.norm(pts - ref, axis=1))
            off_axis = np.maximum(off_axis, np.linalg.norm((pts - axis)[:, :2], axis=1))
        self.moves = spread > 1e-3
        # Points that never leave the base's axis by more than 0.2 m, the root of the
        # upper arm, cannot get out of anyone's way. They must never move closer to him,
        # but they do not send the arm fleeing: while he put a part in the jig 20 cm from
        # the base, they kept the whole arm escaping for 1.5 s, all the way to park.
        self.rooted = (off_axis < 0.20)[self.moves]
        self.history: deque = deque()           # (time, q) of poses that were clear when the arm was there
        self.checks = 0
        self.verdicts: Counter = Counter()
        self.last = ""
        self.ms: list = []

    # ------------------------------------------------------------------ the arm as points
    def points(self, q: np.ndarray) -> np.ndarray:
        """Each joint, the tool tip, the wrist camera, and points along every link."""
        m, d = self.kin.m, self.kin.d
        d.qpos[:6] = q
        mujoco.mj_kinematics(m, d)
        cam = d.xpos[self._cam_body] + d.xmat[self._cam_body].reshape(3, 3) @ self._cam_local
        axes = d.geom_xmat[self._gid].reshape(-1, 3, 3)[:, :, 2]
        return np.vstack([d.xpos[self.joint_bodies], d.site_xpos[self.tip][None, :], cam[None, :],
                          d.geom_xpos[self._gid] + axes * self._off[:, None]])

    def shadow(self, q: np.ndarray) -> np.ndarray:
        """The grid cells the arm covers at these joints."""
        return self.grid.shadow_of(self.points(q), self.radii)

    # ------------------------------------------------------------------ questions
    def _excess(self, pts: np.ndarray, at: float) -> np.ndarray:
        """How far inside the off-limits space each point that moves is, in metres."""
        self.checks += 1
        pts = pts[self.moves]
        radii = self.radii[self.moves]
        e = self.grid.excess(pts, radii, at)
        if self.blind.shape[0]:
            reach = self.grid.safe + radii + self.blind_pad
            gap = np.full(len(pts), np.inf)
            for chunk in range(0, len(self.blind), 256):
                block = self.blind[chunk:chunk + 256]
                gap = np.minimum(gap, np.linalg.norm(pts[:, None, :] - block[None, :, :], axis=2).min(axis=1))
            e = np.maximum(e, reach - gap)
        for point, normal in self.planes:
            under = np.where(self.above_worktop[self.moves], radii - (pts - point) @ normal, -np.inf)
            e = np.maximum(e, under)
        return e

    def _bad(self, pts: np.ndarray, at: float) -> np.ndarray:
        return self._excess(pts, at) > 0

    def explain(self, q: np.ndarray, at: float) -> dict:
        """How many moving points of the arm each check flags, and the deepest, for logs."""
        pts = self.points(q)[self.moves]
        radii = self.radii[self.moves]
        e, cells = self.grid.excess(pts, radii, at, want_cells=True)
        out = {"grid": int((e > 0).sum()), "blind": 0, "worktop": 0,
               "deepest_mm": round(float(max(e.max(), 0.0)) * 1000, 1)}
        if (e > 0).any():
            i = int(np.argmax(e))
            k = int(cells[i])
            out["worst_point"] = [round(float(x), 3) for x in pts[i]]
            out["his_cell"] = [round(float(x), 3) for x in self.grid.cell_point(k)]
            out["cell_hidden"] = bool(self.grid.hidden[k])
            out["cell_speed"] = round(float(self.grid.cell_speed[k]), 2)
            out["cell_age"] = round(float(at - self.grid.seen_at[k]), 3)
        if self.blind.shape[0]:
            gap = np.linalg.norm(pts[:, None, :] - self.blind[None, :, :], axis=2).min(axis=1)
            out["blind"] = int((gap < self.grid.safe + radii + self.blind_pad).sum())
        for point, normal in self.planes:
            out["worktop"] += int((((pts - point) @ normal < radii) & self.above_worktop[self.moves]).sum())
        return out

    def count(self, q: np.ndarray, at: float) -> int:
        return int(self._bad(self.points(q), at).sum())

    def pose_clear(self, q: np.ndarray, at: float) -> bool:
        return not self._bad(self.points(q), at).any()

    def pose_ok(self, q: np.ndarray, at: float, q_from: np.ndarray) -> bool:
        """For planning: every point is clear, or hardly moves from q_from and is no
        deeper than it is there. The root of the upper arm stays near him whatever the
        arm does while he works at the jig; a plan must not wait for that."""
        pts = self.points(q)
        e = self._excess(pts, at)
        if not (e > 0).any():
            return True
        start = self.points(q_from)
        ref = self._excess(start, at)
        stays = np.linalg.norm(pts - start, axis=1)[self.moves] < STAYS_M
        return bool(((e <= 0) | (stays & (e <= np.maximum(ref, 0.0) + TOL_M))).all())

    def path_clear(self, configs, times) -> tuple[bool, float]:
        """True if every configuration is clear at its own time; else the first bad time."""
        for q, at in zip(configs, times):
            if self._bad(self.points(q), at).any():
                return False, at
        return True, np.inf

    # ------------------------------------------------------------------ trajectories to check
    @staticmethod
    def stop_after(q: np.ndarray, qd: np.ndarray, now: float, keep_s: float = CYCLE_S):
        """Keep qd for keep_s, then a synchronised protective stop. Configurations and times."""
        q = np.asarray(q, float)
        qd = np.asarray(qd, float)
        fastest = float(np.max(np.abs(qd)))
        configs, times = [q + qd * keep_s], [now + keep_s]
        if fastest > 1e-9:
            t_stop = fastest / STOP_DECEL
            steps = max(1, int(np.ceil(t_stop / STEP_S)))
            for k in range(1, steps + 1):
                s = min(k * STEP_S, t_stop)
                # distance covered while slowing linearly to zero
                frac = s - 0.5 * s * s / t_stop
                configs.append(q + qd * keep_s + qd * frac)
                times.append(now + keep_s + s)
        return configs, times

    @staticmethod
    def line(q: np.ndarray, goal: np.ndarray, speed: float, now: float, step: float = 0.06):
        """The straight joint line to goal, a sample every step rad of the fastest joint,
        with the time the arm would reach each one at speed."""
        q = np.asarray(q, float)
        goal = np.asarray(goal, float)
        span = float(np.max(np.abs(goal - q)))
        if span < 1e-9:
            return [q], [now]
        n = max(1, int(np.ceil(span / step)))
        configs = [q + (goal - q) * (k / n) for k in range(1, n + 1)]
        times = [now + span * (k / n) / speed for k in range(1, n + 1)]
        return configs, times

    # ------------------------------------------------------------------ the command
    @staticmethod
    def toward(qd_prev: np.ndarray, target: np.ndarray, rate: float = STOP_DECEL) -> np.ndarray:
        """The closest command to target the joints can reach in one cycle at this rate."""
        step = rate * CYCLE_S
        return qd_prev + np.clip(target - qd_prev, -step, step)

    def command(self, q: np.ndarray, qd_prev: np.ndarray, qd_want: np.ndarray, now: float,
                targets=()) -> tuple[np.ndarray, str]:
        """The joint velocity to send this cycle, and why.

        go: the task's own command. slow: its direction at half or a quarter of the
        speed. brake: slow everything to a stop. escape: move toward whichever of the
        way it came, the stand-off or park takes the most depth off the arm.
        """
        t0 = time.perf_counter()
        q = np.asarray(q, float)
        qd_prev = np.asarray(qd_prev, float)
        qd_want = np.asarray(qd_want, float)
        here = self.points(q)
        held: dict = {}

        def hold(at: float) -> np.ndarray:
            """How deep each point would be at that time if the arm stood where it is."""
            key = round(at, 6)
            if key not in held:
                held[key] = self._excess(here, at)
            return held[key]

        inside = hold(now + CYCLE_S)
        if not (inside > 0).any():
            self.history.append((now, q.copy()))
        while self.history and self.history[0][0] < now - HISTORY_S:
            self.history.popleft()
        verdict, qd = None, qd_want
        best = None
        if (inside[~self.rooted] > 0).any():
            best = self.escape(q, qd_prev, now, targets)
            staying = float(np.maximum(hold(now + ESCAPE_SAMPLES[-1])[~self.rooted], 0.0).sum())
            if best[0][0] < staying - ESCAPE_GAIN_M:
                verdict, qd = best[2], best[1]
        if verdict is None:
            tries = [("go", qd_want)]
            if not (inside[~self.rooted] > 0).any():   # inside the margin a slower way in is no better
                tries += [("slow", self.toward(qd_prev, 0.5 * qd_want)),
                          ("slow", self.toward(qd_prev, 0.25 * qd_want))]
            tries.append(("brake", self.toward(qd_prev, np.zeros(6))))
            for name, cand in tries:
                if self._no_worse(*self.stop_after(q, cand, now), hold):
                    verdict, qd = name, cand
                    break
        if verdict is None:
            best = best or self.escape(q, qd_prev, now, targets)
            verdict, qd = best[2], best[1]
        self.verdicts[verdict] += 1
        self.last = verdict
        self.ms.append((time.perf_counter() - t0) * 1000)
        return qd, verdict

    def _no_worse(self, configs, times, hold) -> bool:
        """No point deeper than holding still would leave it at the same moment; a clear
        point must stay clear."""
        for qk, tk in zip(configs, times):
            ref = hold(tk)
            allowed = np.where(ref > 0, ref + TOL_M, 0.0)
            if (self._excess(self.points(qk), tk) > allowed).any():
                return False
        return True

    def escape(self, q: np.ndarray, qd_prev: np.ndarray, now: float, targets=()):
        """Score each way out by how much depth is left on the arm 0.1 and 0.2 s ahead,
        and take the best: (score, first command, name). Braking in place is one way."""
        goals = []
        back = self._retrace(q, now)
        if back is not None:
            goals.append(("escape back", back))
        for name, g in targets:
            if g is None:
                continue
            g = np.asarray(g, float)
            # the same pose twice, or where the arm already is, is not another way out
            if float(np.max(np.abs(g - q))) > 0.03 and all(float(np.max(np.abs(g - other))) > 1e-6 for _, other in goals):
                goals.append((f"escape to {name}", g))
        # straight up, toward the camera: away from anything hidden under the arm, and
        # the one way out when the arm is already where the other ways lead
        goals.append(("escape up", "up"))
        best = None
        for name, goal in [("escape brake", None)] + goals:
            first, scores = self._rollout(q, qd_prev, goal, now)
            score = (scores[-1], sum(scores))
            if best is None or score < best[0]:
                best = (score, first, name)
        return best

    def _rollout(self, q, qd_prev, goal, now):
        """Head for goal (or stop, for None) at the protective rate; the total depth inside
        the off-limits space at each scoring time."""
        qk, qdk = q.copy(), qd_prev.copy()
        first = None
        scores = []
        marks = {int(round(s / CYCLE_S)) for s in ESCAPE_SAMPLES}
        lift = None
        for k in range(1, int(round(ESCAPE_SAMPLES[-1] / CYCLE_S)) + 1):
            if goal is None:
                target = np.zeros(6)
            elif isinstance(goal, str):
                if lift is None or k % 5 == 0:
                    lift = np.linalg.pinv(self.kin.jacobian(qk)) @ np.array([0.0, 0.0, UP_SPEED, 0.0, 0.0, 0.0])
                    fastest = float(np.max(np.abs(lift)))
                    if fastest > ESCAPE_SPEED:
                        lift *= ESCAPE_SPEED / fastest
                target = lift
            else:
                left = goal - qk
                dist = float(np.max(np.abs(left)))
                speed = min(ESCAPE_SPEED, np.sqrt(2.0 * STOP_DECEL * dist), dist / CYCLE_S)
                target = left / max(dist, 1e-9) * speed
            qdk = self.toward(qdk, target)
            qk = qk + qdk * CYCLE_S
            if first is None:
                first = qdk.copy()
            if k in marks:
                depth = np.maximum(self._excess(self.points(qk), now + k * CYCLE_S), 0.0)
                scores.append(float(depth[~self.rooted].sum()))
        return first, scores

    def _retrace(self, q: np.ndarray, now: float):
        """The latest pose the arm passed through that is still clear now."""
        tried = 0
        for i in range(len(self.history) - 1, -1, -5):
            _, past = self.history[i]
            if float(np.max(np.abs(past - q))) < 0.03:
                continue
            tried += 1
            if self.pose_clear(past, now):
                return past
            if tried >= 2:
                break
        return None
