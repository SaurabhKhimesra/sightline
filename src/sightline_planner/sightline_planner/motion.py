"""The motion layer: every cycle, choose the joint velocity that does the job safely.

The task wants a tool velocity. The rules want distances kept. Both go into one
quadratic programme (SPEC.md section 8.5): the task sits in the objective, where it
can be given up, and the rules sit in the constraints, where they cannot.

Distances are held with the velocity damper of Faverjon and Tournassoud 1987: while
a pair is further apart than the influence distance nothing binds, and inside it the
closing speed is capped so that the pair can still be stopped before it reaches the
safety distance. The person's own velocity is in the bound, so a hand moving toward
the arm tightens it.

R2 is the same damper against a line rather than a point: the sight line from the
camera to a person voxel is a segment the arm must stay clear of, or it takes the
camera's view of that voxel away.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import qp
from .kin import shape_line

# All scene choices, stated in MODEL_NOTES.
#
# A pair may close only as fast as it could still stop before its safety distance:
# approach speed <= sqrt(2 a (d - d_s)). The influence distance is where that allows
# the arm's top speed, so a link is never inside the zone faster than it can brake.
# The first version used the linear damper with a 10 cm zone for the worktop: a link
# arriving at 0.6 m/s needed 9 m/s^2 to stop, could not, and went 105 mm under it.
A_BRAKE = 3.0            # m/s^2, what the planner counts on for braking a link (scene choice)
V_LINK_MAX = 1.5         # m/s, fastest a link point moves at the joint speed limit
BRAKE_ZONE = V_LINK_MAX ** 2 / (2 * A_BRAKE)      # 0.375 m
D_SAFE = 0.10            # never closer than this to the person, m
D_INFLUENCE = D_SAFE + BRAKE_ZONE
XI = 1.0                 # kept for reference: the linear damper this replaced
R_VIS = 0.06             # keep this far off a sight line, m
VIS_INFLUENCE = R_VIS + BRAKE_ZONE
D_SAFE_STATIC = 0.02     # the bench and the jig are not people
STATIC_INFLUENCE = D_SAFE_STATIC + BRAKE_ZONE


def approach_limit(dist: float, safe: float) -> float:
    """How fast this pair may close: the speed from which the arm still stops in time."""
    return float(np.sqrt(2.0 * A_BRAKE * max(dist - safe, 0.0)))
# A pair already inside the safety distance is not ordered to open it by the
# constraint: with several such pairs the orders conflict, every cycle is infeasible
# and the arm freezes where it is. The constraint only forbids getting closer, which
# standing still always satisfies against a still person, and backing out is asked
# for in the objective instead, with this weight per metre of shortfall.
RETREAT_WEIGHT = 4.0
PAIRS_PER_LINK = 4       # the nearest few obstacles per link is enough to shape the move
SMOOTH = 0.02            # weight on small joint speeds
CONTINUITY = 0.05        # weight on staying near the last command


@dataclass
class Obstacles:
    """What the motion layer has to keep away from, in the world frame."""
    person: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    person_velocity: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    unseen: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    sight_from: np.ndarray | None = None      # the eyes camera, for R2
    static: list = field(default_factory=list)   # (p0, p1, radius) segments of the cell
    planes: list = field(default_factory=list)   # (point, unit normal): stay on the normal's side

    def guard_points(self) -> np.ndarray:
        """Person voxels and the space behind them: both can hold a hand."""
        if self.unseen.size and self.person.size:
            return np.vstack([self.person, self.unseen])
        return self.person if self.person.size else self.unseen


@dataclass
class Report:
    qd: np.ndarray
    constraints: int
    r1_active: int
    r2_active: int
    fallback: str = ""
    min_person_distance: float = np.inf     # to a voxel the camera actually saw
    min_guard_distance: float = np.inf      # to anything guarded, unseen space included
    solve_ms: float = 0.0


def segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Closest point on segment ab to point p, and the distance."""
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom < 1e-12 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    closest = a + t * ab
    return closest, float(np.linalg.norm(p - closest))


def segment_to_segment(a0, a1, b0, b1):
    """Closest points between two segments, and the distance. Sampled, not solved.

    Eight samples along each segment is enough here: the capsules are short, the
    margins are centimetres, and an exact solve would cost more than the accuracy is
    worth at 100 Hz.
    """
    best = (None, None, np.inf)
    for t in np.linspace(0.0, 1.0, 8):
        p = a0 + t * (a1 - a0)
        q, dist = segment_distance(p, b0, b1)
        if dist < best[2]:
            best = (p, q, dist)
    return best


class MotionLayer:
    """Turns a wanted tool velocity into a joint velocity that keeps the rules."""

    def __init__(self, kin, use_r1: bool = True, use_r2: bool = False,
                 joint_speed: float = np.deg2rad(90.0), joint_accel: float = np.deg2rad(400.0),
                 dt: float = 0.01):
        self.kin = kin
        self.use_r1 = use_r1
        self.use_r2 = use_r2
        self.joint_speed = joint_speed
        self.joint_accel = joint_accel
        self.dt = dt
        self.capsules = self._capsules()
        self.last = np.zeros(6)

    def _capsules(self) -> list:
        """The robot's own collision shapes, as segments with a radius."""
        m = self.kin.m
        out = []
        for g in range(m.ngeom):
            if m.geom_group[g] != 3:
                continue
            half, radius = shape_line(m, g)
            out.append((g, int(m.geom_bodyid[g]), half, radius))
        return out

    def _world_capsules(self, q: np.ndarray) -> list:
        m, d = self.kin.m, self.kin.d
        d.qpos[:6] = q
        mujoco.mj_kinematics(m, d)
        out = []
        for g, body, half, radius in self.capsules:
            centre = d.geom_xpos[g].copy()
            axis = d.geom_xmat[g].reshape(3, 3)[:, 2] * half
            out.append((body, centre - axis, centre + axis, radius))
        return out

    def _point_jacobian(self, point: np.ndarray, body: int) -> np.ndarray:
        m, d = self.kin.m, self.kin.d
        jacp = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jacp, None, np.asarray(point, float), body)
        return jacp[:, :6]

    def solve(self, q: np.ndarray, twist: np.ndarray, obstacles: Obstacles,
              qd_prev: np.ndarray | None = None) -> Report:
        """twist is the tool velocity the task wants: three linear, three angular."""
        q = np.asarray(q, float)
        qd_prev = self.last if qd_prev is None else np.asarray(qd_prev, float)
        J = self.kin.jacobian(q)
        G = J.T @ J + (SMOOTH + CONTINUITY) * np.eye(6)
        a = -(J.T @ np.asarray(twist, float)) - CONTINUITY * qd_prev
        return self._solve(q, G, a, obstacles, qd_prev)

    def solve_joint(self, q: np.ndarray, qd_task: np.ndarray, obstacles: Obstacles,
                    qd_prev: np.ndarray | None = None) -> Report:
        """qd_task is the joint velocity the task wants. With nothing near, it is returned
        as it is, so a guarded controller and an unguarded one only differ near a person."""
        q = np.asarray(q, float)
        qd_prev = self.last if qd_prev is None else np.asarray(qd_prev, float)
        G = (1.0 + CONTINUITY) * np.eye(6)
        a = -np.asarray(qd_task, float) - CONTINUITY * qd_prev
        return self._solve(q, G, a, obstacles, qd_prev)

    def _solve(self, q, G, a, obstacles: Obstacles, qd_prev) -> Report:
        import time
        t0 = time.perf_counter()
        rows, bounds, labels = [], [], []
        # joint speed, joint acceleration and the joint limits themselves
        for j in range(6):
            e = np.zeros(6)
            e[j] = 1.0
            lo_speed = max(-self.joint_speed, qd_prev[j] - self.joint_accel * self.dt)
            hi_speed = min(self.joint_speed, qd_prev[j] + self.joint_accel * self.dt)
            if self.kin.limited[j]:
                lo_speed = max(lo_speed, (self.kin.qlim[j, 0] - q[j]) / self.dt)
                hi_speed = min(hi_speed, (self.kin.qlim[j, 1] - q[j]) / self.dt)
            if lo_speed > hi_speed:            # a limit and a ramp disagree: keep the limit
                lo_speed = hi_speed = float(np.clip(0.0, min(lo_speed, hi_speed), max(lo_speed, hi_speed)))
            rows.append(e.copy())
            bounds.append(lo_speed)
            rows.append(-e)
            bounds.append(-hi_speed)
            labels += [f"joint {j} low", f"joint {j} high"]

        caps = self._world_capsules(q)
        r1_rows = r2_rows = 0
        closest_person = closest_guard = np.inf
        n_seen = int(obstacles.person.shape[0])

        if self.use_r1:
            guard = obstacles.guard_points()
            if guard.size:
                speeds = obstacles.person_velocity
                for body, p0, p1, radius in caps:
                    pairs = self._near_points(guard, p0, p1, radius, D_INFLUENCE)
                    for index, witness, point, dist in pairs:
                        closest_guard = min(closest_guard, dist)
                        if index < n_seen:
                            closest_person = min(closest_person, dist)
                        normal = point - witness
                        norm = float(np.linalg.norm(normal))
                        if norm < 1e-9:
                            continue
                        normal /= norm
                        jac = self._point_jacobian(witness, body)
                        moving = 0.0
                        if index < len(speeds) and speeds.size:
                            moving = float(normal @ speeds[index])
                        # d_dot = n.v_person - n.v_robot must stay above the damper's floor
                        rows.append(-(normal @ jac))
                        floor = -approach_limit(dist, D_SAFE) - moving
                        bounds.append(min(floor, 0.0))
                        if dist < D_SAFE:                    # too close already: ask to back out
                            a = a + RETREAT_WEIGHT * (D_SAFE - dist) * (normal @ jac)
                        labels.append(f"R1 body {body} at {dist * 1000:.0f} mm")
                        r1_rows += 1

        if self.use_r2 and obstacles.sight_from is not None and obstacles.person.size:
            eye = np.asarray(obstacles.sight_from, float)
            for body, p0, p1, radius in caps:
                lines = self._near_lines(obstacles.person, eye, p0, p1, radius, VIS_INFLUENCE)
                for witness, on_line, dist in lines:
                    normal = on_line - witness
                    norm = float(np.linalg.norm(normal))
                    if norm < 1e-9:
                        continue
                    normal /= norm
                    jac = self._point_jacobian(witness, body)
                    rows.append(-(normal @ jac))
                    bounds.append(min(-approach_limit(dist, R_VIS), 0.0))
                    if dist < R_VIS:
                        a = a + RETREAT_WEIGHT * (R_VIS - dist) * (normal @ jac)
                    labels.append(f"R2 body {body} at {dist * 1000:.0f} mm")
                    r2_rows += 1

        for seg in obstacles.static:
            s0, s1, srad = seg
            for body, p0, p1, radius in caps:
                witness, on_static, dist = segment_to_segment(p0, p1, np.asarray(s0), np.asarray(s1))
                dist -= radius + srad
                if dist > STATIC_INFLUENCE:
                    continue
                normal = on_static - witness
                norm = float(np.linalg.norm(normal))
                if norm < 1e-9:
                    continue
                normal /= norm
                jac = self._point_jacobian(witness, body)
                rows.append(-(normal @ jac))
                bounds.append(min(-approach_limit(dist, D_SAFE_STATIC), 0.0))
                if dist < D_SAFE_STATIC:
                    a = a + RETREAT_WEIGHT * (D_SAFE_STATIC - dist) * (normal @ jac)
                labels.append(f"static body {body} at {dist * 1000:.0f} mm")

        for point, normal in obstacles.planes:
            normal = np.asarray(normal, float) / max(float(np.linalg.norm(normal)), 1e-9)
            for body, p0, p1, radius in caps:
                lowest = min(float((p0 - point) @ normal), float((p1 - point) @ normal)) - radius
                if lowest > STATIC_INFLUENCE:
                    continue
                witness = p0 if float((p0 - point) @ normal) <= float((p1 - point) @ normal) else p1
                jac = self._point_jacobian(witness, body)
                rows.append(normal @ jac)          # speed along the normal must stay above the floor
                bounds.append(min(-approach_limit(lowest, D_SAFE_STATIC), 0.0))
                if lowest < D_SAFE_STATIC:
                    a = a - RETREAT_WEIGHT * (D_SAFE_STATIC - lowest) * (normal @ jac)
                labels.append(f"plane body {body} at {lowest * 1000:.0f} mm")

        C = np.array(rows).T if rows else np.zeros((6, 0))
        b = np.array(bounds) if bounds else np.zeros(0)
        sol = qp.solve(G, a, C, b)
        self.debug = {"C": C, "b": b, "labels": labels, "sol": sol}
        fallback = ""
        if not sol.feasible:
            # the rules alone, with no task at all: move away and nothing else, and with
            # the acceleration a protective stop is allowed rather than the working ramp
            fallback = "rules only"
            ramp = 12                                  # the first 12 rows are the joint bounds
            C2, b2 = C.copy(), b.copy()
            for j in range(6):
                lo = max(-self.joint_speed, qd_prev[j] - 3 * self.joint_accel * self.dt)
                hi = min(self.joint_speed, qd_prev[j] + 3 * self.joint_accel * self.dt)
                if self.kin.limited[j]:
                    lo = max(lo, (self.kin.qlim[j, 0] - q[j]) / self.dt)
                    hi = min(hi, (self.kin.qlim[j, 1] - q[j]) / self.dt)
                b2[2 * j], b2[2 * j + 1] = lo, -hi
            sol = qp.solve(np.eye(6) * 1.0, np.zeros(6), C2, b2)
            if not sol.feasible:
                # slow every joint toward zero at three times the working deceleration.
                # The first version set the speed to minus the last one, clipped: that
                # does not brake, it throws each joint into reverse.
                fallback = "brake"
                step = 3 * self.joint_accel * self.dt
                sol = qp.Solution(x=qd_prev - np.clip(qd_prev, -step, step),
                                  active=[], multipliers=np.zeros(0), iterations=0, feasible=False)
        qd = np.clip(sol.x, -self.joint_speed, self.joint_speed)
        self.last = qd
        return Report(qd=qd, constraints=len(bounds), r1_active=r1_rows, r2_active=r2_rows,
                      fallback=fallback, min_person_distance=closest_person,
                      min_guard_distance=closest_guard, solve_ms=(time.perf_counter() - t0) * 1000)

    def _near_points(self, points: np.ndarray, p0, p1, radius: float, influence: float) -> list:
        """The nearest few obstacle points to one capsule, with their witness points."""
        ab = p1 - p0
        denom = float(ab @ ab)
        if denom < 1e-12:
            t = np.zeros(len(points))
        else:
            t = np.clip((points - p0) @ ab / denom, 0.0, 1.0)
        witness = p0 + t[:, None] * ab
        gap = np.linalg.norm(points - witness, axis=1) - radius
        near = np.nonzero(gap < influence)[0]
        if near.size == 0:
            return []
        order = near[np.argsort(gap[near])[:PAIRS_PER_LINK]]
        return [(int(i), witness[i], points[i], float(gap[i])) for i in order]

    def _near_lines(self, voxels: np.ndarray, eye: np.ndarray, p0, p1, radius: float,
                    influence: float) -> list:
        """Sight lines from the camera to person voxels that this capsule is close to.

        Done for all voxels at once. The first version looped in Python over every
        voxel for every link every cycle and turned a 3.5 minute episode into hours.
        Lines that point nowhere near the capsule, seen from the camera, are dropped
        before any distance is computed.
        """
        if voxels.shape[0] == 0:
            return []
        eye = np.asarray(eye, float)
        centre = (p0 + p1) / 2
        to_centre = centre - eye
        reach = float(np.linalg.norm(to_centre))
        half = float(np.linalg.norm(p1 - p0)) / 2 + radius + influence
        rays = voxels - eye
        lengths = np.linalg.norm(rays, axis=1)
        ok = lengths > reach - half                     # the voxel is beyond the capsule, not in front of it
        cos_limit = np.cos(np.arctan2(half, max(reach, 1e-6)))
        ok &= (rays @ to_centre) >= cos_limit * lengths * reach
        idx = np.nonzero(ok)[0]
        if idx.size == 0:
            return []
        samples = p0 + np.linspace(0.0, 1.0, 8)[:, None] * (p1 - p0)       # 8 points along the capsule
        d_line = rays[idx]                                                  # line from eye to each voxel
        denom = np.maximum(np.einsum("ij,ij->i", d_line, d_line), 1e-12)
        rel = samples[:, None, :] - eye                                     # (8, n, 3)
        tpar = np.clip(np.einsum("snk,nk->sn", rel, d_line) / denom, 0.0, 1.0)
        on_line = eye + tpar[..., None] * d_line                            # (8, n, 3)
        dist = np.linalg.norm(samples[:, None, :] - on_line, axis=2) - radius
        best_sample = np.argmin(dist, axis=0)
        cols = np.arange(idx.size)
        best = dist[best_sample, cols]
        near = np.nonzero(best < influence)[0]
        if near.size == 0:
            return []
        order = near[np.argsort(best[near])[:PAIRS_PER_LINK]]
        return [(samples[best_sample[k]], on_line[best_sample[k], k], float(best[k])) for k in order]
