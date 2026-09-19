"""Kinematics on the planner's own robot model.

The model holds the robot, its tool and its cameras and nothing else, so nothing
here can read the person. It is the same model the cell was built from, which is
what a real controller has in its URDF.
"""

from __future__ import annotations

import mujoco
import numpy as np

# Shapes with a centre line: a capsule or a cylinder has half its length in size[1].
# Compared as plain ints: "geom_type[g] in (mjGEOM_CAPSULE, ...)" is always False in
# these bindings, and it quietly turned every link of the arm into a ball at its centre.
LINE_SHAPES = (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER))
BOX = int(mujoco.mjtGeom.mjGEOM_BOX)


def shape_line(m: mujoco.MjModel, g: int) -> tuple[float, float]:
    """Half length of the centre line and the radius around it, for one collision shape.
    A box is a point with the radius of its corners."""
    kind = int(m.geom_type[g])
    if kind in LINE_SHAPES:
        return float(m.geom_size[g][1]), float(m.geom_size[g][0])
    if kind == BOX:
        return 0.0, float(np.linalg.norm(m.geom_size[g][:3]))
    return 0.0, float(m.geom_size[g][0])


class ToolKinematics:
    """Forward kinematics, the tool tip Jacobian, and a damped least squares step."""

    def __init__(self, model: mujoco.MjModel, site: str = "tool_tip", camera: str = "wrist_cam"):
        self.m = model
        self.d = mujoco.MjData(model)
        self.site = model.site(site).id
        self.cam = model.cam(camera).id
        self.nq = 6
        self.qlim = np.array([[model.jnt_range[j][0], model.jnt_range[j][1]] for j in range(self.nq)])
        self.limited = np.array([bool(model.jnt_limited[j]) for j in range(self.nq)])

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tool tip position and rotation for these joint angles."""
        self.d.qpos[:self.nq] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def camera_pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Where the wrist camera is, and which way it looks.

        mj_kinematics moves bodies and sites but not cameras, so this needs
        mj_camlight as well. Without it the pose stays at the origin with a zero
        rotation, and every hole lands behind the camera.
        """
        self.d.qpos[:self.nq] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_camlight(self.m, self.d)
        return self.d.cam_xpos[self.cam].copy(), self.d.cam_xmat[self.cam].reshape(3, 3).copy()

    def camera_fovy(self) -> float:
        return float(self.m.cam_fovy[self.cam])

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """6 x 6 tool tip Jacobian: linear rows then angular rows."""
        self.d.qpos[:self.nq] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, self.d, jp, jr, self.site)
        return np.vstack([jp[:, :self.nq], jr[:, :self.nq]])

    def velocity_to(self, q: np.ndarray, target_pos, target_R, gain_pos: float = 2.0,
                    gain_rot: float = 2.0, max_tool_speed: float = 0.25,
                    max_joint_speed: float = np.deg2rad(90.0), damping: float = 0.05) -> tuple[np.ndarray, float]:
        """Joint velocity that moves the tool tip toward a pose. Returns (qd, position error)."""
        pos, R = self.fk(q)
        e_pos = np.asarray(target_pos, float) - pos
        err = float(np.linalg.norm(e_pos))
        v = gain_pos * e_pos
        speed = float(np.linalg.norm(v))
        if speed > max_tool_speed:
            v *= max_tool_speed / speed
        if target_R is None:
            w = np.zeros(3)
        else:
            dR = np.asarray(target_R, float) @ R.T
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, dR.flatten())
            angle = 2.0 * np.arctan2(float(np.linalg.norm(quat[1:])), float(quat[0]))
            axis = quat[1:] / max(float(np.linalg.norm(quat[1:])), 1e-9)
            w = gain_rot * ((angle - 2 * np.pi) if angle > np.pi else angle) * axis
        J = self.jacobian(q)
        task = np.concatenate([v, w])
        qd = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), task)
        fastest = float(np.max(np.abs(qd)))
        if fastest > max_joint_speed:
            qd *= max_joint_speed / fastest
        return qd, err

    def clamp(self, q: np.ndarray) -> np.ndarray:
        out = q.copy()
        for j in range(self.nq):
            if self.limited[j]:
                out[j] = float(np.clip(out[j], self.qlim[j, 0], self.qlim[j, 1]))
        return out

    def nearest(self, q_goal: np.ndarray, q_now: np.ndarray) -> np.ndarray:
        """Same pose, fewest turns: of the whole-turn copies of each joint that lie inside
        its limits, take the one closest to where the joint is now.

        Never clamps. The first version shifted a joint past its limit and then clamped
        it, which is a different pose: a taught feeder pose moved 397 mm, onto the jig.
        """
        out = np.asarray(q_goal, float).copy()
        for j in range(self.nq):
            candidates = [out[j] + k * 2 * np.pi for k in (-2, -1, 0, 1, 2)]
            if self.limited[j]:
                candidates = [c for c in candidates if self.qlim[j, 0] - 1e-9 <= c <= self.qlim[j, 1] + 1e-9]
            if candidates:
                out[j] = min(candidates, key=lambda c: abs(c - q_now[j]))
        return out

    def ik_multi(self, target_pos, target_R, q_now: np.ndarray, iters: int = 300) -> tuple[np.ndarray, float]:
        """Solve from several starts and keep the answer that moves the joints least.

        One seed and a straight velocity step is not enough: driving the tool along a
        line walks joints into their limits and stalls there, which is what a real
        controller avoids by planning the joint move first.
        """
        best_q, best_err, best_cost = None, np.inf, np.inf
        seeds = [np.asarray(q_now, float)]
        for pan in (-np.pi / 2, 0.0, np.pi / 2, np.pi):
            for lift in (-2.2, -1.2):
                for elbow in (-1.8, 1.8):
                    seeds.append(np.array([pan, lift, elbow, -1.6, -1.57, 0.0]))
        for seed in seeds:
            q, err = self.ik(target_pos, target_R, seed, iters=iters)
            if err > 1e-4:
                continue
            q = self.nearest(q, q_now)
            q, err = self.ik(target_pos, target_R, q, iters=60)   # the shift may need a nudge back
            if err > 1e-4:
                continue
            cost = float(np.max(np.abs(q - q_now)))
            if cost < best_cost:
                best_q, best_err, best_cost = q, err, cost
        if best_q is None:
            return np.asarray(q_now, float), np.inf
        return best_q, best_err

    def ik_all(self, target_pos, target_R, q_ref: np.ndarray, extra_seeds=(), iters: int = 300,
               apart: float = 0.3) -> list:
        """Every distinct way the arm can put the tool there: the answers from the same
        starts as ik_multi and any extra ones, each shifted to the whole-turn copy
        nearest q_ref, nearest first. Two answers are the same arm configuration if no
        joint differs by more than apart."""
        seeds = [np.asarray(q_ref, float)] + [np.asarray(s, float) for s in extra_seeds]
        for pan in (-np.pi / 2, 0.0, np.pi / 2, np.pi):
            for lift in (-2.2, -1.2):
                for elbow in (-1.8, 1.8):
                    seeds.append(np.array([pan, lift, elbow, -1.6, -1.57, 0.0]))
        found = []
        for seed in seeds:
            q, err = self.ik(target_pos, target_R, seed, iters=iters)
            if err > 1e-4:
                continue
            q = self.nearest(q, q_ref)
            q, err = self.ik(target_pos, target_R, q, iters=60)
            if err > 1e-4:
                continue
            if all(float(np.max(np.abs(q - other))) > apart for other in found):
                found.append(q)
        found.sort(key=lambda q: float(np.max(np.abs(q - q_ref))))
        return found

    def ik(self, target_pos, target_R, seed: np.ndarray, iters: int = 200) -> tuple[np.ndarray, float]:
        """Damped least squares solve, starting from a seed. Returns (q, position error)."""
        q = np.asarray(seed, float).copy()
        err = np.inf
        for _ in range(iters):
            qd, err = self.velocity_to(q, target_pos, target_R, gain_pos=1.0, gain_rot=1.0,
                                       max_tool_speed=1e9, max_joint_speed=1e9, damping=0.02)
            if err < 1e-5:
                break
            q = self.clamp(q + 0.5 * qd)
        return q, err
