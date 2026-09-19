"""Posing: stand the worker with flat feet, reach hands to targets, aim the head, and robot IK."""

from __future__ import annotations

import mujoco
import numpy as np

from .worker import HUMAN_PREFIX, STANDING_POSE, _compile_alone, cmu_humanoid_spec, scale_humanoid

ROBOT_PREFIX = "ur5e/"
UR5E_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")

ARM_CHAIN = ("claviclerz", "claviclery", "humerusrz", "humerusry", "humerusrx", "radiusrx", "wristry", "handrz", "handrx")
SPINE = ("lowerbackrx", "upperbackrx", "thoraxrx")
NECK = ("lowerneckrx", "lowerneckry", "upperneckrx", "upperneckry", "headrx", "headry")


def _arms_at_rest() -> dict:
    """CobotSafe's measured rest pose for hanging arms (cobotsafe.human_motion.ARMS_AT_REST)."""
    from sightline_sim.human import ARMS_AT_REST
    return dict(ARMS_AT_REST)


def _foot_pitch_flat(m, d, prefix: str, side: str) -> float:
    """footrx angle that puts the heel end of the foot capsules level with the toe spheres."""
    j = m.joint(f"{prefix}{side}footrx")
    foot = m.body(f"{prefix}{side}foot").id
    toes = m.body(f"{prefix}{side}toes").id
    foot_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == foot and m.geom_group[g] == 3]
    toe_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == toes and m.geom_group[g] == 3]

    def gap(angle):
        d.qpos[j.qposadr[0]] = angle
        mujoco.mj_kinematics(m, d)
        ends = []
        for g in foot_geoms:
            Rg = d.geom_xmat[g].reshape(3, 3)
            h = m.geom_size[g][1]
            ends += [d.geom_xpos[g] - Rg[:, 2] * h, d.geom_xpos[g] + Rg[:, 2] * h]
        heel_z = max(e[2] for e in ends)
        toe_z = min(d.geom_xpos[g][2] for g in toe_geoms)
        return heel_z - toe_z

    lo, hi = -0.78, 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if gap(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def measure_elbow_height(k: float) -> float:
    """Elbow height above the soles for the scaled humanoid standing with flat feet and hanging arms."""
    spec, upright = cmu_humanoid_spec()
    scale_humanoid(spec, k)
    for g in spec.geoms:
        g.group = 3
    m, d = _compile_alone(spec, upright)
    prefix = "h/"
    for j, a in {**STANDING_POSE, **_arms_at_rest()}.items():
        d.qpos[m.joint(prefix + j).qposadr[0]] = a
    for s in "lr":
        d.qpos[m.joint(f"{prefix}{s}footrx").qposadr[0]] = _foot_pitch_flat(m, d, prefix, s)
    mujoco.mj_kinematics(m, d)
    sole = min(d.geom_xpos[g][2] - m.geom_rbound[g] for g in range(m.ngeom)
               if m.body(m.geom_bodyid[g]).name.startswith((prefix + "lfoot", prefix + "rfoot", prefix + "ltoes", prefix + "rtoes")))
    elbows = [d.xpos[m.body(f"{prefix}{s}radius").id][2] for s in "lr"]
    return float(np.mean(elbows) - sole)


class WorkerPoser:
    """Sets worker joint angles in a compiled scene."""

    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData):
        self.m, self.d = m, d
        self.free = m.joint("worker_free")

    def j(self, name: str):
        return self.m.joint(HUMAN_PREFIX + name)

    def set(self, name: str, angle: float) -> None:
        jj = self.j(name)
        lo, hi = self.m.jnt_range[jj.id]
        self.d.qpos[jj.qposadr[0]] = np.clip(angle, lo, hi) if lo < hi else angle

    def get(self, name: str) -> float:
        return float(self.d.qpos[self.j(name).qposadr[0]])

    def stand(self, xy, yaw: float, hang=()) -> None:
        """hang names the arms, "l" or "r", that have nothing to reach and hang."""
        m, d = self.m, self.d
        # a hanging arm goes back to hanging whole, elbow, wrist and collarbone included.
        # Resetting only the shoulder left the elbow bent from the last reach at the
        # bench, and on the later trips to the rack the "hanging" right hand stood 1.42 m
        # up behind him, reaching into the robot's side of the cell. An arm with a target
        # keeps its joints as the starting guess: from a straight arm the left hand's
        # reach to its rest spot on the worktop missed by 54 cm.
        for side in hang:
            for n in ARM_CHAIN:
                self.set(f"{side}{n}", 0.0)
        for name, angle in {**STANDING_POSE, **_arms_at_rest()}.items():
            self.set(name, angle)
        a = self.free.qposadr[0]
        d.qpos[a:a + 3] = [xy[0], xy[1], 1.0]
        d.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        for s in "lr":
            self.set(f"{s}footrx", _foot_pitch_flat(m, d, HUMAN_PREFIX, s))
        self.settle_on_floor()

    def settle_on_floor(self) -> None:
        m, d = self.m, self.d
        mujoco.mj_kinematics(m, d)
        feet = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(
            tuple(HUMAN_PREFIX + p for p in ("lfoot", "rfoot", "ltoes", "rtoes"))) and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
        lowest = np.inf
        for g in feet:
            mesh = m.geom_dataid[g]
            adr, n = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            v = m.mesh_vert[adr:adr + n]
            world = d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T
            lowest = min(lowest, float(world[:, 2].min()))
        d.qpos[self.free.qposadr[0] + 2] -= lowest
        mujoco.mj_kinematics(m, d)

    def lean(self, total: float) -> None:
        for name, share in zip(SPINE, (0.45, 0.35, 0.20)):
            self.set(name, total * share)

    def _ik(self, joints, target_fn, iters: int = 300, damping: float = 1e-3) -> float:
        m, d = self.m, self.d
        ids = [self.j(n).id for n in joints]
        dofs = [m.jnt_dofadr[i] for i in ids]
        qadr = [m.jnt_qposadr[i] for i in ids]
        err = np.inf
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            e, J = target_fn()
            err = float(np.linalg.norm(e))
            if err < 1e-5:
                break
            Jr = J[:, dofs]
            dq = Jr.T @ np.linalg.solve(Jr @ Jr.T + damping * np.eye(len(e)), e)
            for i, qa, step in zip(ids, qadr, dq):
                lo, hi = m.jnt_range[i]
                d.qpos[qa] = np.clip(d.qpos[qa] + 0.6 * step, lo, hi)
        mujoco.mj_kinematics(m, d)
        return err

    def _ik_priority(self, joints, primary_fn, secondary_fn, iters: int = 600, damping: float = 1e-4) -> None:
        """Two-level damped least squares: the secondary task only uses the primary task's null space."""
        m, d = self.m, self.d
        ids = [self.j(n).id for n in joints]
        dofs = [m.jnt_dofadr[i] for i in ids]
        qadr = [m.jnt_qposadr[i] for i in ids]
        n = len(ids)
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            e1, J1 = primary_fn()
            e2, J2 = secondary_fn()
            J1, J2 = J1[:, dofs], J2[:, dofs]
            J1p = J1.T @ np.linalg.inv(J1 @ J1.T + damping * np.eye(len(e1)))
            dq1 = J1p @ e1
            N = np.eye(n) - J1p @ J1
            J2n = J2 @ N
            J2np = J2n.T @ np.linalg.inv(J2n @ J2n.T + 1e-2 * np.eye(len(e2)))
            dq = dq1 + N @ (J2np @ (e2 - J2 @ dq1))
            if np.max(np.abs(dq)) < 1e-7:
                break
            for i, qa, step in zip(ids, qadr, dq):
                lo, hi = m.jnt_range[i]
                d.qpos[qa] = np.clip(d.qpos[qa] + 0.5 * step, lo, hi)
        mujoco.mj_kinematics(m, d)

    def seed_arm(self, pose: dict) -> None:
        for name, angle in pose.items():
            self.set(name, angle)

    def reach(self, side: str, target, palm_down=True, finger_dir=None) -> float:
        """Hand body to target, fingers along finger_dir, then the palm turned down. Returns the position error, m.

        Position is the primary task; finger direction and palm orientation use its
        null space. One weighted solve of all three got stuck 0.3 m from the target.
        """
        m, d = self.m, self.d
        hand = m.body(f"{HUMAN_PREFIX}{side}hand").id
        fingers = m.body(f"{HUMAN_PREFIX}{side}fingers").id
        thumb = m.body(f"{HUMAN_PREFIX}{side}thumb").id
        target = np.asarray(target, float)
        arm = [f"{side}{n}" for n in ARM_CHAIN]
        jacp, jacr, jacf = np.zeros((3, m.nv)), np.zeros((3, m.nv)), np.zeros((3, m.nv))

        def position():
            mujoco.mj_jacBody(m, d, jacp, None, hand)
            return target - d.xpos[hand], jacp.copy()

        def orientation():
            e, J = [], []
            if finger_dir is not None:
                mujoco.mj_jacBody(m, d, jacp, None, hand)
                mujoco.mj_jacBody(m, d, jacf, None, fingers)
                v = d.xpos[fingers] - d.xpos[hand]
                want = np.asarray(finger_dir, float)
                want = want / np.linalg.norm(want) * np.linalg.norm(v)
                e.append(want - v)
                J.append(jacf - jacp)
            if palm_down:
                along = d.xpos[fingers] - d.xpos[hand]
                across = d.xpos[thumb] - d.xpos[hand]
                nrm = np.cross(along, across) if side == "l" else np.cross(across, along)
                nrm /= np.linalg.norm(nrm)
                mujoco.mj_jacBody(m, d, None, jacr, hand)
                e.append(0.1 * np.cross(nrm, np.array([0.0, 0.0, -1.0])))
                J.append(0.1 * jacr.copy())
            if not e:
                return np.zeros(1), np.zeros((1, m.nv))
            return np.concatenate(e), np.vstack(J)

        self._ik(arm, position)
        self._ik_priority(arm, position, orientation, iters=300)
        # clipping at joint limits can break the primary task, so finish on position alone
        self._ik(arm, position)
        return float(np.linalg.norm(target - d.xpos[hand]))

    def _find_palm_axis(self, side: str) -> np.ndarray:
        """Local hand axis pointing out of the palm (toward the thumb side's palm face)."""
        m, d = self.m, self.d
        mujoco.mj_kinematics(m, d)
        hand = m.body(f"{HUMAN_PREFIX}{side}hand").id
        thumb = m.body(f"{HUMAN_PREFIX}{side}thumb").id
        fingers = m.body(f"{HUMAN_PREFIX}{side}fingers").id
        R = d.xmat[hand].reshape(3, 3)
        along = d.xpos[fingers] - d.xpos[hand]
        across = d.xpos[thumb] - d.xpos[hand]
        normal = np.cross(along, across) if side == "l" else np.cross(across, along)
        normal /= np.linalg.norm(normal)
        return R.T @ normal

    def look_at(self, point) -> float:
        m, d = self.m, self.d
        head = m.body(HUMAN_PREFIX + "head").id
        face = m.body(HUMAN_PREFIX + "face").id
        point = np.asarray(point, float)
        jacr = np.zeros((3, m.nv))

        def fn():
            fwd = d.xpos[face] - d.xpos[head]
            fwd /= np.linalg.norm(fwd)
            want = point - d.xpos[head]
            want /= np.linalg.norm(want)
            mujoco.mj_jacBody(m, d, None, jacr, head)
            return np.cross(fwd, want), jacr.copy()

        return self._ik(list(NECK), fn, iters=200)


def robot_qadr(m: mujoco.MjModel) -> np.ndarray:
    return np.array([m.joint(ROBOT_PREFIX + j).qposadr[0] for j in UR5E_JOINTS])


def robot_ik(m: mujoco.MjModel, d: mujoco.MjData, site: str, target_pos, target_R, seeds, iters: int = 400):
    """Damped least squares on a site pose over the six UR5e joints, from several seeds.

    Returns a list of (error, q) sorted by error.
    """
    sid = m.site(site).id
    ids = [m.joint(ROBOT_PREFIX + j).id for j in UR5E_JOINTS]
    dofs = [m.jnt_dofadr[i] for i in ids]
    qadr = [m.jnt_qposadr[i] for i in ids]
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    target_pos = np.asarray(target_pos, float)
    target_R = np.asarray(target_R, float)
    results = []
    for seed in seeds:
        d.qpos[qadr] = seed
        err = np.inf
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p = d.site_xpos[sid]
            R = d.site_xmat[sid].reshape(3, 3)
            e_rot = 0.5 * sum(np.cross(R[:, i], target_R[:, i]) for i in range(3))
            e = np.concatenate([target_pos - p, e_rot])
            err = float(np.linalg.norm(e))
            if err < 1e-6:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, sid)
            J = np.vstack([jacp[:, dofs], jacr[:, dofs]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            for i, qa, step in zip(ids, qadr, dq):
                lo, hi = m.jnt_range[i]
                d.qpos[qa] = np.clip(d.qpos[qa] + 0.5 * step, lo, hi)
        results.append((err, d.qpos[qadr].copy()))
    results.sort(key=lambda r: r[0])
    return results
