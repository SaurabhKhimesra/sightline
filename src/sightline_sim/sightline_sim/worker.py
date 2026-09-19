"""The worker: dm_control's CMU humanoid, scaled to a normal stature and dressed.

The humanoid's own capsules stay as the collision body that the rules are
judged on (group 3, hidden in beauty renders). Clothing, skin and hair are
visual meshes (group 1) fixed to the same bodies, so they move with every pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import geometry as G

HUMAN_PREFIX = "worker/"

# Stature of the dressed worker, metres. Scene choice inside the range of
# mean adult male height for the 1996 birth cohort, about 1.60 m to 1.825 m
# (NCD Risk Factor Collaboration, "A century of trends in adult human height",
# eLife 2016;5:e13410). The CMU humanoid as shipped is 2.05 m tall (measured).
WORKER_STATURE_M = 1.75

# Lower-body standing pose from CobotSafe (scene.py STANDING_POSE): feet about
# 0.17 m apart, toes forward. Re-used so both projects stand the model the same way.
STANDING_POSE = {"lfemurrz": -0.385, "rfemurrz": 0.385}

GROUP_WORKER = 1
GROUP_COLLISION = 3

WORKER_MATERIALS = {
    "w_jacket": dict(rgba=(0.10, 0.22, 0.44, 1), specular=0.12, shininess=0.15),
    "w_jacket_dark": dict(rgba=(0.06, 0.13, 0.27, 1), specular=0.10, shininess=0.15),
    "w_reflective": dict(rgba=(0.78, 0.80, 0.80, 1), specular=0.9, shininess=0.9),
    "w_trousers": dict(rgba=(0.13, 0.14, 0.16, 1), specular=0.08, shininess=0.1),
    "w_shoe": dict(rgba=(0.05, 0.05, 0.06, 1), specular=0.35, shininess=0.5),
    "w_sole": dict(rgba=(0.20, 0.20, 0.20, 1), specular=0.05, shininess=0.1),
    "w_glove": dict(rgba=(0.62, 0.63, 0.62, 1), specular=0.08, shininess=0.1),
    "w_glove_coat": dict(rgba=(0.24, 0.25, 0.27, 1), specular=0.2, shininess=0.3),
    "w_skin": dict(rgba=(0.78, 0.60, 0.48, 1), specular=0.12, shininess=0.3),
    "w_hair": dict(rgba=(0.09, 0.07, 0.05, 1), specular=0.25, shininess=0.3),
    "w_dark": dict(rgba=(0.03, 0.03, 0.03, 1), specular=0.3, shininess=0.4),
    "w_lens": dict(rgba=(0.75, 0.88, 0.97, 0.28), specular=1.0, shininess=1.0),
    "w_badge": dict(rgba=(0.95, 0.95, 0.93, 1), specular=0.2, shininess=0.3),
}


def cmu_humanoid_spec() -> tuple[mujoco.MjSpec, np.ndarray]:
    """The walker as an MjSpec, and its upright quaternion."""
    from dm_control.locomotion.walkers import cmu_humanoid

    walker = cmu_humanoid.CMUHumanoidPositionControlledV2020()
    spec = mujoco.MjSpec.from_string(walker.mjcf_model.to_xml_string(), walker.mjcf_model.get_assets())
    upright = np.array(walker.upright_pose.xquat, float)
    return spec, upright / np.linalg.norm(upright)


def _compile_alone(spec: mujoco.MjSpec, upright) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Compile a copy of the humanoid, upright, root at the origin, x forward, y left, z up."""
    host = mujoco.MjSpec()
    body = host.worldbody.add_body(name="w", quat=upright)
    host.attach(spec.copy(), prefix="h/", frame=body.add_frame())
    m = host.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def stature(spec: mujoco.MjSpec, upright) -> float:
    """Head top to sole in the standing pose, metres (capsule and sphere extents)."""
    m, d = _compile_alone(spec, upright)
    for j, a in STANDING_POSE.items():
        d.qpos[m.joint("h/" + j).qposadr[0]] = a
    mujoco.mj_kinematics(m, d)
    lo, hi = np.inf, -np.inf
    for g in range(m.ngeom):
        R = d.geom_xmat[g].reshape(3, 3)
        sz = m.geom_size[g]
        z = d.geom_xpos[g][2]
        kind = int(m.geom_type[g])
        if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            ext = abs(R[2, 2]) * sz[1] + sz[0]
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            ext = sz[0]
        else:
            ext = m.geom_rbound[g]
        lo, hi = min(lo, z - ext), max(hi, z + ext)
    return float(hi - lo)


def scale_humanoid(spec: mujoco.MjSpec, k: float) -> None:
    """Scale every length by k and every density by 1 / k^3, so the total mass is unchanged."""
    for b in spec.bodies:
        b.pos = np.asarray(b.pos) * k
    for g in spec.geoms:
        g.pos = np.asarray(g.pos) * k
        g.size = np.asarray(g.size) * k
        ft = np.asarray(g.fromto)
        if not np.isnan(ft[0]):
            g.fromto = ft * k
        if np.isnan(g.mass):  # mass from density
            g.density = g.density / k ** 3
    for j in spec.joints:
        j.pos = np.asarray(j.pos) * k
    for s in spec.sites:
        s.pos = np.asarray(s.pos) * k
        s.size = np.asarray(s.size) * k
    for c in spec.cameras:
        c.pos = np.asarray(c.pos) * k


@dataclass
class Clothing:
    """Visual pieces per body: (body name, mesh or primitive, material)."""
    pieces: list = field(default_factory=list)


class _Landmarks:
    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData):
        self.m, self.d = m, d

    def body(self, name: str) -> np.ndarray:
        return self.d.xpos[self.m.body("h/" + name).id].copy()

    def frame(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        i = self.m.body("h/" + name).id
        return self.d.xpos[i].copy(), self.d.xmat[i].reshape(3, 3).copy()

    def capsule(self, geom_index_in_body: int, body: str):
        b = self.m.body("h/" + body).id
        gs = [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == b]
        g = gs[geom_index_in_body]
        R = self.d.geom_xmat[g].reshape(3, 3)
        p = self.d.geom_xpos[g]
        r, half = self.m.geom_size[g][:2]
        return p - R[:, 2] * half, p + R[:, 2] * half, float(r)

    def geoms(self, body: str) -> list[int]:
        b = self.m.body("h/" + body).id
        return [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == b]


def _horizontal_loft(stations, k: float, n: int = 28, cap0: bool = True, cap1: bool = True) -> G.Mesh:
    """Loft along +z through (z, x centre, half depth along x, half width along y, exponent) in units of k."""
    secs = []
    for z, cx, hx, hy, p in stations:
        loop = G.superellipse(hx * k, hy * k, p, n)
        secs.append(np.column_stack([loop[:, 0] + cx * k, loop[:, 1], np.full(n, z * k)]))
    return G.loft(secs, cap0, cap1)


def build_clothing(spec: mujoco.MjSpec, upright, k: float) -> Clothing:
    """Clothing meshes in the zero pose of the scaled humanoid, expressed per body."""
    m, d = _compile_alone(spec, upright)
    L = _Landmarks(m, d)
    out = Clothing()

    def add(body: str, mesh: G.Mesh, material: str):
        out.pieces.append((body, mesh, material))

    # Jacket torso in two overlapping pieces so the spine can bend. The upper piece
    # starts slightly smaller inside the lower one to avoid coincident surfaces.
    lower = [(-0.045, -0.002, 0.127, 0.178, 2.6), (0.050, 0.008, 0.116, 0.163, 2.6),
             (0.140, 0.020, 0.121, 0.169, 2.6), (0.200, 0.030, 0.125, 0.178, 2.6)]
    upper = [(0.150, 0.022, 0.114, 0.160, 2.6), (0.240, 0.034, 0.128, 0.186, 2.6),
             (0.330, 0.050, 0.132, 0.199, 2.7), (0.420, 0.058, 0.125, 0.206, 2.8),
             (0.480, 0.062, 0.118, 0.228, 3.0), (0.525, 0.064, 0.104, 0.214, 2.8),
             (0.555, 0.066, 0.086, 0.160, 2.4), (0.578, 0.070, 0.070, 0.095, 2.2),
             (0.592, 0.073, 0.058, 0.066, 2.0)]
    add("lowerback", _horizontal_loft(lower, k, cap0=False), "w_jacket")
    add("thorax", _horizontal_loft(upper, k), "w_jacket")
    # reflective band at chest height and a collar
    band = [(0.300, 0.048, 0.1335, 0.2005, 2.7), (0.330, 0.050, 0.1335, 0.2015, 2.7)]
    add("thorax", _horizontal_loft(band, k), "w_reflective")
    collar = [(0.548, 0.066, 0.080, 0.098, 2.2), (0.590, 0.075, 0.066, 0.078, 2.0), (0.612, 0.080, 0.058, 0.066, 2.0)]
    add("thorax", _horizontal_loft(collar, k), "w_jacket_dark")
    badge = G.merge([G.prism(G.rounded_rect(0.012 * k, 0.050 * k, 0.003 * k), 0, 0.002 * k)])
    badge = G.Mesh(badge.v[:, [2, 1, 0]] * np.array([1, 1, 1]), badge.f[:, ::-1])
    add("thorax", badge.moved(pos=(0.188 * k, 0.085 * k, 0.40 * k)), "w_badge")

    # Trousers: pelvis block, thighs, shins with a hem over the shoe.
    pelvis = [(-0.175, -0.005, 0.090, 0.120, 2.2), (-0.130, -0.008, 0.118, 0.175, 2.6),
              (-0.070, -0.010, 0.123, 0.176, 2.6), (-0.015, -0.006, 0.112, 0.162, 2.6)]
    add("root", _horizontal_loft(pelvis, k), "w_trousers")
    for s in "lr":
        hip = L.body(f"{s}femur")
        knee = L.body(f"{s}tibia")
        ankle = L.body(f"{s}foot")
        add(f"{s}femur", G.limb(hip + (knee - hip) * -0.05, knee + (knee - hip) * 0.06,
                                [0.092 * k, 0.088 * k, 0.078 * k, 0.064 * k], hint=(1, 0, 0),
                                round_ends=(True, False)), "w_trousers")
        add(f"{s}tibia", G.limb(knee - (ankle - knee) * 0.07, ankle + (ankle - knee) * 0.02,
                                [0.063 * k, 0.062 * k, 0.056 * k, 0.052 * k, 0.057 * k], hint=(1, 0, 0),
                                round_ends=(True, False)), "w_trousers")
        _add_shoe(add, L, s, k)
        _add_arm(add, L, s, k)
    _add_head(add, L, k)
    return out


def _add_shoe(add, L: _Landmarks, s: str, k: float) -> None:
    a1, b1, r = L.capsule(0, f"{s}foot")
    a2, b2, _ = L.capsule(1, f"{s}foot")
    ends = [a1, b1, a2, b2]
    heel = min(ends, key=lambda p: p[0])
    toe = max(ends, key=lambda p: p[0])
    heel = (heel + [e for e in ends if e[0] < 0.0][-1]) / 2 if len([e for e in ends if e[0] < 0.0]) > 1 else heel
    toes = [L.d.geom_xpos[g] for g in L.geoms(f"{s}toes")]
    toe_tip = np.mean(toes, axis=0)
    e1 = toe_tip - heel
    e1 /= np.linalg.norm(e1)
    e3 = np.array([0.0, 0.0, 1.0]) - e1[2] * e1
    e3 /= np.linalg.norm(e3)
    e2 = np.cross(e3, e1)
    length = float(np.linalg.norm(toe_tip - heel)) + 0.030 * k + 0.034 * k
    start = heel - e1 * 0.034 * k - e3 * (r + 0.004 * k)
    stations = [(0.00, 0.034, 0.036), (0.06, 0.042, 0.046), (0.30, 0.046, 0.052),
                (0.60, 0.050, 0.038), (0.85, 0.047, 0.031), (1.00, 0.030, 0.022)]
    F = np.column_stack([e2, e3, e1])
    secs, sole = [], []
    for t, w, h in stations:
        loop = G.superellipse(w * k, h * k, 2.6, 24)
        c = start + e1 * t * length + e3 * h * k
        secs.append(G.section(c, F, loop))
        loop_s = G.superellipse(w * k * 1.05, 0.006 * k, 3.0, 24)
        sole.append(G.section(start + e1 * t * length + e3 * 0.005 * k, F, loop_s))
    add(f"{s}foot", G.loft(secs), "w_shoe")
    add(f"{s}foot", G.loft(sole), "w_sole")


def _add_arm(add, L: _Landmarks, s: str, k: float) -> None:
    sh = L.body(f"{s}humerus")
    el = L.body(f"{s}radius")
    wr = L.body(f"{s}wrist")
    hd = L.body(f"{s}hand")
    out = np.sign(el[1] - sh[1])  # +1 left arm (along +y), -1 right arm
    along = np.array([0.0, out, 0.0])
    add(f"{s}humerus", G.limb(sh + along * 0.012 * k, el + along * 0.016 * k,
                              [0.046 * k, 0.053 * k, 0.050 * k, 0.047 * k], hint=(0, 0, 1),
                              squash=0.95, round_ends=(True, False), round_scale=0.6), "w_jacket")
    add(f"{s}humerus", G.limb(sh + along * 0.080 * k, sh + along * 0.105 * k,
                              [0.0555 * k, 0.0545 * k], hint=(0, 0, 1), squash=0.95,
                              round_ends=(False, False)), "w_reflective")
    add(f"{s}radius", G.limb(el - along * 0.018 * k, wr - along * 0.012 * k,
                             [0.050 * k, 0.047 * k, 0.042 * k, 0.038 * k], hint=(0, 0, 1),
                             squash=0.92, round_ends=(True, False), round_scale=0.6), "w_jacket")
    add(f"{s}radius", G.limb(wr - along * 0.030 * k, wr - along * 0.004 * k,
                             [0.041 * k, 0.041 * k], hint=(0, 0, 1), squash=0.9,
                             round_ends=(False, False)), "w_jacket_dark")
    add(f"{s}wrist", G.limb(wr - along * 0.012 * k, wr + along * 0.048 * k,
                            [0.031 * k, 0.029 * k], hint=(0, 0, 1), squash=0.8,
                            round_ends=(False, False)), "w_glove")
    # palm: loft along the arm, flat in z
    secs = []
    n = 24
    for dy, hw, ht in ((0.030, 0.029, 0.019), (0.075, 0.040, 0.022), (0.125, 0.045, 0.021), (0.168, 0.043, 0.015)):
        loop = G.superellipse(ht * k, hw * k, 2.6, n)  # (u = z, v = x) runs counter-clockwise about +y
        c = wr + along * dy * k
        pts = np.column_stack([c[0] + loop[:, 1], np.full(n, c[1]), c[2] + loop[:, 0]])
        secs.append(pts if out > 0 else pts[::-1])
    add(f"{s}hand", G.loft(secs), "w_glove_coat")
    for g in L.geoms(f"{s}fingers"):
        R = L.d.geom_xmat[g].reshape(3, 3)
        p = L.d.geom_xpos[g]
        half = L.m.geom_size[g][1]
        a, b = p - R[:, 2] * half, p + R[:, 2] * half
        base, tip = (a, b) if abs(a[1]) < abs(b[1]) else (b, a)
        add(f"{s}fingers", G.limb(base - (tip - base) * 0.15, tip, [0.0105 * k, 0.0098 * k, 0.0088 * k],
                                  hint=(0, 0, 1), squash=0.85), "w_glove")
    a, b, _ = L.capsule(0, f"{s}thumb")
    base, tip = (a, b) if abs(a[1]) < abs(b[1]) else (b, a)
    add(f"{s}thumb", G.limb(base - (tip - base) * 0.2, tip, [0.0125 * k, 0.0105 * k], hint=(0, 0, 1)), "w_glove")


def _add_head(add, L: _Landmarks, k: float) -> None:
    neck0 = L.body("lowerneck")
    add("upperneck", G.limb(neck0 + np.array([0.012, 0, 0.050]) * k, np.array([0.100, 0.0, 0.640]) * k,
                            [0.050 * k, 0.047 * k, 0.046 * k], hint=(1, 0, 0), squash=1.05), "w_skin")
    head = [(0.600, 0.128, 0.040, 0.036, 2.0), (0.622, 0.150, 0.072, 0.054, 2.2),
            (0.662, 0.152, 0.090, 0.066, 2.3), (0.702, 0.147, 0.099, 0.074, 2.4),
            (0.745, 0.141, 0.102, 0.078, 2.4), (0.790, 0.136, 0.100, 0.078, 2.3),
            (0.830, 0.131, 0.086, 0.068, 2.2), (0.857, 0.128, 0.052, 0.043, 2.0),
            (0.867, 0.126, 0.016, 0.014, 2.0)]
    add("head", _horizontal_loft(head, k, 32), "w_skin")
    hair = [(0.690, 0.086, 0.060, 0.071, 2.2), (0.740, 0.110, 0.093, 0.083, 2.4),
            (0.790, 0.124, 0.105, 0.083, 2.4), (0.835, 0.127, 0.093, 0.073, 2.3),
            (0.866, 0.125, 0.059, 0.047, 2.1), (0.880, 0.123, 0.020, 0.016, 2.0)]
    add("head", _horizontal_loft(hair, k, 32), "w_hair")
    # primitives: (kind, size, pos, material)
    add("head", ("ellipsoid", (0.013, 0.010, 0.022), (0.245, 0.0, 0.706)), "w_skin")
    for side in (1, -1):
        add("head", ("ellipsoid", (0.011, 0.006, 0.021), (0.130, side * 0.079, 0.722)), "w_skin")
        add("head", ("sphere", (0.0062,), (0.2365, side * 0.034, 0.748)), "w_dark")
        add("head", ("box", (0.004, 0.017, 0.0022), (0.2395, side * 0.034, 0.766)), "w_hair")
        add("head", ("box", (0.050, 0.0022, 0.0028), (0.200, side * 0.083, 0.752)), "w_dark")
    add("head", ("box", (0.0025, 0.072, 0.021), (0.2555, 0.0, 0.748)), "w_lens")
    add("head", ("box", (0.004, 0.074, 0.0035), (0.2565, 0.0, 0.770)), "w_dark")
    add("head", ("box", (0.003, 0.012, 0.0025), (0.2575, 0.0, 0.738)), "w_dark")


def dress(spec: mujoco.MjSpec, upright, k: float) -> None:
    """Hide the collision capsules from beauty renders and fix the clothing to the bodies."""
    for name, props in WORKER_MATERIALS.items():
        mat = spec.add_material(name=name)
        for key, val in props.items():
            setattr(mat, key, val)
    clothing = build_clothing(spec, upright, k)
    m, d = _compile_alone(spec, upright)
    for g in spec.geoms:
        g.group = GROUP_COLLISION
    for lt in list(spec.lights):
        spec.delete(lt)
    counter = 0
    for body_name, piece, material in clothing.pieces:
        bid = m.body("h/" + body_name).id
        p_b = d.xpos[bid]
        R_b = d.xmat[bid].reshape(3, 3)
        body = spec.body(body_name)
        counter += 1
        common = dict(group=GROUP_WORKER, contype=0, conaffinity=0, density=0.0, material=material)
        if isinstance(piece, G.Mesh):
            local = G.Mesh((piece.v - p_b) @ R_b, piece.f)
            mesh = spec.add_mesh(name=f"cloth_{counter}")
            mesh.uservert = local.v.ravel().tolist()
            mesh.userface = local.f.ravel().tolist()
            mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name, name=f"cloth_{counter}", **common)
        else:
            kind, size, pos = piece
            world = np.asarray(pos, float) * k
            gtype = {"box": mujoco.mjtGeom.mjGEOM_BOX, "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
                     "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID}[kind]
            sz = np.zeros(3)
            sz[:len(size)] = np.asarray(size, float) * k
            body.add_geom(type=gtype, size=sz, pos=(world - p_b) @ R_b,
                          quat=G.mat_to_quat(R_b.T), name=f"cloth_{counter}", **common)


def make_worker() -> tuple[mujoco.MjSpec, np.ndarray, float]:
    """Scaled, dressed humanoid spec, its upright quaternion and the scale factor."""
    spec, upright = cmu_humanoid_spec()
    k = WORKER_STATURE_M / stature(spec, upright)
    scale_humanoid(spec, k)
    dress(spec, upright, k)
    return spec, upright, k


def attach_worker(scene: mujoco.MjSpec, pos, yaw: float) -> tuple[float, np.ndarray]:
    """Add the worker on a free joint. Returns (scale, upright quaternion)."""
    spec, upright, k = make_worker()
    body = scene.worldbody.add_body(name="worker_base", pos=pos)
    body.add_freejoint(name="worker_free")
    frame = body.add_frame(quat=upright)
    scene.attach(spec, prefix=HUMAN_PREFIX, frame=frame)
    return k, upright
