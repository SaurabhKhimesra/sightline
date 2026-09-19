"""Mesh builders for the station props, in numpy.

Every builder returns a Mesh: vertices (N, 3) in metres and triangles (M, 3)
wound counter-clockwise seen from outside, with optional texture coordinates.
These are visual meshes. Collision uses MuJoCo primitives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh:
    v: np.ndarray
    f: np.ndarray
    uv: np.ndarray | None = None

    def __post_init__(self):
        self.v = np.asarray(self.v, float).reshape(-1, 3)
        self.f = np.asarray(self.f, int).reshape(-1, 3)
        if self.uv is not None:
            self.uv = np.asarray(self.uv, float).reshape(-1, 2)

    def moved(self, rot=None, pos=None) -> "Mesh":
        v = self.v if rot is None else self.v @ np.asarray(rot, float).T
        if pos is not None:
            v = v + np.asarray(pos, float)
        return Mesh(v, self.f.copy(), None if self.uv is None else self.uv.copy())

    def scaled(self, s) -> "Mesh":
        return Mesh(self.v * np.asarray(s, float), self.f.copy(), None if self.uv is None else self.uv.copy())


def merge(meshes) -> Mesh:
    meshes = [m for m in meshes if m is not None and len(m.f)]
    with_uv = all(m.uv is not None for m in meshes)
    vs, fs, uvs, n = [], [], [], 0
    for m in meshes:
        vs.append(m.v)
        fs.append(m.f + n)
        n += len(m.v)
        if with_uv:
            uvs.append(m.uv)
    return Mesh(np.vstack(vs), np.vstack(fs), np.vstack(uvs) if with_uv else None)


# ---------------------------------------------------------------- rotations

def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def frame_from_axis(w, hint=(1.0, 0.0, 0.0)) -> np.ndarray:
    """Right-handed frame (columns u, v, w) with w along the given axis."""
    w = np.asarray(w, float)
    w = w / np.linalg.norm(w)
    h = np.asarray(hint, float)
    if abs(float(h @ w)) > 0.95:
        h = np.array([0.0, 0.0, 1.0]) if abs(w[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = h - (h @ w) * w
    u /= np.linalg.norm(u)
    return np.column_stack([u, np.cross(w, u), w])


def mat_to_quat(R) -> np.ndarray:
    """Rotation matrix to MuJoCo quaternion (w, x, y, z)."""
    R = np.asarray(R, float)
    q = np.empty(4)
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        q[:] = [0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2.0
        q[i + 1] = 0.25 * s
        q[0] = (R[k, j] - R[j, k]) / s
        q[j + 1] = (R[j, i] + R[i, j]) / s
        q[k + 1] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q)


def look_at_xyaxes(pos, target, up=(0.0, 0.0, 1.0)) -> list[float]:
    """MuJoCo camera xyaxes for a camera at pos looking at target."""
    fwd = np.asarray(target, float) - np.asarray(pos, float)
    fwd /= np.linalg.norm(fwd)
    x = np.cross(fwd, np.asarray(up, float))
    if np.linalg.norm(x) < 1e-9:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, fwd)
    return [*x, *y]


# ---------------------------------------------------------------- 2D loops

def signed_area(poly) -> float:
    p = np.asarray(poly, float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def circle(r: float, n: int = 32, phase: float = 0.0) -> np.ndarray:
    t = phase + 2 * np.pi * np.arange(n) / n
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def superellipse(a: float, b: float, p: float = 2.0, n: int = 32) -> np.ndarray:
    t = 2 * np.pi * np.arange(n) / n
    c, s = np.cos(t), np.sin(t)
    return np.column_stack([a * np.sign(c) * np.abs(c) ** (2 / p), b * np.sign(s) * np.abs(s) ** (2 / p)])


def rounded_rect(w: float, h: float, r: float, n_corner: int = 6) -> np.ndarray:
    """Counter-clockwise rectangle centred at the origin with rounded corners."""
    r = min(max(r, 0.0), w / 2, h / 2)
    if r < 1e-9:
        return np.array([[w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2], [-w / 2, -h / 2]])
    pts = []
    for cx, cy, a0 in ((w / 2 - r, h / 2 - r, 0.0), (-w / 2 + r, h / 2 - r, np.pi / 2),
                       (-w / 2 + r, -h / 2 + r, np.pi), (w / 2 - r, -h / 2 + r, 1.5 * np.pi)):
        for i in range(n_corner + 1):
            a = a0 + (np.pi / 2) * i / n_corner
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return np.array(pts)


def stadium(length: float, width: float, n_end: int = 10) -> np.ndarray:
    """Slot outline along x: two half circles joined by straight sides."""
    r = width / 2
    c = max(length / 2 - r, 0.0)
    right = [(c + r * np.cos(a), r * np.sin(a)) for a in np.linspace(-np.pi / 2, np.pi / 2, n_end + 1)]
    left = [(-c + r * np.cos(a), r * np.sin(a)) for a in np.linspace(np.pi / 2, 1.5 * np.pi, n_end + 1)]
    return np.array(right + left)


def drop_collinear(poly, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(poly, float)
    keep = []
    n = len(p)
    for i in range(n):
        a, b, c = p[i - 1], p[i], p[(i + 1) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) > eps and np.linalg.norm(b - a) > 1e-12:
            keep.append(i)
    return p[keep]


def ear_clip(poly) -> np.ndarray:
    """Triangle indices for a simple counter-clockwise polygon without holes."""
    pts = np.asarray(poly, float)
    idx = list(range(len(pts)))
    tris = []

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    while len(idx) > 3:
        n = len(idx)
        for k in range(n):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % n]
            a, b, c = pts[i0], pts[i1], pts[i2]
            if cross(a, b, c) <= 1e-14:
                continue
            # A vertex inside the ear or on its new diagonal blocks it. Axis-aligned
            # profiles put many vertices exactly on candidate diagonals.
            blocked = False
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                q = pts[j]
                if min(np.linalg.norm(q - a), np.linalg.norm(q - b), np.linalg.norm(q - c)) < 1e-12:
                    continue
                if cross(a, b, q) >= -1e-14 and cross(b, c, q) >= -1e-14 and cross(c, a, q) >= -1e-14:
                    blocked = True
                    break
            if not blocked:
                tris.append((i0, i1, i2))
                idx.pop(k)
                break
        else:
            raise ValueError("ear clipping failed: polygon is not simple or not counter-clockwise")
    tris.append(tuple(idx))
    return np.array(tris)


# ---------------------------------------------------------------- walls and solids

def _corner_copies(loop2d, closed: bool, sharp_deg: float):
    """Expand a polyline so sharp corners get two vertices (one per adjacent edge).

    Returns (points, out_index, in_index): out_index[i] is the copy used by the
    edge leaving point i, in_index[i] by the edge arriving at it.
    """
    p = np.asarray(loop2d, float)
    n = len(p)
    pts, out_i, in_i = [], [], []
    for i in range(n):
        has_prev = closed or i > 0
        has_next = closed or i < n - 1
        sharp = False
        if has_prev and has_next:
            d0 = p[i] - p[i - 1]
            d1 = p[(i + 1) % n] - p[i]
            n0, n1 = np.linalg.norm(d0), np.linalg.norm(d1)
            if n0 > 1e-12 and n1 > 1e-12:
                cosang = float(np.clip(d0 @ d1 / (n0 * n1), -1, 1))
                sharp = np.degrees(np.arccos(cosang)) > sharp_deg
        if sharp:
            in_i.append(len(pts))
            pts.append(p[i])
            out_i.append(len(pts))
            pts.append(p[i])
        else:
            in_i.append(len(pts))
            out_i.append(len(pts))
            pts.append(p[i])
    return np.array(pts), out_i, in_i


def wall(loop2d, z0: float, z1: float, sharp_deg: float = 40.0, inward: bool = False) -> Mesh:
    """Vertical wall along a closed counter-clockwise loop, facing outward (or inward)."""
    pts, out_i, in_i = _corner_copies(loop2d, True, sharp_deg)
    m = len(pts)
    v = np.vstack([np.column_stack([pts, np.full(m, z0)]), np.column_stack([pts, np.full(m, z1)])])
    n = len(out_i)
    f = []
    for i in range(n):
        a, b = out_i[i], in_i[(i + 1) % n]
        f += [(a, b, b + m), (a, b + m, a + m)]
    f = np.array(f)
    return Mesh(v, f[:, ::-1] if inward else f)


def cap(loop2d, z: float, up: bool) -> Mesh:
    p = np.asarray(loop2d, float)
    if signed_area(p) < 0:
        p = p[::-1]
    p = drop_collinear(p)
    t = ear_clip(p)
    return Mesh(np.column_stack([p, np.full(len(p), z)]), t if up else t[:, ::-1])


def prism(poly, z0: float, z1: float, sharp_deg: float = 40.0) -> Mesh:
    """Closed prism over a simple polygon between heights z0 and z1."""
    p = np.asarray(poly, float)
    if signed_area(p) < 0:
        p = p[::-1]
    return merge([cap(p, z0, up=False), cap(p, z1, up=True), wall(p, z0, z1, sharp_deg)])


def extrude_x(poly_yz, x0: float, x1: float, sharp_deg: float = 40.0) -> Mesh:
    """Prism of a polygon given in the (y, z) plane, extruded along x."""
    m = prism(poly_yz, x0, x1, sharp_deg)
    # (y, z, x) -> (x, y, z) is a cyclic permutation, so the winding stays outward.
    return Mesh(m.v[:, [2, 0, 1]], m.f)


def ring_prism(outer2d, inner2d, z0: float, z1: float, sharp_deg: float = 40.0) -> Mesh:
    """Hollow wall between two loops with the same point count (like a box without lid or floor)."""
    o = np.asarray(outer2d, float)
    i = np.asarray(inner2d, float)
    n = len(o)
    top = []
    vt = np.vstack([np.column_stack([o, np.full(n, z1)]), np.column_stack([i, np.full(n, z1)])])
    for k in range(n):
        j = (k + 1) % n
        top += [(k, j, n + j), (k, n + j, n + k)]
    top = Mesh(vt, top)
    bottom = top.moved(pos=(0, 0, z0 - z1))
    bottom = Mesh(bottom.v, bottom.f[:, ::-1])
    return merge([top, bottom, wall(o, z0, z1, sharp_deg), wall(i, z0, z1, sharp_deg, inward=True)])


def lathe(profile, n: int = 32, sharp_deg: float = 35.0) -> Mesh:
    """Surface of revolution about z. profile: (r, z) points; an end with r = 0 closes at the axis.

    Walk the profile with the solid on the left in the (r, z) plane, for
    example outward along a bottom face, then up the outside.
    """
    prof = np.asarray(profile, float)
    pts, out_i, in_i = _corner_copies(prof, False, sharp_deg)
    ang = 2 * np.pi * np.arange(n) / n
    rings, verts = [], []
    for r, z in pts:
        if r < 1e-9:
            rings.append([len(verts)])
            verts.append((0.0, 0.0, z))
        else:
            rings.append(list(range(len(verts), len(verts) + n)))
            verts += [(r * np.cos(a), r * np.sin(a), z) for a in ang]
    f = []
    for k in range(len(prof) - 1):
        A, B = rings[out_i[k]], rings[in_i[k + 1]]
        if len(A) == 1 and len(B) == 1:
            continue
        for i in range(n):
            j = (i + 1) % n
            if len(A) == 1:
                f.append((A[0], B[j], B[i]))
            elif len(B) == 1:
                f.append((A[i], A[j], B[0]))
            else:
                f += [(A[i], A[j], B[j]), (A[i], B[j], B[i])]
    return compact(Mesh(np.array(verts), np.array(f)))


def compact(mesh: Mesh) -> Mesh:
    """Drop vertices no triangle uses."""
    used = np.unique(mesh.f)
    remap = -np.ones(len(mesh.v), int)
    remap[used] = np.arange(len(used))
    return Mesh(mesh.v[used], remap[mesh.f], None if mesh.uv is None else mesh.uv[used])


def tube(r_out: float, r_in: float, h: float, n: int = 32) -> Mesh:
    """Hollow cylinder from z = 0 to h."""
    return lathe([(r_in, 0.0), (r_out, 0.0), (r_out, h), (r_in, h), (r_in, 0.0)], n)


def cylinder(r: float, h: float, n: int = 32) -> Mesh:
    return lathe([(0.0, 0.0), (r, 0.0), (r, h), (0.0, h)], n)


def loft(sections, cap0: bool = True, cap1: bool = True) -> Mesh:
    """Skin through closed 3D loops with equal point counts.

    Each loop must run counter-clockwise about the direction from the first
    section to the last.
    """
    S = [np.asarray(s, float) for s in sections]
    k, m = len(S[0]), len(S)
    v = np.vstack(S)
    f = []
    for a in range(m - 1):
        for i in range(k):
            j = (i + 1) % k
            p, q, r, s = a * k + i, a * k + j, (a + 1) * k + j, (a + 1) * k + i
            f += [(p, q, r), (p, r, s)]
    verts = [v]
    base = len(v)
    if cap0:
        verts.append(S[0].mean(axis=0, keepdims=True))
        f += [(base, j, i) for i, j in ((i, (i + 1) % k) for i in range(k))]
        base += 1
    if cap1:
        verts.append(S[-1].mean(axis=0, keepdims=True))
        off = (m - 1) * k
        f += [(base, off + i, off + (i + 1) % k) for i in range(k)]
    return Mesh(np.vstack(verts), np.array(f))


def section(center, frame, loop2d) -> np.ndarray:
    """3D loop from a 2D loop in the (u, v) plane of frame (columns u, v, w)."""
    loop2d = np.asarray(loop2d, float)
    return np.asarray(center, float) + loop2d[:, :1] * frame[:, 0] + loop2d[:, 1:2] * frame[:, 1]


def limb(p0, p1, radii, hint=(1.0, 0.0, 0.0), n: int = 20, squash: float = 1.0,
         round_ends: tuple[bool, bool] = (True, True), round_scale: float = 0.8) -> Mesh:
    """Tapered tube from p0 to p1, radii spread evenly along the axis.

    squash < 1 flattens the section along the hint direction. Rounded ends add
    shrinking rings so the tube closes like a capsule.
    """
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    axis = p1 - p0
    if np.linalg.norm(axis) < 1e-9:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), int))
    F = frame_from_axis(axis, hint)
    w = F[:, 2]

    def ring(c, r):
        return section(c, F, superellipse(r * squash, r, 2.0, n))

    secs = []
    if round_ends[0]:
        for th in (80.0, 60.0, 35.0):
            a = np.radians(th)
            secs.append(ring(p0 - w * radii[0] * np.sin(a) * round_scale, radii[0] * np.cos(a)))
    for t, r in zip(np.linspace(0.0, 1.0, len(radii)), radii):
        secs.append(ring(p0 + axis * t, r))
    if round_ends[1]:
        for th in (35.0, 60.0, 80.0):
            a = np.radians(th)
            secs.append(ring(p1 + w * radii[-1] * np.sin(a) * round_scale, radii[-1] * np.cos(a)))
    return loft(secs)


# ---------------------------------------------------------------- plates with holes

def _zip(inner, outer, v2, center):
    """Triangles filling the region between an inner loop and an outer loop (both CCW, star-shaped)."""
    def ordered(ids):
        ang = np.mod(np.arctan2(v2[ids, 1] - center[1], v2[ids, 0] - center[0]), 2 * np.pi)
        start = int(np.argmin(ang))
        ids = list(ids[start:]) + list(ids[:start])
        a = np.unwrap(np.mod(np.arctan2(v2[ids, 1] - center[1], v2[ids, 0] - center[0]), 2 * np.pi))
        a = np.append(a, a[0] + 2 * np.pi)
        return ids + [ids[0]], a

    I, ai = ordered(np.asarray(inner))
    O, ao = ordered(np.asarray(outer))
    i = j = 0
    ni, no = len(I) - 1, len(O) - 1
    tris = []
    while i < ni or j < no:
        if j < no and (i >= ni or ao[j + 1] <= ai[i + 1]):
            tris.append((O[j], O[j + 1], I[i]))
            j += 1
        else:
            tris.append((I[i + 1], I[i], O[j]))
            i += 1
    return tris


def plate_with_holes(w: float, h: float, t: float, holes=(), corner_r: float = 0.0,
                     n_corner: int = 6) -> tuple[Mesh, Mesh]:
    """Plate w x h x t centred at the origin (top at z = +t/2) with through holes.

    holes: (x, y, loop2d) with loop2d a counter-clockwise loop around its own
    origin. Returns (plate, hole_walls) so the hole walls can get a dark material.
    """
    holes = list(holes)
    xs, ys = {-w / 2, w / 2}, {-h / 2, h / 2}
    for hx, hy, loop in holes:
        loop = np.asarray(loop, float)
        mx = np.max(np.abs(loop[:, 0])) * 1.6 + 1e-4
        my = np.max(np.abs(loop[:, 1])) * 1.6 + 1e-4
        lo_x, hi_x, lo_y, hi_y = hx - mx, hx + mx, hy - my, hy + my
        # a hole cell next to a rounded corner runs to the edge, so the corner arc fits inside it
        if hi_x > w / 2 - corner_r - 1e-9:
            hi_x = w / 2
        if lo_x < -w / 2 + corner_r + 1e-9:
            lo_x = -w / 2
        if hi_y > h / 2 - corner_r - 1e-9:
            hi_y = h / 2
        if lo_y < -h / 2 + corner_r + 1e-9:
            lo_y = -h / 2
        xs |= {lo_x, hi_x}
        ys |= {lo_y, hi_y}
    xs, ys = sorted(xs), sorted(ys)
    v2 = [(x, y) for y in ys for x in xs]
    nx = len(xs)

    def gid(ix, iy):
        return iy * nx + ix

    tris = []
    hole_loops = []
    for iy in range(len(ys) - 1):
        for ix in range(nx - 1):
            x0, x1, y0, y1 = xs[ix], xs[ix + 1], ys[iy], ys[iy + 1]
            poly = [gid(ix, iy), gid(ix + 1, iy), gid(ix + 1, iy + 1), gid(ix, iy + 1)]
            # rounded outer corners
            if corner_r > 0:
                corners = {(0, 0): (x0 + corner_r, y0 + corner_r, np.pi, 0),
                           (nx - 2, 0): (x1 - corner_r, y0 + corner_r, 1.5 * np.pi, 1),
                           (nx - 2, len(ys) - 2): (x1 - corner_r, y1 - corner_r, 0.0, 2),
                           (0, len(ys) - 2): (x0 + corner_r, y1 - corner_r, 0.5 * np.pi, 3)}
                key = (ix, iy)
                if key in corners and (x1 - x0) >= corner_r - 1e-9 and (y1 - y0) >= corner_r - 1e-9:
                    cx, cy, a0, slot = corners[key]
                    arc = []
                    for s in range(n_corner + 1):
                        a = a0 + (np.pi / 2) * s / n_corner
                        arc.append(len(v2))
                        v2.append((cx + corner_r * np.cos(a), cy + corner_r * np.sin(a)))
                    poly = poly[:slot] + arc + poly[slot + 1:]
            inside = [hh for hh in holes if x0 < hh[0] < x1 and y0 < hh[1] < y1]
            V = np.array(v2)
            if inside:
                hx, hy, loop = inside[0]
                ids = []
                for q in np.asarray(loop, float):
                    ids.append(len(v2))
                    v2.append((hx + q[0], hy + q[1]))
                V = np.array(v2)
                tris += _zip(ids, poly, V, (hx, hy))
                hole_loops.append(ids)
            else:
                c = V[poly].mean(axis=0)
                ci = len(v2)
                v2.append(tuple(c))
                tris += [(ci, poly[k], poly[(k + 1) % len(poly)]) for k in range(len(poly))]
    V = np.array(v2)
    T = np.array(tris)
    n = len(V)
    top = Mesh(np.column_stack([V, np.full(n, t / 2)]), T)
    bottom = Mesh(np.column_stack([V, np.full(n, -t / 2)]), T[:, ::-1])
    # outer boundary loop, counter-clockwise
    outer = rounded_rect(w, h, corner_r, n_corner) if corner_r > 0 else np.array(
        [[w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2], [-w / 2, -h / 2]])
    side = wall(outer, -t / 2, t / 2, 40.0)
    plate = merge([compact(top), compact(bottom), side])
    walls = merge([wall(V[ids], -t / 2, t / 2, 60.0, inward=True) for ids in hole_loops]) if hole_loops else None
    return plate, walls
