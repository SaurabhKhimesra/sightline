"""The camera's view of the cell as a grid of directions, and where the robot may not be.

The user's rule, as stated: imagine a grid as the camera sees it. Every direction in
which the camera sees the person is off limits to the robot, from the camera out to
just behind him. In front of him the arm would block the view (R2); at him it would
touch him (R1). Around that goes a margin: the safety distance, the size of the part
of the arm being checked, and how far he could have moved since the picture was
taken. The grid is rebuilt from every picture, every 40 ms, and the arm checks
against it every 10 ms.

Each cell of the grid that holds him keeps the distance from the camera out to which
he may be, and when that was seen. A point of the arm is off limits if, for some such
cell, it is in front of him and its line of sight passes his closer than the margin
(R2), or it is closer than the margin to where he may be along that line of sight
(R1). Distances to a line of sight are measured at the point's own depth: the
distance from a point to a line through the camera is its depth times the angle
between them. A first version grew the grid by the margin once per picture, at his
nearest depth and as a square, and turned a 12 cm margin into 19 to 26 cm; every
cover hole was off limits with his hand 19 cm away.

Two things the camera cannot give directly are added:
- where the arm itself hides the view, he may be under it. A cell of him that the arm
  covers in the next picture is kept, at the depth it was seen, growing at that part's
  own speed (at most the ISO 2000 mm/s), for up to 0.5 s or until the camera sees
  into it again. It counts for touching (R1) only: the arm is in front of it by
  definition, and moving the arm away can only give the view back. Two first versions
  failed. Kept with no time limit, a hidden cell's margin grew for as long as the
  parked arm hid it, to 53 m by the end of the cycle. Spread through the arm's shadow
  at his speed, his hands at the jig, next to the robot's base in the picture, filled
  the whole parked arm's shadow in half a second, and the arm read as touching him
  everywhere;
- how fast each part of him moves. It is measured from the grid itself: how far each
  cell he is in now is from the nearest cell he was in three pictures ago, 120 ms
  back, where each old cell counts both its nearest and its farthest pixel of him. A
  reading needs a neighbouring cell that agrees. Only the leading edge of a moving
  hand shows the jump, so each cell takes the fastest reading within a hand's size
  around it, and keeps the fastest of the last 0.2 s. SPEC.md section 17.2 decided the
  measured speed leads; the constant 2000 mm/s of ISO 13855 is the other setting.

  Measured against his true hands over one cycle with the arm parked (SPEC 17.2 asked
  for this before the measured speed may lead): still hands read 0.26 m/s median and
  0.48 at the 90th percentile, moving hands 1.27 times their true speed, and the
  reading is below the hand's true peak until the next picture in 11 % of pictures,
  by up to 0.41 m/s. That shortfall is the lag of any speed read from past pictures
  while a hand speeds up; at the oldest picture it uses 6.8 cm of the 10 cm safety
  distance. The acceleration term that would cover it needs a published bound.

  Two first versions read still hands at 0.92 m/s median and 2.14 at p90: a cell
  straddling two surfaces of him, say the edge of his head over his hand, flips its
  nearest pixel from one to the other, and read as a jump of up to 1.2 m.

  A first version used one speed for all of him, his fastest part's. Things coming
  into view, a part lifted off the rack or a hand coming out from behind it, read as
  jumps of 2.4 to 2.7 m/s while his hands moved at 0.03 to 0.7 m/s, and that one
  number grew the margin round every part of him: the parked arm, far from him, was
  off limits for whole seconds.
"""

from __future__ import annotations

from collections import deque

import numpy as np

CELL_PX = 4                  # a grid cell is 4 by 4 pixels, about 2.3 cm at 3.2 m (scene choice)
MIN_PIXELS = 2               # a cell needs two pixels of him: one stray pixel is noise (scene choice)
# A cell with no occupied neighbour is noise too. On the eight work poses, three noisy
# pictures each, this took the cells nowhere near him from 18.9 to 1.75 a picture and
# lost none of his (40.46 against 40.42 cells missed of 474, the ones the 40 mm
# background gate already hides). Three pixels a cell instead lost 15 more of his.
SOLID_CELLS = 6              # speed is read only where a 5 by 5 cell window holds this much of him (scene choice)
BEHIND_M = 0.15              # behind a seen surface a hand can still hide: a limb's thickness (scene choice)
SAFE_M = 0.10                # R1's safety distance, the same as the motion layer (scene choice)
SPEED_BASELINE = 3           # speed is measured against the picture this many frames back (scene choice)
SPEED_RANK = 3               # for logs: his third fastest cell
SPEED_HOLD_S = 0.2           # each cell keeps the fastest speed of the last 0.2 s (scene choice)
PART_M = 0.15                # a moving part of him, a hand and wrist, shares its leading edge's speed (scene choice)
# ISO 13855's person speed, 2000 mm/s, as reproduced by Marvel and Norcross 2017,
# Sec. 3, pp. 146 and 148. The constant setting of SPEC.md section 8.6 (a).
ISO_SPEED = 2.0
SPEED_CAP = ISO_SPEED        # a reading faster than the standard's own worst case is noise, not him
MEMORY_S = 0.5               # a hidden cell is kept this long at most: the guard has moved the arm by then (scene choice)


class ViewGrid:
    """Off-limits directions and depths, rebuilt from each picture of the person."""

    def __init__(self, calib, cell_px: int = CELL_PX, safe: float = SAFE_M,
                 behind: float = BEHIND_M, speed_mode: str = "measured", accel: float = 0.0,
                 timed_above: float = -np.inf):
        self.calib = calib
        self.cell = cell_px
        self.gw = calib.width // cell_px
        self.gh = calib.height // cell_px
        self.f = calib.focal()
        self.safe = safe
        self.behind = behind
        self.speed_mode = speed_mode
        # Setting (b) of SPEC.md section 8.6 adds a bounded acceleration term. Its bound
        # has to come from published data on human arm movement, and none has been
        # opened yet, so it is zero until one is (SPEC.md section 17.2).
        self.accel = accel
        # His speed is read from the parts of him above this height. His feet stand in
        # the floor's depth noise, flicker in and out of the picture and read as 0.4 to
        # 0.6 m/s while he stands still, and they cannot reach an arm that stays above
        # the worktop.
        self.timed_above = timed_above
        # the cells that hold him, or may: row, column, where along the axis he may be
        # (near to far), when, how fast that part of him moves, and whether it is only
        # remembered under the arm rather than seen
        self.rows = np.zeros(0, dtype=np.int64)
        self.cols = np.zeros(0, dtype=np.int64)
        self.near = np.zeros(0)
        self.far = np.zeros(0)
        self.seen_at = np.zeros(0)
        self.cell_speed = np.zeros(0)                            # m/s, per cell of him
        self.hidden = np.zeros(0, dtype=bool)
        self.background = None                                   # the empty cell's depth per grid cell
        self.seen = np.zeros((self.gh, self.gw), dtype=bool)     # this picture's cells of him
        self.taken_at: float | None = None
        self.speed = ISO_SPEED if speed_mode == "iso" else 0.0   # his fastest part, for logs
        self.measured = 0.0                                      # the same before holding, for logs
        self.remembered = 0                                      # cells kept under the arm's shadow
        self._history: deque = deque(maxlen=SPEED_BASELINE + 1)
        self._fields: deque = deque()                            # (time, speed per grid cell)
        cols = (np.arange(self.gw) + 0.5) * cell_px
        rows = (np.arange(self.gh) + 0.5) * cell_px
        self._rx = (cols - calib.width / 2) / self.f              # ray slope per column
        self._ry = -(rows - calib.height / 2) / self.f            # and per row
        self._half_diag = cell_px / np.sqrt(2.0)                  # centre to corner of a cell, pixels

    # ------------------------------------------------------------------ building
    def reach(self, age, speed=None):
        """How far he could have moved in this long, at this speed (his fastest part's if none)."""
        age = np.maximum(age, 0.0)
        speed = self.speed if speed is None else speed
        return speed * age + 0.5 * self.accel * age * age

    def set_background(self, depth: np.ndarray) -> None:
        """The empty cell's depth image: nothing of him can be behind it. No return reads
        as unlimited."""
        d = np.where(depth > 0, depth, np.inf).reshape(self.gh, self.cell, self.gw, self.cell)
        self.background = d.max(axis=(1, 3))

    def update(self, person_px: np.ndarray, person_z: np.ndarray, taken_at: float,
               shadow: np.ndarray | None = None) -> None:
        """person_px are pixel indices of him in the picture, person_z their depths.
        shadow is, per cell, how far out the arm's own surface was when the picture was
        taken, inf where the arm was not in the way."""
        near_g = np.full(self.gh * self.gw, np.inf)
        if person_px.size:
            u = person_px % self.calib.width
            v = person_px // self.calib.width
            key = (v // self.cell) * self.gw + (u // self.cell)
            np.minimum.at(near_g, key, person_z)
            count = np.bincount(key, minlength=self.gh * self.gw)
            near_g[count < MIN_PIXELS] = np.inf
        near_g = near_g.reshape(self.gh, self.gw)
        far_px = np.full(self.gh * self.gw, -np.inf)          # his farthest pixel per cell, for the speed
        if person_px.size:
            np.maximum.at(far_px, key, person_z)
        self._far_px = far_px.reshape(self.gh, self.gw)
        seen = np.isfinite(near_g)
        seen &= _window(seen, 1) > 1          # itself and at least one neighbour
        near_g[~seen] = np.inf
        field = self._measure_speed(seen, near_g, taken_at)
        rows, cols = np.nonzero(seen)
        near = near_g[rows, cols]
        far = near + self.behind
        speed = field[rows, cols]
        hidden = np.zeros(rows.size, dtype=bool)
        seen_at = np.full(rows.size, taken_at)
        self.remembered = 0
        if shadow is not None and self.rows.size:
            # what the arm covers now keeps what was there, with the time it was seen
            keep = np.isfinite(shadow[self.rows, self.cols]) & ~seen[self.rows, self.cols] & \
                (taken_at - self.seen_at <= MEMORY_S)
            if keep.any():
                rows, cols = np.concatenate([rows, self.rows[keep]]), np.concatenate([cols, self.cols[keep]])
                near, far = np.concatenate([near, self.near[keep]]), np.concatenate([far, self.far[keep]])
                speed = np.concatenate([speed, np.minimum(self.cell_speed[keep], ISO_SPEED)])
                seen_at = np.concatenate([seen_at, self.seen_at[keep]])
                hidden = np.concatenate([hidden, np.ones(int(keep.sum()), dtype=bool)])
                self.remembered = int(keep.sum())
        self.rows, self.cols, self.near, self.far = rows, cols, near, far
        self.seen_at = seen_at
        self.cell_speed, self.hidden = speed, hidden
        self.seen = seen
        self.taken_at = taken_at

    def _points(self, rows: np.ndarray, cols: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Cell centres at these depths, in the camera's frame: distances are what matter."""
        return np.column_stack([self._rx[cols] * z, self._ry[rows] * z, -z]).astype(np.float32)

    def _measure_speed(self, seen: np.ndarray, near: np.ndarray, t: float) -> np.ndarray:
        """Speed per grid cell: how far each piece of him is from where he was three
        pictures back, spread over a hand's size and held for 0.2 s."""
        # only where a real piece of him is: two noise cells side by side, which the
        # neighbour test lets through, would otherwise read as a jump of a metre
        v, u = np.nonzero(seen & (_window(seen, 2) >= SOLID_CELLS))
        pts = self._points(v, u, near[v, u])
        back = self._points(v, u, self._far_px[v, u])
        if np.isfinite(self.timed_above) and pts.shape[0]:
            height = (pts.astype(float) @ self.calib.R.T + self.calib.pos)[:, 2]
            keep = height > self.timed_above
            v, u, pts, back = v[keep], u[keep], pts[keep], back[keep]
        # this picture's nearest pixels, compared later with both layers of this one
        self._history.append((t, np.vstack([pts, back])))
        local = np.zeros((self.gh, self.gw))
        raw = 0.0
        if len(self._history) > SPEED_BASELINE and pts.shape[0]:
            t_old, old = self._history[0]
            if old.shape[0] and t > t_old:
                gap = np.full(pts.shape[0], np.inf, dtype=np.float32)
                for i in range(0, old.shape[0], 256):
                    block = old[i:i + 256]
                    d2 = ((pts[:, None, :] - block[None, :, :]) ** 2).sum(axis=2).min(axis=1)
                    np.minimum(gap, d2, out=gap)
                speed = np.minimum(np.sqrt(gap) / (t - t_old), SPEED_CAP)
                local[v, u] = speed
                k = min(SPEED_RANK, speed.size)
                raw = float(np.partition(speed, speed.size - k)[speed.size - k])
        # a reading needs a neighbour that agrees: the second largest of its 3 by 3
        padded = np.pad(local, 1)
        stack = np.stack([padded[1 + dy:1 + dy + self.gh, 1 + dx:1 + dx + self.gw]
                          for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
        local = np.minimum(local, np.sort(stack, axis=0)[-2])
        # the leading edge's speed for the whole part, a hand's size round it
        z = float(np.median(near[v, u])) if v.size else 3.0
        field = _max_filter(local, int(np.ceil(PART_M * self.f / (z * self.cell))))
        self._fields.append((t, field))
        while self._fields and self._fields[0][0] < t - SPEED_HOLD_S:
            self._fields.popleft()
        held = np.maximum.reduce([f for _, f in self._fields])
        self.measured = raw
        if self.speed_mode == "iso":
            return np.full((self.gh, self.gw), ISO_SPEED)
        self.speed = float(held.max())
        return held

    def seen_points(self) -> np.ndarray:
        """Where the cells he is seen in are, in the world, at their near depth."""
        seen = ~self.hidden
        if not seen.any():
            return np.zeros((0, 3))
        p = self._points(self.rows[seen], self.cols[seen], self.near[seen]).astype(float)
        return p @ self.calib.R.T + self.calib.pos

    def cell_point(self, k: int) -> np.ndarray:
        """Where cell k of him is, in the world, at its near depth. For logs."""
        p = self._points(self.rows[k:k + 1], self.cols[k:k + 1], self.near[k:k + 1]).astype(float)
        return (p @ self.calib.R.T + self.calib.pos)[0]

    def speed_near(self, point: np.ndarray, within: float = 0.10) -> float:
        """The speed the grid gives the part of him nearest this point, for logs."""
        if self.rows.size == 0:
            return 0.0
        pts = self._points(self.rows, self.cols, self.near).astype(float) @ self.calib.R.T + self.calib.pos
        gap = np.linalg.norm(pts - np.asarray(point, float), axis=1)
        close = (gap < within) & ~self.hidden
        return float(self.cell_speed[close].max()) if close.any() else float("nan")

    # ------------------------------------------------------------------ asking
    def project(self, points: np.ndarray):
        """Pixel column, pixel row and depth along the axis of each point; in front of the camera."""
        rel = (np.asarray(points, float) - self.calib.pos) @ self.calib.R
        depth = -rel[:, 2]
        ahead = depth > 1e-3
        safe_depth = np.where(ahead, depth, 1.0)
        u = self.calib.width / 2 + self.f * rel[:, 0] / safe_depth
        v = self.calib.height / 2 - self.f * rel[:, 1] / safe_depth
        return u, v, depth, ahead

    def excess(self, points: np.ndarray, radii: np.ndarray, now, want_cells: bool = False):
        """How far inside the off-limits space each point is, in metres; negative is
        clear by that much, and -inf is nowhere near him. radii is how far the robot's
        surface reaches beyond each point; now is one time or one per point.

        Two parts, added when both bite, so that every way toward him reads as worse:
        - R2: in front of a cell where he is seen, and the line of sight to it closer than
          the margin: the margin less that sideways distance;
        - R1: closer than the margin to where he may be along that line of sight: the
          margin less that distance.
        Sideways alone, moving down the line of sight onto his hand would not change
        the number at all.
        """
        pts = np.atleast_2d(np.asarray(points, float))
        out = np.full(len(pts), -np.inf)
        nothing = (out, np.full(len(pts), -1, dtype=np.int64)) if want_cells else out
        if self.rows.size == 0:
            return nothing
        now = np.broadcast_to(np.asarray(now, float), (len(pts),))
        u, v, depth, ahead = self.project(pts)
        idx = np.nonzero(ahead)[0]
        if idx.size == 0:
            return nothing
        radii = np.asarray(radii, float)
        radii = np.broadcast_to(radii, (len(pts),))[idx]
        when = now[idx]
        cu = (self.cols + 0.5) * self.cell
        cv = (self.rows + 0.5) * self.cell
        # only cells that could be within the largest margin of some point matter
        widest = self.reach(when.max() - self.seen_at, self.cell_speed).max()
        reach_px = (self.safe + radii.max() + widest) * self.f / depth[idx].min() + self._half_diag
        box = (cu > u[idx].min() - reach_px) & (cu < u[idx].max() + reach_px) & \
              (cv > v[idx].min() - reach_px) & (cv < v[idx].max() + reach_px)
        if not box.any():
            return nothing
        cu, cv = cu[box], cv[box]
        near, far, hidden = self.near[box], self.far[box], self.hidden[box]
        grow = self.reach(when[:, None] - self.seen_at[box][None, :], self.cell_speed[box][None, :])
        margin = self.safe + radii[:, None] + grow
        z = depth[idx][:, None]
        # the point's line of sight to the nearest corner of the cell, in metres at its depth
        px = np.hypot(u[idx][:, None] - cu[None, :], v[idx][:, None] - cv[None, :]) - self._half_diag
        side = np.maximum(px, 0.0) * z / self.f
        along = np.maximum(near[None, :] - z, 0.0) + np.maximum(z - far[None, :], 0.0)
        r1 = margin - np.sqrt(side * side + along * along)
        r2 = np.where((z < near[None, :]) & ~hidden[None, :], margin - side, -np.inf)
        both = np.where((r1 > 0) | (r2 > 0), np.maximum(r1, 0.0) + np.maximum(r2, 0.0), np.maximum(r1, r2))
        out[idx] = both.max(axis=1)
        if want_cells:
            cells = np.full(len(pts), -1, dtype=np.int64)
            cells[idx] = np.nonzero(box)[0][both.argmax(axis=1)]
            return out, cells
        return out

    def forbidden(self, points: np.ndarray, radii: np.ndarray, now: float) -> np.ndarray:
        """True for every point the robot may not occupy at time now."""
        return self.excess(points, radii, now) > 0

    def shadow_of(self, points: np.ndarray, radii: np.ndarray) -> np.ndarray:
        """Per cell, how far out from the camera the arm's own surface is: whatever is
        behind it is hidden. inf where the arm is not in the way."""
        out = np.full((self.gh, self.gw), np.inf)
        u, v, depth, ahead = self.project(points)
        for i in np.nonzero(ahead)[0]:
            r = float(radii[i]) * self.f / depth[i]                # pixels
            c0, c1 = int(np.floor((u[i] - r) / self.cell)), int(np.floor((u[i] + r) / self.cell))
            r0, r1 = int(np.floor((v[i] - r) / self.cell)), int(np.floor((v[i] + r) / self.cell))
            if c1 < 0 or r1 < 0 or c0 >= self.gw or r0 >= self.gh:
                continue
            block = out[max(r0, 0):r1 + 1, max(c0, 0):c1 + 1]
            np.minimum(block, depth[i] - float(radii[i]), out=block)
        return out


def _max_filter(a: np.ndarray, r: int) -> np.ndarray:
    """Square max filter of half width r, done as two passes of shifted maxima."""
    if r <= 0:
        return a
    out = a.copy()
    for axis in (0, 1):
        src = out.copy()
        n = src.shape[axis]
        for k in range(1, min(r, n - 1) + 1):
            if axis == 0:
                np.maximum(out[k:, :], src[:-k, :], out=out[k:, :])
                np.maximum(out[:-k, :], src[k:, :], out=out[:-k, :])
            else:
                np.maximum(out[:, k:], src[:, :-k], out=out[:, k:])
                np.maximum(out[:, :-k], src[:, k:], out=out[:, :-k])
    return out


def _window(mask: np.ndarray, r: int) -> np.ndarray:
    """How many cells are set in the square of half width r around each cell."""
    c = np.pad(mask.astype(np.int32), ((r + 1, r), (r + 1, r))).cumsum(0).cumsum(1)
    k = 2 * r + 1
    return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
