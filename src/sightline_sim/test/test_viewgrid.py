"""The camera's grid and the guard that checks the arm against it every 10 ms.

A synthetic camera looks straight down from 3 m. The person is a block of pixels at
a known depth, so every answer can be worked out by hand.
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

from sightline_planner import ToolKinematics  # noqa: E402
from sightline_planner.guard import TrajectoryGuard  # noqa: E402
from sightline_planner.perceive import Calibration  # noqa: E402
from sightline_planner.viewgrid import ViewGrid  # noqa: E402

W, H = 640, 480


def camera(pos=(0.0, 0.0, 3.0)) -> Calibration:
    """Looking straight down: the camera's -z is the world's -z."""
    return Calibration(pos=np.array(pos, float), R=np.eye(3), width=W, height=H, fovy_deg=45.0)


def block(u0: int, v0: int, size: int, depth: float):
    """Pixel indices and depths of a square patch of him."""
    uu, vv = np.meshgrid(np.arange(u0, u0 + size), np.arange(v0, v0 + size))
    px = (vv * W + uu).reshape(-1)
    return px, np.full(px.size, depth)


class TestViewGrid(unittest.TestCase):
    def setUp(self):
        self.calib = camera()
        self.grid = ViewGrid(self.calib)
        # a hand 2 m below the camera, 40 pixels square round the middle of the picture
        self.px, self.z = block(300, 220, 40, 2.0)
        self.grid.update(self.px, self.z, 0.0)
        self.r = np.array([0.01])

    def forbidden(self, point, now=0.0, grid=None):
        return bool((grid or self.grid).forbidden(np.array([point]), self.r, now)[0])

    def test_between_the_camera_and_him_is_off_limits(self):
        self.assertTrue(self.forbidden((0.0, 0.0, 2.0)), "the arm would block the view of him")
        self.assertTrue(self.forbidden((0.0, 0.0, 1.05)), "the arm would touch him")

    def test_beside_him_is_not(self):
        self.assertFalse(self.forbidden((0.5, 0.0, 1.5)))

    def test_behind_him_only_as_far_as_a_limb_and_the_margin(self):
        # he is 2.0 m out, a limb can hide 0.15 behind, the margin is 0.10 + 0.01
        self.assertTrue(self.forbidden((0.0, 0.0, 3.0 - 2.20)))
        self.assertFalse(self.forbidden((0.0, 0.0, 3.0 - 2.30)))

    def test_the_margin_grows_with_the_age_of_the_picture(self):
        grid = ViewGrid(self.calib, speed_mode="iso")          # 2 m/s, the constant setting
        grid.update(self.px, self.z, 0.0)
        # the patch reaches about 0.07 m out from the middle at 2 m; 0.25 m is beside it
        point = (0.25, 0.0, 1.0)
        self.assertFalse(self.forbidden(point, now=0.0, grid=grid))
        self.assertTrue(self.forbidden(point, now=0.1, grid=grid), "in 0.1 s at 2 m/s he could be there")

    def test_what_the_arm_hides_is_remembered_until_it_is_seen_again(self):
        shadow = np.full((self.grid.gh, self.grid.gw), np.inf)
        shadow[50:70, 70:90] = 1.0                              # the arm, 1 m out, now covers where he was
        empty = np.zeros(0, dtype=np.int64)
        self.grid.update(empty, np.zeros(0), 0.04, shadow)
        self.assertGreater(self.grid.remembered, 0)
        self.assertTrue(self.forbidden((0.0, 0.0, 1.05), now=0.04), "he may still be there, under the arm")
        self.assertFalse(self.forbidden((0.0, 0.0, 1.5), now=0.04),
                         "hidden cells count for touching only: the arm is in front of them anyway")
        self.grid.update(empty, np.zeros(0), 0.08, np.full_like(shadow, np.inf))
        self.assertFalse(self.forbidden((0.0, 0.0, 1.05), now=0.08), "the camera sees he has gone")

    def test_what_the_arm_hides_is_forgotten_after_half_a_second(self):
        """Held over where he was, a hidden cell counts for 0.5 s, long enough for the
        guard to move the arm; it does not grow into a wall the arm can never leave."""
        grid = ViewGrid(self.calib)
        grid.update(self.px, self.z, 0.0)
        shadow = np.full((grid.gh, grid.gw), np.inf)
        shadow[50:70, 70:90] = 1.0
        empty = np.zeros(0, dtype=np.int64)
        for k in range(1, 26):
            grid.update(empty, np.zeros(0), 0.04 * k, shadow)
            if k == 10:
                self.assertTrue(self.forbidden((0.0, 0.0, 1.05), now=0.4, grid=grid))
        self.assertEqual(grid.remembered, 0)
        self.assertFalse(self.forbidden((0.0, 0.0, 1.05), now=1.0, grid=grid))

    def test_a_moving_hand_reads_its_speed(self):
        grid = ViewGrid(self.calib)
        f = self.calib.focal()
        step_px = 0.04 * f / 2.0                                # 4 cm a picture at 2 m: 1 m/s
        for k in range(8):
            px, z = block(200 + int(round(k * step_px)), 220, 40, 2.0)
            grid.update(px, z, 0.04 * k)
        self.assertGreater(grid.measured, 0.8)
        self.assertLess(grid.measured, 1.25)

    def test_a_still_hand_reads_still(self):
        grid = ViewGrid(self.calib)
        for k in range(6):
            grid.update(self.px, self.z, 0.04 * k)
        self.assertLess(grid.measured, 0.05)


class TestGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sightline_sim import station
        cls.kin = ToolKinematics(station.robot_only_model())
        cls.q0 = np.array([0.0, -1.6, 1.6, -1.6, -1.57, 0.0])
        cls.tip, _ = cls.kin.fk(cls.q0)

    def guard_with_hand(self, offset_px: int) -> TrajectoryGuard:
        """A camera 2 m above the tool, and a hand 0.3 m under the tool's height."""
        calib = camera(pos=self.tip + [0.0, 0.0, 2.0])
        grid = ViewGrid(calib)
        px, z = block(300 + offset_px, 220, 40, 2.3)
        grid.update(px, z, 0.0)
        return TrajectoryGuard(self.kin, grid)

    def test_the_arm_over_his_hand_backs_out(self):
        guard = self.guard_with_hand(0)
        self.assertFalse(guard.pose_clear(self.q0, 0.0))
        _, verdict = guard.command(self.q0, np.zeros(6), np.zeros(6), 0.0,
                                   targets=[("park", self.q0 + [0.8, 0.0, 0.0, 0.0, 0.0, 0.0])])
        self.assertTrue(verdict.startswith("escape"), verdict)

    def test_a_clear_arm_keeps_its_command(self):
        guard = self.guard_with_hand(280)
        want = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
        qd, verdict = guard.command(self.q0, want, want, 0.0)
        self.assertEqual(verdict, "go")
        np.testing.assert_allclose(qd, want)

    def test_a_move_toward_him_is_stopped_in_time(self):
        """His hand is where the tool is heading, 0.3 m under it; the base turns toward
        it at 60 deg/s. The guard must stop the arm before any part of it is off limits."""
        goal = self.q0 + [0.8, 0.0, 0.0, 0.0, 0.0, 0.0]
        there, _ = self.kin.fk(goal)
        calib = camera(pos=self.tip + [0.0, 0.0, 2.0])
        f = calib.focal()
        u = int(round(W / 2 + f * (there[0] - self.tip[0]) / 2.0))
        v = int(round(H / 2 - f * (there[1] - self.tip[1]) / 2.0))
        grid = ViewGrid(calib)
        px, z = block(u - 20, v - 20, 40, 2.3)
        grid.update(px, z, 0.0)
        guard = TrajectoryGuard(self.kin, grid)
        self.assertTrue(guard.pose_clear(self.q0, 0.0), "the test needs a clear start")
        self.assertFalse(guard.pose_clear(goal, 0.0), "the test needs a goal over his hand")
        q = self.q0.copy()
        qd = np.zeros(6)
        want = np.array([np.deg2rad(60.0), 0.0, 0.0, 0.0, 0.0, 0.0])
        for k in range(150):
            qd, _ = guard.command(q, qd, want, 0.0)
            q = q + qd * 0.01
            self.assertTrue(guard.pose_clear(q, 0.0), f"the arm reached the margin at step {k}")
        self.assertGreater(q[0] - self.q0[0], 0.05, "it did not move toward the goal at all")
        self.assertLess(float(np.max(np.abs(qd))), 1e-6, "it did not come to a stop")

    def test_inside_the_margin_it_may_not_close_but_may_leave(self):
        """He has put his hand 0.15 m under the tool: already inside the margin. Going
        down onto it is refused; going up and away is allowed."""
        calib = camera(pos=self.tip + [0.0, 0.0, 2.0])
        grid = ViewGrid(calib)
        px, z = block(300, 220, 40, 2.15)
        grid.update(px, z, 0.0)
        guard = TrajectoryGuard(self.kin, grid)
        self.assertFalse(guard.pose_clear(self.q0, 0.0))
        J = self.kin.jacobian(self.q0)[:3]
        down = np.linalg.pinv(J) @ np.array([0.0, 0.0, -0.10])
        up = np.linalg.pinv(J) @ np.array([0.0, 0.0, 0.10])
        hold = {}
        here = guard.points(self.q0)

        def still(at):
            return hold.setdefault(at, guard._excess(here, at))
        self.assertFalse(guard._no_worse(*guard.stop_after(self.q0, down, 0.0), still), "it moved onto his hand")
        self.assertTrue(guard._no_worse(*guard.stop_after(self.q0, up, 0.0), still), "it may not even back off")

    def test_a_point_that_hardly_moves_does_not_stop_a_plan(self):
        """His hand is next to the root of the upper arm. A plan that swings the wrist
        round, far from him, is fine: the root hardly moves and gets no closer."""
        root = self.kin.m.body("ur5e/upper_arm_link").id
        self.kin.d.qpos[:6] = self.q0
        import mujoco
        mujoco.mj_kinematics(self.kin.m, self.kin.d)
        at = self.kin.d.xpos[root].copy()
        calib = camera(pos=at + [0.0, 0.0, 2.0])
        grid = ViewGrid(calib)
        px, z = block(300, 220, 40, 2.12)
        grid.update(px, z, 0.0)
        guard = TrajectoryGuard(self.kin, grid)
        goal = self.q0 + [0.0, 0.0, 0.0, 0.4, 0.4, 0.0]           # wrist joints only
        self.assertFalse(guard.pose_clear(goal, 0.0), "the test needs the root inside the margin")
        self.assertTrue(guard.pose_ok(goal, 0.0, self.q0))


if __name__ == "__main__":
    unittest.main()
