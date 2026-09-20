"""The QP solver and the motion layer.

docs/design.md section 8.5: the solver is tested by checking the optimality conditions on
random problems. The motion layer is tested on the three things it must do: not
close on a person in the way, not slow an empty cell, and back away from a person
walking in.
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

from sightline_planner import ToolKinematics, qp  # noqa: E402
from sightline_planner.motion import D_SAFE, MotionLayer, Obstacles, segment_distance  # noqa: E402


class TestSolver(unittest.TestCase):
    def test_optimality_conditions_on_random_problems(self):
        rng = np.random.default_rng(0)
        solved = 0
        for _ in range(200):
            n = int(rng.integers(2, 8))
            m = int(rng.integers(1, 20))
            A = rng.normal(size=(n, n))
            G = A @ A.T + n * np.eye(n)
            a = rng.normal(size=n)
            C = rng.normal(size=(n, m))
            b = rng.normal(size=m) - 0.5
            sol = qp.solve(G, a, C, b)
            if not sol.feasible:
                continue
            solved += 1
            err = qp.kkt_error(G, a, C, b, sol)
            for key, value in err.items():
                self.assertLess(value, 1e-8, f"{key} {value:.2e}")
        self.assertGreater(solved, 100)

    def test_infeasible_is_really_infeasible(self):
        """A pair of half spaces that cannot both hold."""
        G = np.eye(2)
        a = np.zeros(2)
        C = np.array([[1.0, -1.0], [0.0, 0.0]])
        b = np.array([1.0, 1.0])            # x >= 1 and -x >= 1
        self.assertFalse(qp.solve(G, a, C, b).feasible)

    def test_unconstrained_minimum_when_nothing_binds(self):
        G = np.diag([2.0, 4.0])
        a = np.array([-2.0, -4.0])          # minimum at (1, 1)
        C = np.array([[1.0], [0.0]])
        b = np.array([-10.0])               # x >= -10, not binding
        sol = qp.solve(G, a, C, b)
        self.assertTrue(sol.feasible)
        np.testing.assert_allclose(sol.x, [1.0, 1.0], atol=1e-9)
        self.assertEqual(sol.active, [])


class TestMotionLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sightline_sim import station
        cls.kin = ToolKinematics(station.robot_only_model())
        cls.q0 = np.array([0.0, -1.6, 1.6, -1.6, -1.57, 0.0])
        cls.start, _ = cls.kin.fk(cls.q0)

    @staticmethod
    def cloud(centre):
        g = np.arange(-0.05, 0.051, 0.02)
        return np.array([centre + (x, y, z) for x in g for y in g for z in np.arange(-0.08, 0.081, 0.02)])

    def gap(self, layer, q, pts):
        return min(segment_distance(pt, p0, p1)[1] - r
                   for _, p0, p1, r in layer._world_capsules(q) for pt in pts[::3])

    def drive(self, layer, person, velocity, goal, seconds):
        q = self.q0.copy()
        closest = np.inf
        for _ in range(int(seconds / 0.01)):
            pos, _ = self.kin.fk(q)
            v = 2.0 * (goal - pos)
            s = float(np.linalg.norm(v))
            if s > 0.25:
                v *= 0.25 / s
            rep = layer.solve(q, np.concatenate([v, np.zeros(3)]),
                              Obstacles(person=person, person_velocity=velocity))
            q = self.kin.clamp(q + rep.qd * 0.01)
            if len(person):
                closest = min(closest, self.gap(layer, q, person))
        return q, closest

    def test_it_does_not_close_on_a_person_in_the_way(self):
        goal = self.start + np.array([0.0, -0.55, 0.0])
        # 0.9 of the way: the cloud used to sit at 0.6, which was clear only while every
        # link was modelled as a ball at its centre. With the links at their full length
        # the arm starts 34 mm inside a cloud there, which is not "in the way".
        person = self.cloud(self.start + 0.9 * (goal - self.start))
        at_start = self.gap(MotionLayer(self.kin), self.q0, person)
        _, closest_off = self.drive(MotionLayer(self.kin, use_r1=False), person, np.zeros_like(person), goal, 5.0)
        _, closest_on = self.drive(MotionLayer(self.kin, use_r1=True), person, np.zeros_like(person), goal, 5.0)
        self.assertLess(closest_off, 0.0, "without the rule the arm should go through the cloud")
        self.assertGreaterEqual(closest_on, at_start - 0.002, "with the rule it got closer than where it started")

    def test_an_empty_cell_is_not_slowed(self):
        goal = self.start + np.array([0.0, -0.55, 0.0])
        empty = np.zeros((0, 3))
        q, _ = self.drive(MotionLayer(self.kin, use_r1=True), empty, empty, goal, 5.0)
        pos, _ = self.kin.fk(q)
        self.assertLess(float(np.linalg.norm(goal - pos)), 0.01)

    def test_it_backs_away_from_a_person_walking_in(self):
        layer = MotionLayer(self.kin, use_r1=True)
        q = self.q0.copy()
        side = self.start + np.array([0.50, 0.0, -0.05])
        closest = np.inf
        for step in range(180):
            centre = side + np.array([-0.5, 0.0, 0.0]) * step * 0.01
            person = self.cloud(centre)
            vel = np.tile([-0.5, 0.0, 0.0], (len(person), 1))
            rep = layer.solve(q, np.zeros(6), Obstacles(person=person, person_velocity=vel))
            q = self.kin.clamp(q + rep.qd * 0.01)
            closest = min(closest, self.gap(layer, q, person))
        moved = float(np.linalg.norm(self.kin.fk(q)[0] - self.start))
        self.assertGreater(moved, 0.10, "the arm did not move away")
        self.assertGreater(closest, 0.0, "the person reached the arm")

    def test_the_worktop_plane_is_kept(self):
        """The plane sits below the robot's own base, as the worktop does, and the task
        pushes the tool straight down through it for four seconds."""
        layer = MotionLayer(self.kin, use_r1=True)
        q = self.q0.copy()
        floor = 1.05
        planes = [(np.array([0.0, 0.0, floor]), np.array([0.0, 0.0, 1.0]))]
        lowest = np.inf
        for _ in range(400):
            rep = layer.solve(q, np.array([0.0, 0.0, -0.25, 0.0, 0.0, 0.0]), Obstacles(planes=planes))
            q = self.kin.clamp(q + rep.qd * 0.01)
            for _, p0, p1, r in layer._world_capsules(q):
                lowest = min(lowest, float(min(p0[2], p1[2])) - r)
        tip, _ = self.kin.fk(q)
        self.assertLess(tip[2], self.start[2] - 0.20, "the tool did not go down at all")
        self.assertGreater(lowest, floor - 0.005, f"a link went {(floor - lowest) * 1000:.0f} mm under the plane")


if __name__ == "__main__":
    unittest.main()
