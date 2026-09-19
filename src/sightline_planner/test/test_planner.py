"""The planner's boundary and its kinematics.

SPEC.md section 8.1: the planner package imports nothing from the simulation or the
judge, and a test fails if such an import appears.
"""

from __future__ import annotations

import ast
import os
import unittest
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from sightline_planner import B0, SensorFrame, Taught, ToolKinematics  # noqa: E402
from sightline_planner.b0 import BIT_DOWN, rot_z  # noqa: E402

PLANNER = Path(__file__).resolve().parents[1] / "src" / "sightline" / "planner"
FORBIDDEN = ("sightline_sim", "sightline_ros", "..sim", "..judge")


class TestHonestyBoundary(unittest.TestCase):
    def test_the_planner_imports_neither_the_simulation_nor_the_judge(self):
        for path in sorted(PLANNER.glob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    dots = "." * (node.level or 0)
                    names = [dots + (node.module or "")]
                else:
                    continue
                for name in names:
                    for bad in FORBIDDEN:
                        self.assertNotIn(bad, name, f"{path.name} imports {name}")

    def test_the_planner_never_sees_a_person(self):
        """Its robot model holds the robot and nothing else."""
        from sightline_sim import station
        rm = station.robot_only_model()
        names = [rm.body(i).name for i in range(rm.nbody)]
        self.assertFalse([n for n in names if n.startswith(station.HUMAN_PREFIX)], names)
        self.assertIn("ur5e/wrist_3_link", names)


class TestKinematics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sightline_sim import station
        cls.kin = ToolKinematics(station.robot_only_model())

    def test_the_wrist_camera_pose_follows_the_joints(self):
        """mj_kinematics does not move cameras. Without mj_camlight the pose stays
        at the origin, every hole projects behind the lens and the servo never runs."""
        a, Ra = self.kin.camera_pose(np.array([0.0, -1.6, 1.6, -1.6, -1.57, 0.0]))
        b, _ = self.kin.camera_pose(np.array([0.5, -1.6, 1.6, -1.6, -1.57, 0.0]))
        self.assertGreater(float(np.linalg.norm(a)), 0.3, "camera sits at the origin")
        self.assertGreater(float(np.linalg.norm(a - b)), 0.05, "camera does not follow the arm")
        self.assertAlmostEqual(float(np.linalg.det(Ra)), 1.0, places=5)

    def test_the_planner_model_agrees_with_the_cell(self):
        from sightline_sim import station, pose
        scene = station.build()
        m, d = scene.model, mujoco.MjData(scene.model)
        qadr = pose.robot_qadr(m)
        rng = np.random.default_rng(0)
        for _ in range(3):
            q = rng.uniform(-2.0, 2.0, 6)
            d.qpos[qadr] = q
            mujoco.mj_forward(m, d)
            tip, _ = self.kin.fk(q)
            self.assertLess(float(np.linalg.norm(tip - d.site_xpos[m.site("tool_tip").id])), 1e-6)

    def test_going_into_a_hole_is_straight(self):
        """Interpolating joints between two poses either side of a configuration
        change bowed the bit 37 mm sideways on a 60 mm descent."""
        from sightline_sim import station
        scene = station.build()
        base = np.array(scene.points["box_home"], float)
        taught = Taught(part_pose=tuple(base),
                        cover_holes=[tuple(np.array(scene.points["cover_holes"][0], float) - base)],
                        feeder_pick=tuple(np.array(scene.points["feeder_pick"], float)))
        b0 = B0.build(self.kin, taught, park_tip=(-0.45, 0.33, scene.worktop_z + 0.30))
        screw = b0.screws[0]
        above, R = b0._pose_for(screw, taught.screw_length + taught.approach)
        down, _ = b0._pose_for(screw, taught.screw_length)
        q, err = self.kin.ik_multi(above, R, np.array([0.0, -1.6, 1.6, -1.6, -1.57, 0.0]))
        self.assertLess(err, 1e-4)
        qd, left = b0._straight(q, above, down, R, "test", speed=np.deg2rad(25.0))
        plan = b0._goals["test"]
        self.assertGreaterEqual(len(plan), 2)
        worst = 0.0
        prev = q
        for way in plan:
            for s in np.linspace(0, 1, 9):
                tip, _ = self.kin.fk(prev + s * (way - prev))
                worst = max(worst, float(np.linalg.norm(tip[:2] - down[:2])))
            prev = way
        self.assertLess(worst, 0.004, f"the bit wanders {worst * 1000:.1f} mm going into the hole")


if __name__ == "__main__":
    unittest.main()
