"""The ROS 2 and Gazebo side: the exported cell reloads as the cell, the joint angles
recovered from body orientations are the joint angles, and the replay takes each
camera frame by its sim time stamp.
"""

from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from sightline_sim import station  # noqa: E402



class Export(unittest.TestCase):
    def test_export_reloads_as_the_same_cell(self):
        ref = station.build().model
        with tempfile.TemporaryDirectory() as tmp:
            path = station.export_mjcf(tmp)
            m = mujoco.MjModel.from_xml_path(str(path))
            self.assertEqual((m.nbody, m.ngeom, m.nmesh), (ref.nbody, ref.ngeom, ref.nmesh))
            self.assertLess(float(np.abs(m.geom_size - ref.geom_size).max()), 1e-5)
            self.assertLess(float(np.abs(m.geom_pos - ref.geom_pos).max()), 1e-5)
            self.assertTrue((m.mesh_vertnum == ref.mesh_vertnum).all())
            # no slash in any name (Gazebo's converter reserves it), every mesh in a file
            for b in range(m.nbody):
                self.assertNotIn("/", m.body(b).name)
            self.assertNotIn(" vertex=", path.read_text())
            # visual groups folded into the converter's visual group
            self.assertTrue(set(int(g) for g in m.geom_group) <= {0, 3})

    def test_joint_angles_come_back_from_body_orientations(self):
        from sightline_ros.replay_live import JOINTS  # noqa: E402
        m = station.build().model
        d = mujoco.MjData(m)
        jids = [m.joint("ur5e/" + j).id for j in JOINTS]
        qadr = [m.jnt_qposadr[j] for j in jids]
        rng = np.random.default_rng(3)
        for _ in range(10):
            q = rng.uniform(-3.0, 3.0, 6)
            d.qpos[qadr] = q
            mujoco.mj_forward(m, d)
            got = joint_angles(m, jids, d.xquat)
            wrapped = (got - q + np.pi) % (2 * np.pi) - np.pi
            self.assertLess(float(np.abs(wrapped).max()), 1e-9)


def joint_angles(m, jids, xquat):
    from sightline_ros.replay_live import joint_angles as f
    return f(m, jids, xquat)


class Frames(unittest.TestCase):
    def test_take_frame_by_stamp(self):
        from sightline_sim.gazebo.replay_gazebo import take_frame
        with tempfile.TemporaryDirectory() as tmp:
            for ns in (0, 40_000_000, 80_000_000, 120_000_000):
                open(os.path.join(tmp, f"cam_t{ns:015d}.png"), "w").close()
            open(os.path.join(tmp, "cam_00007.png"), "w").close()      # an earlier taken frame stays
            self.assertTrue(take_frame(tmp, "cam", 40_000_000, 40_000_000, 1, 0.1))
            names = sorted(os.listdir(tmp))
            self.assertIn("cam_00001.png", names)                        # the frame at 0.04 s, by index
            self.assertNotIn("cam_t000000000000000.png", names)          # the one before the pose: dropped
            self.assertIn("cam_00007.png", names)
            self.assertIn("cam_t000000080000000.png", names)             # later frames untouched
            # a frame a whole period past the wanted one is a stepping fault, not a match
            self.assertFalse(take_frame(tmp, "cam", 40_000_000, 40_000_000, 2, 0.1))



class Lockstep(unittest.TestCase):
    """The message helpers of the live nodes: what goes in comes out."""

    def test_messages_round_trip(self):
        from sightline_ros.lockstep import depth_to_msg, msg_to_depth, to_json, from_json, cloud_msg, stamp
        depth = np.random.default_rng(0).uniform(0.5, 4.0, (48, 64))
        back = msg_to_depth(depth_to_msg(depth, 12.345))
        self.assertEqual(back.shape, depth.shape)
        self.assertLess(float(np.abs(back - depth).max()), 1e-6)       # float32 on the wire
        s = stamp(12.345)
        self.assertEqual((s.sec, s.nanosec), (12, 345000000))
        d = from_json(to_json({"tick": np.int64(7), "q": np.arange(6.0), "ok": np.bool_(True), "x": np.float32(1.5)}))
        self.assertEqual(d["tick"], 7)
        self.assertEqual(d["q"], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertIs(d["ok"], True)
        cloud = cloud_msg(np.zeros((10, 3)), (1.0, 0.5, 0.0), 1.0)
        self.assertEqual((cloud.width, cloud.point_step, len(cloud.data)), (10, 16, 160))
        rgb = np.frombuffer(cloud.data, dtype=np.uint32)[3]             # the rgb field of the first point
        self.assertEqual(rgb, (255 << 16) | (128 << 8))
        self.assertEqual(cloud_msg(np.zeros((0, 3)), (1, 1, 1), 1.0).width, 0)


if __name__ == "__main__":
    unittest.main()
