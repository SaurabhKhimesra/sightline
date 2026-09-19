"""The sensors and the perception pipeline, and the gate 4 bugs that must not come back."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from sightline_sim.judge import PerceptionTruth, match  # noqa: E402
from sightline_planner import Taught, ToolKinematics  # noqa: E402
from sightline_planner.perceive import Calibration, Perception, find_hole  # noqa: E402
from sightline_sim import measure, pose, sensors, station  # noqa: E402


class PerceptionCase(unittest.TestCase):
    """One scene and one camera for the whole class: a build costs a second."""

    @classmethod
    def setUpClass(cls):
        cls.scene = station.build()
        cls.m = cls.scene.model
        cls.d = mujoco.MjData(cls.m)
        mujoco.mj_forward(cls.m, cls.d)
        cls.depth = sensors.DepthCamera(cls.m, "eyes_cam", 640, 480, seed=0, delay_frames=0)

    @classmethod
    def tearDownClass(cls):
        cls.depth.close()
        del cls.scene, cls.m, cls.d


class TestSensor(PerceptionCase):
    def test_a_camera_pose_read_too_early_is_refused(self):
        """Camera frames come from mj_camlight. Read after mj_kinematics alone they are
        the origin with a zero rotation, and every unprojected point lands at (0,0,0)."""
        fresh = mujoco.MjData(self.m)
        with self.assertRaises(RuntimeError):
            sensors.camera_pose(self.m, fresh, "eyes_cam")

    def test_depth_noise_is_not_extrapolated_past_its_fit(self):
        """Nguyen et al. fitted 0.5 to 2.75 m. Run out to a 6 m hall wall it gives
        60 mm, which turned half the floor into a person."""
        self.assertAlmostEqual(float(sensors.axial_sigma(2.75)), float(sensors.axial_sigma(6.0)), places=9)
        self.assertLess(float(sensors.axial_sigma(6.0)), 0.015)
        self.assertGreater(float(sensors.axial_sigma(2.5)), float(sensors.axial_sigma(1.0)))

    def test_unprojection_agrees_with_ray_casting(self):
        """Half a pixel out puts every point 4 mm out at 2.5 m."""
        raw = self.depth.raw(self.d)
        f, cx, cy = self.depth.intrinsics()
        cam_pos, cam_R = sensors.camera_pose(self.m, self.d, "eyes_cam")
        groups = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)
        gid = np.zeros(1, np.int32)
        rng = np.random.default_rng(0)
        errors = []
        for _ in range(400):
            v, u = int(rng.integers(40, 440)), int(rng.integers(40, 600))
            z = float(raw[v, u])
            if not 0 < z < 4.0:
                continue
            local = np.array([(u - cx) / f, -(v - cy) / f, -1.0])
            direction = cam_R @ (local / np.linalg.norm(local))
            hit = mujoco.mj_ray(self.m, self.d, cam_pos, direction, groups, 1, -1, gid)
            if gid[0] < 0:
                continue
            here = cam_pos + direction * (z / abs(float(cam_R[:, 2] @ direction)))
            errors.append(float(np.linalg.norm(here - (cam_pos + direction * hit))))
        self.assertGreater(len(errors), 100)
        self.assertLess(float(np.median(errors)), 0.003)


class TestPipeline(PerceptionCase):
    def _perception(self):
        cam_pos, cam_R = sensors.camera_pose(self.m, self.d, "eyes_cam")
        calib = Calibration(pos=cam_pos, R=cam_R, width=640, height=480,
                            fovy_deg=float(self.m.cam_fovy[self.m.cam("eyes_cam").id]))
        kin = ToolKinematics(station.robot_only_model())
        base = np.array(self.scene.points["box_home"], float)
        taught = Taught(part_pose=tuple(base),
                        cover_holes=[tuple(np.array(h, float) - base) for h in self.scene.points["cover_holes"]],
                        feeder_pick=tuple(np.array(self.scene.points["feeder_pick"], float)))
        per = Perception(calib, kin, taught)
        per.set_background(sensors.empty_station_depth(self.depth, self.m, self.d))
        return per

    def test_the_background_map_leaves_the_box_in_the_foreground(self):
        """Taken with the parts in place, the box becomes background and the box fit
        has only the worker's hands to look at."""
        per = self._perception()
        depth = self.depth.spoil(self.depth.raw(self.d)).reshape(-1)
        keep = (depth > 0) & (depth < per.background_gate)
        points = per.calib.pos + per.rays[keep] * (depth[keep] * per.range_scale[keep])[:, None]
        base = np.array(self.scene.points["box_home"], float)
        near_box = (np.linalg.norm(points[:, :2] - base[:2], axis=1) < 0.12) & (points[:, 2] > base[2] + 0.03)
        self.assertGreater(int(near_box.sum()), 200, "the box is not in the foreground")

    def test_it_finds_the_worker(self):
        per = self._perception()
        poser = pose.WorkerPoser(self.m, self.d)
        measure.apply_work_pose(self.scene, self.d, measure.work_poses(self.scene)[0], poser)
        mujoco.mj_forward(self.m, self.d)
        truth = PerceptionTruth(self.m, "eyes_cam", 640, 480)
        try:
            model = per.update(self.depth.spoil(self.depth.raw(self.d)),
                               self.d.qpos[pose.robot_qadr(self.m)].copy(), 0.0)
            recall, precision = match(model.voxels, truth.person_voxels(self.d), per.voxel)
        finally:
            truth.close()
        self.assertGreater(recall, 0.80, f"recall {recall:.2f}")
        self.assertGreater(precision, 0.90, f"precision {precision:.2f}")
        self.assertGreater(len(model.unseen), len(model.voxels), "no unseen space behind the person")


class TestHoleDetector(PerceptionCase):
    def test_it_finds_a_hole_and_not_a_shadow(self):
        """Taking the centroid of everything dark lands 7 to 16 mm off."""
        from sightline_planner import B0
        kin = ToolKinematics(station.robot_only_model())
        base = np.array(self.scene.points["box_home"], float)
        taught = Taught(part_pose=tuple(base),
                        cover_holes=[tuple(np.array(h, float) - base) for h in self.scene.points["cover_holes"]],
                        feeder_pick=tuple(np.array(self.scene.points["feeder_pick"], float)))
        b0 = B0.build(kin, taught, park_tip=(-0.45, 0.33, self.scene.worktop_z + 0.30))
        truth = PerceptionTruth(self.m, "eyes_cam", 640, 480)
        wrist = sensors.Camera(self.m, "wrist_cam", 640, 480)
        try:
            found = []
            for screw in b0.screws:
                for psi in (0, 45, 90, 135, 180, 225, 270, 315):
                    from sightline_planner.b0 import BIT_DOWN, rot_z
                    above = np.asarray(screw.hole, float) + [0.0, 0.0, taught.screw_length + taught.approach]
                    q, err = kin.ik_multi(above, rot_z(np.radians(psi)) @ BIT_DOWN,
                                          np.array([0.0, -1.6, 1.6, -1.6, -1.57, 0.0]))
                    if not np.isfinite(err) or err > 1e-4:
                        continue
                    self.d.qpos[pose.robot_qadr(self.m)] = q
                    mujoco.mj_forward(self.m, self.d)
                    spot = truth.hole_pixel(self.d, screw.hole)
                    if spot is None:
                        continue
                    u, v, dist, f = spot
                    got = find_hole(wrist.rgb(self.d), (u, v), 0.0046 * f / dist)
                    if got is None:
                        continue
                    cam_pos, cam_R = sensors.camera_pose(self.m, self.d, "wrist_cam")
                    ray = cam_R @ np.array([(got[0] - 319.5) / f, -(got[1] - 239.5) / f, -1.0])
                    world = cam_pos + ray * ((screw.hole[2] - cam_pos[2]) / ray[2])
                    found.append(float(np.linalg.norm(world[:2] - np.asarray(screw.hole)[:2])))
                    break
        finally:
            truth.close()
            wrist.close()
        self.assertGreaterEqual(len(found), 4, "the detector found almost nothing")
        self.assertLess(float(np.median(found)), 0.004, f"median error {np.median(found) * 1000:.1f} mm")


if __name__ == "__main__":
    unittest.main()
