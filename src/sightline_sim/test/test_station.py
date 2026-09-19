"""Station invariants, and the findings from gate 1 encoded so they cannot come back."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from sightline_sim import geometry as G, measure, pose, station, worker  # noqa: E402


class StationCase(unittest.TestCase):
    """One built scene for the whole class: a build takes a second and holds meshes."""

    @classmethod
    def setUpClass(cls):
        cls.scene = station.build()
        cls.m = cls.scene.model
        cls.d = mujoco.MjData(cls.m)
        mujoco.mj_forward(cls.m, cls.d)

    @classmethod
    def tearDownClass(cls):
        del cls.scene, cls.m, cls.d


class TestScene(StationCase):
    def test_scene_compiles_with_the_expected_pieces(self):
        names = [self.m.body(i).name for i in range(self.m.nbody)]
        for wanted in ("bench", "jig", "props", "rack", "eyes", "robot_mount", "worker_base",
                       station.ROBOT_PREFIX + "wrist_3_link", station.HUMAN_PREFIX + "head"):
            self.assertIn(wanted, names)
        self.assertGreater(self.m.ngeom, 400)

    def test_worker_keeps_its_mass_and_height(self):
        """Scaling the humanoid must not change what the judge weighs."""
        total = sum(self.m.body_mass[i] for i in range(self.m.nbody)
                    if self.m.body(i).name.startswith(station.HUMAN_PREFIX))
        self.assertAlmostEqual(total, 70.0, delta=0.5)
        spec, upright = worker.cmu_humanoid_spec()
        worker.scale_humanoid(spec, self.scene.worker_scale)
        self.assertAlmostEqual(worker.stature(spec, upright), worker.WORKER_STATURE_M, delta=0.01)

    def test_bench_height_follows_elbow_height(self):
        """CCOHS: light work 5 to 10 cm below elbow height. We use the middle of that band."""
        drop = self.scene.points["elbow_height"] - self.scene.worktop_z
        self.assertAlmostEqual(drop, self.scene.layout.elbow_to_worktop, delta=0.001)
        self.assertGreaterEqual(drop, 0.05)
        self.assertLessEqual(drop, 0.10)


class TestTool(StationCase):
    def test_tool_tip_matches_the_datasheet(self):
        """Flange centre 166.6 mm above the bit tip, screw axis 93.5 mm off the flange face."""
        site = self.m.site("tool_tip").id
        w3 = self.m.body(station.ROBOT_PREFIX + "wrist_3_link").id
        R = self.d.xmat[w3].reshape(3, 3)
        local = R.T @ (self.d.site_xpos[site] - self.d.xpos[w3])
        # the attachment site sits at y = 0.1 in the wrist frame (Menagerie ur5e.xml)
        self.assertAlmostEqual(local[1] - 0.1, station.SCREW_AXIS_OFFSET, delta=0.001)
        self.assertAlmostEqual(local[2], station.SD_FLANGE_ABOVE_TIP, delta=0.001)

    def test_turning_around_the_screw_changes_reach(self):
        """The tool mounts from the side, so some turn angles cannot be reached at all.

        Found in gate 1: at the rear right cover hole the angles that put the mount
        toward the robot base have no solution, while the opposite side does.
        """
        hole = np.array(self.scene.points["cover_holes"][3])
        tip = hole + [0.0, 0.0, 0.0236]
        base = np.column_stack([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        seeds = [np.array([pan, lift, elbow, -1.6, w2, 0.0])
                 for pan in np.linspace(-np.pi, np.pi, 8, endpoint=False)
                 for lift in (-2.2, -1.4) for elbow in (-2.0, 2.0) for w2 in (-1.57, 1.57)]
        toward_base = pose.robot_ik(self.m, self.d, "tool_tip", tip, base, seeds, iters=250)[0][0]
        away = pose.robot_ik(self.m, self.d, "tool_tip", tip, G.rot_z(np.pi) @ base, seeds, iters=250)[0][0]
        self.assertGreater(toward_base, 1e-3)
        self.assertLess(away, 1e-5)


class TestCameras(StationCase):
    def test_eyes_camera_sees_the_jig(self):
        """The camera must not look into its own housing or its mount."""
        cam = np.asarray(self.scene.points["eyes_pos"], float)
        # the middle of the cover, a surface the camera has to see (a point inside the box is hidden by the box)
        target = np.mean(np.array(self.scene.points["cover_holes"], float), axis=0)
        vec = target - cam
        dist = float(np.linalg.norm(vec))
        gid = np.zeros(1, np.int32)
        # station geoms only: the robot and the worker stand wherever the default pose puts them,
        # and this test is about the camera's own housing, mount and the cell frame
        station_only = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        hit = mujoco.mj_ray(self.m, self.d, cam, vec / dist, station_only, 1,
                            self.m.body("eyes").id, gid)
        blocker = self.m.body(self.m.geom_bodyid[int(gid[0])]).name if gid[0] >= 0 else "nothing"
        self.assertTrue(gid[0] < 0 or hit >= dist - 0.01,
                        f"{blocker} blocks the view of the cover at {hit:.3f} m of {dist:.3f} m")

    def test_camera_fields_of_view_come_from_the_datasheets(self):
        self.assertAlmostEqual(self.m.cam("eyes_cam").fovy[0], station.D455_FOVY, places=3)
        self.assertAlmostEqual(self.m.cam("wrist_cam").fovy[0], station.D405_FOVY, places=3)


class TestWorkerPoses(StationCase):
    def test_hands_reach_their_targets(self):
        """Solving position and hand orientation together used to stall 0.3 m away."""
        poser = pose.WorkerPoser(self.m, self.d)
        for wp in measure.work_poses(self.scene)[:3]:
            errs = measure.apply_work_pose(self.scene, self.d, wp, poser)
            for side, err in errs.items():
                self.assertLess(err, 0.001, f"{wp.name}: {side} off by {err * 1000:.1f} mm")


if __name__ == "__main__":
    unittest.main()


class TestCycle(StationCase):
    """Gate 2 findings: the scripted cycle has to stay inside the worker's own envelope."""

    def test_every_cycle_target_is_within_arm_reach(self):
        """Twice now a target was written where the arm simply cannot go.

        The first cycle put the rack picks 0.82 to 0.96 m from the shoulder against
        an arm of 0.562 m, and the arm hung in the air. This checks the geometry
        before anyone waits for the solver to tell them.
        """
        from sightline_sim import script
        poser = pose.WorkerPoser(self.m, self.d)
        L = self.scene.layout
        default = (L.worker_xy[0], L.worker_xy[1], np.pi / 2)
        arm = self._max_arm_reach(poser)
        worst = ("", 0.0)
        for seg in script.cycle(self.scene):
            stance = seg.stance or default
            poser.stand(stance[:2], stance[2])
            poser.lean(seg.lean)
            mujoco.mj_kinematics(self.m, self.d)
            for side, target in (("l", seg.left), ("r", seg.right)):
                shoulder = self.d.xpos[self.m.body(station.HUMAN_PREFIX + side + "clavicle").id]
                if target is None:
                    continue
                dist = float(np.linalg.norm(np.asarray(target, float) - shoulder))
                if dist > worst[1]:
                    worst = (f"{seg.name} ({side})", dist)
        self.assertGreater(arm, 0.4)
        self.assertLess(worst[1], arm,
                        f"{worst[0]}: {worst[1] * 1000:.0f} mm from the shoulder, the arm reaches {arm * 1000:.0f} mm")

    def _max_arm_reach(self, poser) -> float:
        """How far the hand can get from the clavicle, sampled over the arm's joint limits.

        The hanging arm is not the answer: the elbow is bent, so measuring there
        understates the reach by about 7 cm.
        """
        rng = np.random.default_rng(0)
        joints = [poser.j("l" + n) for n in pose.ARM_CHAIN]  # poser.j adds the prefix
        adr = [int(self.m.jnt_qposadr[j.id]) for j in joints]
        lo = np.array([j.range[0] for j in joints])
        hi = np.array([j.range[1] for j in joints])
        clav = self.m.body(station.HUMAN_PREFIX + "lclavicle").id
        hand = self.m.body(station.HUMAN_PREFIX + "lhand").id
        best = 0.0
        for _ in range(400):
            self.d.qpos[adr] = lo + rng.random(len(adr)) * (hi - lo)
            mujoco.mj_kinematics(self.m, self.d)
            best = max(best, float(np.linalg.norm(self.d.xpos[hand] - self.d.xpos[clav])))
        return best

    def test_lift_segments_add_a_via_pose(self):
        """A hand that slides sideways off a part cuts through it, so transits lift first."""
        from sightline_sim import script
        segs = script.cycle(self.scene)
        lifts = [s for s in segs if s.lift is not None]
        self.assertGreaterEqual(len(lifts), 6)
        for s in lifts:
            dz, secs = s.lift
            self.assertGreater(dz, 0.0)
            self.assertLess(secs, s.duration, f"{s.name}: the lift eats the whole segment")


class TestCollisionModel(StationCase):
    """What the judge can see depends on these shapes being right."""

    def test_the_bit_tip_is_covered_by_a_collision_geom(self):
        """R1 covers the arm, the screwdriver, the wrist camera and the screw.

        The tip once sat 74 mm outside every robot collision geom, so the contact
        most likely to happen, a poke with the screw, would have scored as clear.
        """
        tip = self.d.site_xpos[self.m.site("tool_tip").id]
        best = 1e9
        for g in range(self.m.ngeom):
            if not self.m.body(self.m.geom_bodyid[g]).name.startswith(station.ROBOT_PREFIX):
                continue
            if self.m.geom_group[g] != station.GROUP_COLLISION:
                continue
            if self.m.geom_type[g] != mujoco.mjtGeom.mjGEOM_CAPSULE:
                continue
            p, R, s = self.d.geom_xpos[g], self.d.geom_xmat[g].reshape(3, 3), self.m.geom_size[g]
            local = R.T @ (tip - p)
            along = np.array([local[0], local[1], max(abs(local[2]) - float(s[1]), 0.0)])
            best = min(best, float(np.linalg.norm(along)) - float(s[0]))
        self.assertLess(best, 0.0, f"the bit tip is {best * 1000:.0f} mm outside every robot capsule")

    def test_the_enclosure_is_hollow(self):
        """The rail screws are driven inside the box.

        With a solid collision block every pose that reached them counted as a
        crash and the reach table read 0 of 24.
        """
        inside = np.array(self.scene.points["box_home"], float)
        inside[2] += self.scene.layout.box_size[2] / 2
        for g in range(self.m.ngeom):
            if self.m.body(self.m.geom_bodyid[g]).name != "part_base":
                continue
            if self.m.geom_group[g] != station.GROUP_COLLISION:
                continue
            local = self.d.geom_xmat[g].reshape(3, 3).T @ (inside - self.d.geom_xpos[g])
            s = self.m.geom_size[g]
            if self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                hit = bool(np.all(np.abs(local) <= s[:3]))
            else:  # the screw bosses
                hit = bool(np.hypot(local[0], local[1]) <= s[0] and abs(local[2]) <= s[1])
            self.assertFalse(hit, f"geom {g} fills the middle of the enclosure")

    def test_a_hand_sample_includes_the_thumb(self):
        """Dropping it moved every per-hand visibility number by a few points."""
        poser = pose.WorkerPoser(self.m, self.d)
        poser.stand((0.02, -0.61), np.pi / 2)
        mujoco.mj_forward(self.m, self.d)
        cam = np.asarray(self.scene.points["eyes_pos"], float)
        seen = measure.worker_seen(self.m, self.d, cam, stride=5)
        whole = len(measure.surface_points(self.m, self.d, ("lhand", "lfingers", "lthumb"), 5))
        self.assertEqual(seen["lhand"]["points"], whole)
        self.assertGreater(whole, len(measure.surface_points(self.m, self.d, ("lhand", "lfingers"), 5)))
