"""The cell as a ROS 2 node: MuJoCo, the worker's cycle, the eyes camera, the jig's
signals and the judge, in lockstep with the planner node over topics.

    python ros2/sim_node.py --seed 0 [--seconds 6] [--out results/ros2/stage1]

Every 10 ms of sim time it publishes /joint_states and /sightline/tick (JSON: the
tick number, the time, the joint state, the jig signal and the index of the last
picture) and waits for the /sightline/cmd that answers the tick. Every 40 ms it
publishes the eyes camera's depth picture (/eyes/depth, one frame late and spoiled
as in gate 4) with /sightline/frame naming it, then the colour picture, the judge's
record, the worker's true shapes and the clock for rviz. Once, kept for a late
subscriber: the planner's calibration (with its errors, drawn as gate 5 draws them),
the empty station's depth, the blind cells and the taught points.

The run is the gate 5 runner's (scripts/gate5_planner.py) with the planner on the
other side of the topics; the judge stays here, with the truth. The poses of every
body are recorded as --record-poses does, so ros2/replay_gazebo.py can render the run.
"""
import argparse
import os
import sys
import time
from collections import deque

from . import lockstep  # noqa: E402,F401  (fixes the import path before rclpy)
from .lockstep import TOPICS, LATCHED, stamp, to_json, from_json, depth_to_msg, rgb_to_msg, caption_image  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from sightline_sim.data import park_pose  # noqa: E402
import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.utilities import remove_ros_args  # noqa: E402
from rosgraph_msgs.msg import Clock  # noqa: E402
from sensor_msgs.msg import Image, JointState  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from geometry_msgs.msg import TransformStamped  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402
from visualization_msgs.msg import MarkerArray  # noqa: E402

from sightline_planner.settings import (FPS, MOTION_HZ, CAM_W, CAM_H, CALIB_POS_MM, CALIB_DEG,  # noqa: E402
                                        HELD_FRACTION, HOLD_ASK_S)
from sightline_sim.episode import ROBOT_WORDS, blind_cells, rotation_error  # noqa: E402
from sightline_sim.gates.gate3_b0 import jig_signal_times, jittered_cycle  # noqa: E402
from .replay_live import JOINTS, WORKER_COLOR, shape_markers, place_markers  # noqa: E402
from sightline_sim.judge import R1Judge, R2Judge  # noqa: E402
from sightline_planner import Taught, ToolKinematics  # noqa: E402
from sightline_sim import pose, script, sensors, station  # noqa: E402


class Cell(Node):
    def __init__(self, args):
        super().__init__("sightline_cell")
        self.args = args
        seed = args.seed
        self.rng = rng = np.random.default_rng(seed)
        self.scene = scene = station.build()
        self.m, self.d = m, d = scene.model, mujoco.MjData(scene.model)
        self.qadr = pose.robot_qadr(m)
        self.park = park = park_pose()
        d.qpos[self.qadr] = park
        mujoco.mj_forward(m, d)
        script.parts_to_start(scene, d)
        self.get_logger().info(f"seed {seed}: solving the worker's key poses")
        self.pb = pb = script.build_playback(scene, d, segments=jittered_cycle(scene, rng))
        self.signals = jig_signal_times(pb)
        zw = scene.worktop_z
        self.kin = ToolKinematics(station.robot_only_model())
        base = np.array(scene.points["box_home"], float)
        self.taught = Taught(part_pose=tuple(base),
                             rail_holes=[tuple(np.array(h, float) - base) for h in scene.points["rail_holes"]],
                             cover_holes=[tuple(np.array(h, float) - base) for h in scene.points["cover_holes"]],
                             feeder_pick=tuple(np.array(scene.points["feeder_pick"], float) + [0, 0, 0.004]))
        self.eyes = sensors.DepthCamera(m, "eyes_cam", CAM_W, CAM_H, seed=seed, delay_frames=1)
        cam_pos, cam_R = sensors.camera_pose(m, d, "eyes_cam")
        # the planner's belief of where the camera is, drawn from the same generator in
        # the same order as the gate 5 runner, so seed 0 here is seed 0 there
        self.calib = dict(pos=cam_pos + rng.normal(0, CALIB_POS_MM / 1000, 3),
                          R=rotation_error(rng, CALIB_DEG) @ cam_R, width=CAM_W, height=CAM_H,
                          fovy_deg=float(m.cam_fovy[m.cam("eyes_cam").id]))
        self.empty = sensors.empty_station_depth(self.eyes, m, d)
        jig = (base, np.array(self.taught.cover_size) / 2 + 0.02, base[2] + self.taught.cover_top + 0.03)
        self.hidden = blind_cells(scene, cam_pos, jig=jig)
        self.get_logger().info(f"static knowledge: {len(self.hidden)} blind cells, worktop at {zw:.3f} m")
        self.r1 = R1Judge(m, self.qadr)
        self.r2 = R2Judge(m, self.qadr, "eyes_cam", CAM_W, CAM_H)
        self.colour = sensors.Camera(m, "eyes_cam", 640, 480)
        self.dt = pb.dt
        self.motion_every = max(1, int(round(1.0 / (MOTION_HZ * self.dt))))
        self.cycle_back = max(1, int(round(0.010 / self.dt)))
        self.frame_back = max(1, int(round((1.0 / FPS) / self.dt)))
        self.q = np.array(park, float)
        self.qd = np.zeros(6)
        self.history: list = []
        self.state = {"step": 0, "tick": 0, "frame_index": 0, "signal": "empty", "held_since": None, "asks": [],
                      "held_s": 0.0, "hand_speed": 0.0, "prev_hand": None, "yield_s": 0.0, "q_last_frame": None,
                      "cmd": {"state": "wait", "verdict": "", "screw": "", "screws_done": 0, "want_speed": 0.0,
                              "yield_since": 0.0}, "timeouts": 0, "waits_s": []}
        self.cmds: dict = {}
        self.poses: list = []
        worker_bodies = {b for b in range(m.nbody) if m.body(b).name.startswith(station.HUMAN_PREFIX)}
        kinds = [int(k) for k in m.geom_type]
        self.worker_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in worker_bodies
                             and m.geom_group[g] == station.GROUP_COLLISION and kinds[g] != int(mujoco.mjtGeom.mjGEOM_MESH)]
        self.worker_shapes, mid = [], 0
        for g in self.worker_geoms:
            shaped = shape_markers(m, g, WORKER_COLOR, "worker", mid)
            mid += len(shaped)
            self.worker_shapes.append((g, shaped))
        self.worker_array = MarkerArray()
        self.worker_array.markers = [mk for _, shaped in self.worker_shapes for mk, _ in shaped]
        # publishers
        self.pub = {
            "joints": self.create_publisher(JointState, TOPICS["joints"], 10),
            "tick": self.create_publisher(String, TOPICS["tick"], 10),
            "clock": self.create_publisher(Clock, TOPICS["clock"], 10),
            "depth": self.create_publisher(Image, TOPICS["depth"], 5),
            "frame": self.create_publisher(String, TOPICS["frame"], 10),
            "image": self.create_publisher(Image, TOPICS["image"], 5),
            "judge": self.create_publisher(String, TOPICS["judge"], 10),
            "worker": self.create_publisher(MarkerArray, TOPICS["worker"], 5),
            "station": self.create_publisher(MarkerArray, TOPICS["station"], LATCHED),
            "caption": self.create_publisher(Image, TOPICS["caption"], 5),
            "tf": self.create_publisher(TFMessage, "/tf", 10),
            "calibration": self.create_publisher(String, TOPICS["calibration"], LATCHED),
            "background": self.create_publisher(Image, TOPICS["background"], LATCHED),
            "blind": self.create_publisher(String, TOPICS["blind"], LATCHED),
            "cell": self.create_publisher(String, TOPICS["cell"], LATCHED),
        }
        self.create_subscription(String, TOPICS["cmd"], self.on_cmd, 50)
        self.planner_ready = False
        self.create_subscription(String, TOPICS["ready"], self.on_ready, LATCHED)
        self.publish_knowledge()
        self.t_wall = time.time()

    # ------------------------------------------------------------ once, kept
    def publish_knowledge(self):
        self.pub["calibration"].publish(to_json(self.calib))
        self.pub["background"].publish(depth_to_msg(self.empty, 0.0))
        self.pub["blind"].publish(to_json({"cells": self.hidden}))
        tg = self.taught
        self.pub["cell"].publish(to_json({
            "part_pose": tg.part_pose, "rail_holes": tg.rail_holes, "cover_holes": tg.cover_holes,
            "feeder_pick": tg.feeder_pick, "park": self.park, "worktop_z": self.scene.worktop_z,
            "dt": self.dt, "fps": FPS, "motion_hz": MOTION_HZ}))
        # the station's shapes, the same as the replay shows
        arr = MarkerArray()
        d, m = self.d, self.m
        kinds = [int(k) for k in m.geom_type]
        worker_bodies = {b for b in range(m.nbody) if m.body(b).name.startswith(station.HUMAN_PREFIX)}
        robot_bodies = {b for b in range(m.nbody) if m.body(b).name.startswith(pose.ROBOT_PREFIX)}
        mid = 0
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in worker_bodies | robot_bodies or m.geom_group[g] != 0:
                continue
            if kinds[g] not in (int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
                continue
            mat = m.geom_matid[g]
            rgba = m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[g]
            shaped = shape_markers(m, g, (rgba[0], rgba[1], rgba[2], 0.55), "station", mid)
            mid += len(shaped)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, d.geom_xmat[g])
            place_markers(shaped, d.geom_xpos[g].copy(), quat, stamp(0.0))
            arr.markers.extend(mk for mk, _ in shaped)
        self.pub["station"].publish(arr)

    # ------------------------------------------------------------ every 10 ms
    def on_cmd(self, msg):
        cmd = from_json(msg)
        self.cmds[int(cmd["tick"])] = cmd

    def on_ready(self, msg):
        self.planner_ready = True

    def wait_planner(self, timeout_s: float = 900.0) -> bool:
        """The planner builds its world once it has the cell's knowledge; no tick before."""
        t0 = time.time()
        while not self.planner_ready and time.time() - t0 < timeout_s:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info(f"planner ready after {time.time() - t0:.0f} s" if self.planner_ready
                               else "no planner after the wait; running anyway, every tick will hold")
        return self.planner_ready

    def wait_cmd(self, tick: int):
        deadline = time.time() + self.args.timeout
        while tick not in self.cmds:
            rclpy.spin_once(self, timeout_sec=0.002)
            if time.time() > deadline:
                return None
        return self.cmds.pop(tick)

    def on_step(self, t: float) -> None:
        st, i = self.state, self.state["step"]
        while self.signals and t >= self.signals[0][0]:
            st["signal"] = self.signals.pop(0)[1]
        if i % self.motion_every == 0:
            n = st["tick"]
            self.pub["clock"].publish(Clock(clock=stamp(t)))
            js = JointState()
            js.header.stamp = stamp(t)
            js.name, js.position, js.velocity = JOINTS, self.q.tolist(), self.qd.tolist()
            self.pub["joints"].publish(js)
            self.pub["tick"].publish(to_json({"tick": n, "t": t, "frame": st["frame_index"], "signal": st["signal"],
                                              "q": self.q, "qd": self.qd}))
            t_wait = time.time()
            cmd = self.wait_cmd(n)
            st["waits_s"].append(time.time() - t_wait)
            if cmd is None:
                st["timeouts"] += 1
                self.get_logger().warning(f"tick {n} at t={t:.2f} s: no command within {self.args.timeout} s, holding")
                self.qd = np.zeros(6)
            else:
                self.qd = np.asarray(cmd["qd"], float)
                st["cmd"] = cmd
                # held: the task wanted to move and the rules let almost none of it through
                want = float(cmd["want_speed"])
                if want > 0.02 and float(np.linalg.norm(self.qd)) < HELD_FRACTION * want:
                    st["held_s"] += self.motion_every * self.dt
                    if st["held_since"] is None:
                        st["held_since"] = t
                else:
                    st["held_since"] = None
                if cmd["state"] == "yield":
                    st["yield_s"] += self.motion_every * self.dt
                    since = float(cmd["yield_since"])
                    if t - since >= HOLD_ASK_S and (not st["asks"] or st["asks"][-1] < since):
                        st["asks"].append(round(t, 2))
            st["tick"] = n + 1
        self.q = self.kin.clamp(self.q + self.qd * self.dt)
        self.d.qpos[self.qadr] = self.q
        st["step"] = i + 1

    def after_step(self, t: float) -> None:
        st, d, m = self.state, self.d, self.m
        self.history.append(self.q.copy())
        before = self.history[-self.cycle_back] if len(self.history) > self.cycle_back else None
        hands = [d.xpos[m.body(station.HUMAN_PREFIX + s + "hand").id].copy() for s in ("l", "r")]
        if st["prev_hand"] is not None:
            st["hand_speed"] = max(float(np.linalg.norm(a - b)) / self.dt for a, b in zip(hands, st["prev_hand"]))
        st["prev_hand"] = hands
        self.r1.step(d, t, before, person_speed=st["hand_speed"], measure_distance=(st["step"] % self.motion_every == 0))

    # ------------------------------------------------------------ every 40 ms
    def on_frame(self, t: float, index: int, seg) -> None:
        st, d, m = self.state, self.d, self.m
        depth = self.eyes.capture(d)
        seen_q = st["q_last_frame"] if st["q_last_frame"] is not None else self.q.copy()
        st["q_last_frame"] = self.q.copy()
        if index > 0:
            # the picture first, then the tick that names it: the planner answers no
            # tick before it has looked at the picture
            self.pub["depth"].publish(depth_to_msg(depth, t))
            self.pub["frame"].publish(to_json({"index": index, "t": t, "taken_at": t - 1.0 / FPS, "q": self.q,
                                               "seen_q": seen_q, "signal": st["signal"]}))
            st["frame_index"] = index
        before = self.history[-self.cycle_back] if len(self.history) > self.cycle_back else None
        last = self.history[-self.frame_back] if len(self.history) > self.frame_back else None
        self.r2.frame(d, t, before, last)
        # for rviz and the record: what the judge saw
        cmd = st["cmd"]
        by_robot = sum(c.who_moved_in == "robot" for c in self.r1.contacts)
        by_him = sum(c.who_moved_in == "person" for c in self.r1.contacts)
        views = sum(e.who_moved_in_frame == "robot" for e in self.r2.events)
        blocked = self.r2.blocked_frames if hasattr(self.r2, "blocked_frames") else len(self.r2.blocked)
        alert = ""
        if self.r1._open:
            alert = "TOUCHING HIM (" + ("HE MOVED IN" if all(c.who_moved_in == "person" for c in self.r1._open.values())
                                       else "THE ROBOT MOVED IN") + ")"
        elif self.r2._open is not None:
            alert = "HIS VIEW IS BLOCKED"
        screw = cmd.get("screw", "")
        doing = ROBOT_WORDS.get(cmd["state"], cmd["state"].upper()).format(s=screw.replace("_", " SCREW ").upper())
        verdict = cmd.get("verdict", "")
        if verdict == "brake":
            doing += " | HOLDING: HE IS CLOSE"
        elif verdict.startswith("escape"):
            doing += " | BACKING AWAY FROM HIM"
        judge = {"t": t, "index": index, "worker": seg.name, "robot": cmd["state"], "screw": screw,
                 "screws_done": int(cmd["screws_done"]), "contacts": len(self.r1.contacts), "by_robot": by_robot,
                 "by_him": by_him, "blocked_frames": int(blocked), "views_by_robot": views, "alert": alert,
                 "verdict": verdict}
        self.pub["judge"].publish(to_json(judge))
        lines = [(f"SIGHTLINE  B4 ON ROS 2   SIMULATION  T {t:5.1f} S", (0.95, 0.95, 0.92)),
                 (f"ROBOT: {doing}", (0.55, 0.85, 1.0)),
                 (f"WORKER: {seg.name.upper()}", (1.0, 0.85, 0.45)),
                 (f"SCREWS {int(cmd['screws_done'])}/6 | CONTACTS: BY ROBOT {by_robot} BY HIM {by_him} | "
                  f"VIEW BLOCKED BY ROBOT {views}", (0.85, 0.95, 0.85))]
        if alert:
            lines.append((alert, (1.0, 0.35, 0.3)))
        self.pub["caption"].publish(caption_image(lines, t))
        self.pub["image"].publish(rgb_to_msg(self.colour.rgb(d), t))
        stamp_t = stamp(t)
        for g, shaped in self.worker_shapes:
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, d.geom_xmat[g])
            place_markers(shaped, d.geom_xpos[g].copy(), quat, stamp_t)
        self.pub["worker"].publish(self.worker_array)
        tfm = TFMessage()
        for child, body in (("robot_mount", pose.ROBOT_PREFIX + "base"), ("eyes", "eyes")):
            b = m.body(body).id
            tr = TransformStamped()
            tr.header.stamp, tr.header.frame_id, tr.child_frame_id = stamp_t, "world", child
            tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z = map(float, d.xpos[b])
            w, x, y, z = map(float, d.xquat[b])
            tr.transform.rotation.x, tr.transform.rotation.y, tr.transform.rotation.z, tr.transform.rotation.w = x, y, z, w
            tfm.transforms.append(tr)
        self.pub["tf"].publish(tfm)
        self.poses.append((t, d.xpos.copy(), d.xquat.copy(), cmd["state"], screw, seg.name, int(cmd["screws_done"]),
                           len(self.r1.contacts), int(blocked), verdict))
        rclpy.spin_once(self, timeout_sec=0.0)

    # ------------------------------------------------------------ the run
    def run(self):
        args = self.args
        duration = self.pb.duration if args.seconds is None else min(args.seconds, self.pb.duration)
        self.wait_planner()
        t0 = time.time()
        stats = script.run(self.scene, self.d, self.pb, on_frame=self.on_frame, fps=FPS, on_step=self.on_step,
                           after_step=self.after_step, duration=duration)
        wall = time.time() - t0
        st = self.state
        # the last tick tells the planner the run is over, so it writes its own summary
        self.pub["tick"].publish(to_json({"tick": st["tick"], "t": duration, "frame": st["frame_index"], "end": True,
                                          "signal": st["signal"], "q": self.q, "qd": self.qd}))
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.05)
        out = args.out
        os.makedirs(out, exist_ok=True)
        cmd = st["cmd"]
        result = {
            "stage": 1, "variant": "B4", "seed": args.seed, "date": "2026-09-19",
            "episode_s": round(duration, 2), "worker_cycle_s": round(self.pb.duration, 2),
            "wall_time_s": round(wall, 1), "real_time_factor": round(duration / wall, 3),
            "ticks": st["tick"], "command_timeouts": st["timeouts"],
            "wait_for_command_ms_median": round(float(np.median(st["waits_s"])) * 1000, 2) if st["waits_s"] else None,
            "wait_for_command_ms_p99": round(float(np.percentile(st["waits_s"], 99)) * 1000, 2) if st["waits_s"] else None,
            "screws_driven": int(cmd["screws_done"]),
            "held_by_rules_s": round(st["held_s"], 2), "stepped_back_s": round(st["yield_s"], 2),
            "asked_worker_to_move": st["asks"], "blind_cells_used": int(len(self.hidden)),
            "R1": self.r1.results(), "R2": self.r2.results(),
            "worker": {k: stats[k] for k in ("peak_hand_speed_m_s", "peak_hand_accel_m_s2")},
            "fps": FPS,
        }
        path = os.path.join(out, f"cell_B4_seed{args.seed}.yaml")
        yaml.safe_dump(result, open(path, "w"), sort_keys=False)
        p = self.poses
        np.savez_compressed(os.path.join(out, f"run_B4_seed{args.seed}_poses.npz"),
                            t=np.array([x[0] for x in p]), xpos=np.array([x[1] for x in p]),
                            xquat=np.array([x[2] for x in p]),
                            body_names=np.array([self.m.body(i).name for i in range(self.m.nbody)]),
                            robot_state=np.array([x[3] for x in p]), screw=np.array([x[4] for x in p]),
                            worker_step=np.array([x[5] for x in p]), screws_done=np.array([x[6] for x in p]),
                            contacts=np.array([x[7] for x in p]), blocked=np.array([x[8] for x in p]),
                            verdict=np.array([x[9] for x in p]), fps=FPS, variant="B4", seed=args.seed)
        r1, r2 = result["R1"], result["R2"]
        print(f"CELL: {duration:.1f} s of sim in {wall:.0f} s wall (RTF {duration / wall:.2f}), {st['tick']} ticks, "
              f"{st['timeouts']} timeouts, wait for command {result['wait_for_command_ms_median']} ms median / "
              f"{result['wait_for_command_ms_p99']} p99", flush=True)
        print(f"screws {cmd['screws_done']}/6, held by the rules {st['held_s']:.2f} s, asked him {len(st['asks'])} times",
              flush=True)
        print(f"R1: {r1['contacts']} contacts {r1.get('contacts_by_who_moved_in', '')}, deepest "
              f"{r1['worst_contact_depth_mm']} mm, judge min distance {r1.get('min_distance_mm')} mm", flush=True)
        print(f"R2: {r2.get('blocked_frames')}/{r2.get('frames')} frames ({r2.get('blocked_frames_pct')} %), "
              f"events by who moved in {r2.get('blocked_events_by_who_moved_in')}", flush=True)
        print(f"wrote {path}", flush=True)
        self.eyes.close()
        self.colour.close()
        self.r2.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--out", default="results/ros2/stage1")
    ap.add_argument("--timeout", type=float, default=30.0, help="seconds to wait for a command before holding")
    args = ap.parse_args(remove_ros_args(sys.argv)[1:])   # launch adds --ros-args
    rclpy.init()
    cell = Cell(args)
    try:
        cell.run()
    finally:
        cell.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
