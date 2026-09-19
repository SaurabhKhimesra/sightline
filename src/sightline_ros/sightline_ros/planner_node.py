"""The planner as a ROS 2 node: B4 on the camera's grid, fed by topics, in lockstep
with the cell node (ros2/sim_node.py).

    python ros2/planner_node.py [--seed 0] [--out results/ros2/stage1]

It waits for what the cell publishes once (the calibration it may believe, the
empty station's depth, the blind cells, the taught points), then answers every
/sightline/tick with a /sightline/cmd (JSON: the tick, six joint velocities, its
state, the guard's verdict). A tick that names a picture it has not yet processed
waits for that picture. Every picture rebuilds the grid; every tick runs the task,
the motion layer and the guard, exactly as scripts/gate5_planner.py does for B4.
For rviz it publishes the grid's off-limits cells, the 63 points of the arm it
checks, the hole it is going for, and a caption of its own state.

It imports nothing from the simulation: what it knows of the cell arrives on
topics, and its model of its own arm is the robot-only model (station.robot_only_model,
the one thing it takes from the station module, as the gate 5 runner does).
"""
import argparse
import os
import sys
import time

from . import lockstep  # noqa: E402,F401  (fixes the import path before rclpy)
from .lockstep import TOPICS, LATCHED, stamp, to_json, from_json, msg_to_depth, caption_image, cloud_msg  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402
import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.utilities import remove_ros_args  # noqa: E402
from sensor_msgs.msg import Image, PointCloud2  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from geometry_msgs.msg import Point  # noqa: E402
from visualization_msgs.msg import Marker, MarkerArray  # noqa: E402

from sightline_planner.settings import HAND_REACH_M  # noqa: E402
from sightline_planner import B4, SensorFrame, Taught, ToolKinematics  # noqa: E402
from sightline_planner.motion import MotionLayer, Obstacles  # noqa: E402
from sightline_planner.perceive import Calibration, KnownWorld, Perception  # noqa: E402
from sightline_planner.viewgrid import ViewGrid  # noqa: E402
from sightline_sim.station import robot_only_model  # noqa: E402


class Planner(Node):
    def __init__(self, args):
        super().__init__("sightline_planner")
        self.args = args
        self.knowledge: dict = {}
        self.ready = False
        self.create_subscription(String, TOPICS["calibration"], lambda m: self.learn("calibration", from_json(m)), LATCHED)
        self.create_subscription(Image, TOPICS["background"], lambda m: self.learn("background", msg_to_depth(m)), LATCHED)
        self.create_subscription(String, TOPICS["blind"], lambda m: self.learn("blind", from_json(m)), LATCHED)
        self.create_subscription(String, TOPICS["cell"], lambda m: self.learn("cell", from_json(m)), LATCHED)
        self.create_subscription(Image, TOPICS["depth"], self.on_depth, 5)
        self.create_subscription(String, TOPICS["frame"], self.on_frame, 10)
        self.create_subscription(String, TOPICS["tick"], self.on_tick, 50)
        self.pub_cmd = self.create_publisher(String, TOPICS["cmd"], 50)
        self.pub_grid = {name: self.create_publisher(PointCloud2, TOPICS["grid"] + "/" + name, 5)
                         for name in ("seen", "remembered", "behind")}
        self.pub_ready = self.create_publisher(String, TOPICS["ready"], LATCHED)
        self.pub_arm = self.create_publisher(Marker, TOPICS["arm"], 5)
        self.pub_target = self.create_publisher(Marker, TOPICS["target"], 5)
        self.pub_caption = self.create_publisher(Image, TOPICS["planner_caption"], 5)
        self.depths: dict = {}          # by picture time in ms, until its frame message arrives
        self.frames: dict = {}
        self.last_frame = 0
        self.pending = None
        self.stats = {"ticks": 0, "frames": 0, "solve_ms": [], "guard_ms": [], "frame_ms": [], "verdicts": {},
                      "perceive_ms": [], "grid_ms": [], "blind_ms": [], "rviz_ms": []}
        self.done = False

    # ------------------------------------------------------------ what the cell knows
    def learn(self, key, value):
        if key in self.knowledge:
            return
        self.knowledge[key] = value
        self.get_logger().info(f"received the cell's {key}")
        if not self.ready and all(k in self.knowledge for k in ("calibration", "background", "blind", "cell")):
            self.setup()

    def setup(self):
        k = self.knowledge
        cal, cell = k["calibration"], k["cell"]
        self.fps = float(cell["fps"])
        self.calib = Calibration(pos=np.array(cal["pos"], float), R=np.array(cal["R"], float),
                                 width=int(cal["width"]), height=int(cal["height"]), fovy_deg=float(cal["fovy_deg"]))
        self.kin = ToolKinematics(robot_only_model())
        self.taught = Taught(part_pose=tuple(cell["part_pose"]), rail_holes=[tuple(h) for h in cell["rail_holes"]],
                             cover_holes=[tuple(h) for h in cell["cover_holes"]], feeder_pick=tuple(cell["feeder_pick"]))
        zw = float(cell["worktop_z"])
        self.park = np.array(cell["park"], float)
        self.planes = [(np.array([0.0, 0.0, zw]), np.array([0.0, 0.0, 1.0]))]
        self.known = KnownWorld(robot_only_model(), self.calib)
        self.grid = ViewGrid(self.calib, speed_mode="measured")
        self.b4 = B4.build(self.kin, self.taught, park_tip=(-0.45, 0.33, zw + 0.30), home_q=self.park,
                           eye=np.asarray(self.calib.pos, float), grid=self.grid, planes=self.planes)
        self.b4.motion = MotionLayer(self.kin, use_r1=False, use_r2=False)
        self.per = Perception(self.calib, self.kin, self.taught, known=self.known)
        self.per.set_background(k["background"])
        self.grid.set_background(k["background"])
        self.hidden = np.array(k["blind"]["cells"], float).reshape(-1, 3)
        self.ready = True
        self.get_logger().info(f"planner ready: {len(self.hidden)} blind cells, worktop at {zw:.3f} m")
        self.pub_ready.publish(to_json({"ready": True, "blind_cells": int(len(self.hidden))}))
        if self.pending is not None:
            tick, self.pending = self.pending, None
            self.on_tick_ready(tick)

    def live_blind(self) -> np.ndarray:
        """A blind cell counts only if a seen part of him is close enough to have put a
        hand in it (the gate 5 runner's rule)."""
        person = self.grid.seen_points()
        if not (person.shape[0] and self.hidden.shape[0]):
            return np.zeros((0, 3))
        near = np.zeros(len(self.hidden), dtype=bool)
        for chunk in range(0, len(person), 400):
            block = person[chunk:chunk + 400]
            gap = np.linalg.norm(self.hidden[:, None, :] - block[None, :, :], axis=2).min(axis=1)
            near |= gap < HAND_REACH_M
        return self.hidden[near]

    # ------------------------------------------------------------ pictures
    @staticmethod
    def key(t: float) -> int:
        return int(round(t * 1000))

    def on_depth(self, msg: Image):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.depths[self.key(t)] = msg_to_depth(msg)
        self.try_frame(self.key(t))

    def on_frame(self, msg: String):
        info = from_json(msg)
        self.frames[self.key(info["t"])] = info
        self.try_frame(self.key(info["t"]))

    def try_frame(self, key: int):
        if key in self.depths and key in self.frames and self.ready:
            self.process_frame(self.frames.pop(key), self.depths.pop(key))

    def process_frame(self, info: dict, depth: np.ndarray):
        t0 = time.perf_counter()
        b4, per, grid = self.b4, self.per, self.grid
        t, index = float(info["t"]), int(info["index"])
        q, seen_q = np.array(info["q"], float), np.array(info["seen_q"], float)
        model = per.update(depth, q, t, b4.parts_in_jig(info["signal"], per.last_box is not None), q_at_capture=seen_q)
        t1 = time.perf_counter()
        b4.see(model.person_px, model.person_z, float(info["taken_at"]), seen_q)
        t2 = time.perf_counter()
        b4.guard.blind = self.live_blind()
        t3 = time.perf_counter()
        self.last_frame = index
        self.stats["frames"] += 1
        self.stats["frame_ms"].append((t3 - t0) * 1000)
        self.stats["perceive_ms"].append((t1 - t0) * 1000)
        self.stats["grid_ms"].append((t2 - t1) * 1000)
        self.stats["blind_ms"].append((t3 - t2) * 1000)
        self.publish_grid(t)
        self.stats["rviz_ms"].append((time.perf_counter() - t3) * 1000)
        if self.pending is not None and int(self.pending["frame"]) <= index:
            tick, self.pending = self.pending, None
            self.on_tick_ready(tick)

    # ------------------------------------------------------------ ticks
    def on_tick(self, msg: String):
        tick = from_json(msg)
        if tick.get("end"):
            self.finish(tick)
            return
        if not self.ready or int(tick["frame"]) > self.last_frame:
            self.pending = tick               # the picture it names has not arrived yet
            return
        self.on_tick_ready(tick)

    def on_tick_ready(self, tick: dict):
        b4 = self.b4
        t = float(tick["t"])
        frame = SensorFrame(t=t, q=np.array(tick["q"], float), qd=np.array(tick["qd"], float), jig_signal=tick["signal"])
        wanted = b4._task(frame)
        rep = b4.motion.solve_joint(frame.q, wanted, Obstacles(planes=self.planes))
        t0 = time.perf_counter()
        qd = b4.safe_command(frame, rep.qd)
        self.stats["guard_ms"].append((time.perf_counter() - t0) * 1000)
        b4.motion.last = qd                    # the next ramp starts from what was sent
        verdict = b4.guard.last
        self.stats["verdicts"][verdict] = self.stats["verdicts"].get(verdict, 0) + 1
        self.stats["solve_ms"].append(rep.solve_ms)
        self.stats["ticks"] += 1
        screw = b4.screws[b4.index].name if b4.index < len(b4.screws) else ""
        self.pub_cmd.publish(to_json({
            "tick": int(tick["tick"]), "qd": qd, "state": b4.state, "verdict": verdict,
            "want_speed": float(np.linalg.norm(wanted)), "screw": screw,
            "screws_done": sum(1 for s in b4.screws if s.driven), "yield_since": float(b4.yield_since),
            "solve_ms": rep.solve_ms}))
        if int(tick["tick"]) % 5 == 0:
            self.publish_arm(frame.q, t, screw)

    # ------------------------------------------------------------ for rviz
    def publish_grid(self, t: float):
        """The grid's cells as three point clouds: where the camera sees him (red), the
        cells its own arm hides and it remembers (orange), and the far end of every
        cell, 15 cm behind him (dark red): the arm keeps off the whole stretch."""
        grid = self.grid
        if grid.rows.size:
            world = lambda z: grid._points(grid.rows, grid.cols, z).astype(float) @ grid.calib.R.T + grid.calib.pos  # noqa: E731
            near, far = world(grid.near), world(grid.far)
            seen = ~grid.hidden
            clouds = {"seen": (near[seen], (0.92, 0.16, 0.12)), "remembered": (near[~seen], (1.0, 0.59, 0.0)),
                      "behind": (far, (0.45, 0.05, 0.05))}
        else:
            clouds = {name: (np.zeros((0, 3)), (0, 0, 0)) for name in ("seen", "remembered", "behind")}
        for name, (pts, rgb) in clouds.items():
            self.pub_grid[name].publish(cloud_msg(pts, rgb, t))
        b4 = self.b4
        lines = [(f"PLANNER B4 ON ROS 2   PICTURE {self.stats['frames']}   T {t:5.1f} S", (0.95, 0.95, 0.92)),
                 (f"STATE {b4.state.upper()}   GUARD {b4.guard.last.upper() or 'GO'}   OFF LIMITS CELLS {int(grid.rows.size)} "
                  f"({int(grid.hidden.sum()) if grid.rows.size else 0} REMEMBERED)   BLIND CELLS LIVE {len(b4.guard.blind)}",
                  (0.55, 0.85, 1.0)),
                 (f"PICTURE {np.median(self.stats['frame_ms'][-25:]):.0f} MS   GUARD {np.median(self.stats['guard_ms'][-100:]) if self.stats['guard_ms'] else 0:.1f} MS   "
                  f"QP {np.median(self.stats['solve_ms'][-100:]) if self.stats['solve_ms'] else 0:.1f} MS", (0.85, 0.95, 0.85))]
        self.pub_caption.publish(caption_image(lines, t))

    def publish_arm(self, q: np.ndarray, t: float, screw: str):
        mk = Marker()
        mk.header.frame_id, mk.header.stamp = "world", stamp(t)
        mk.ns, mk.id, mk.type, mk.action = "arm", 0, Marker.SPHERE_LIST, Marker.ADD
        mk.pose.orientation.w = 1.0
        mk.scale.x = mk.scale.y = mk.scale.z = 0.03
        mk.color.r, mk.color.g, mk.color.b, mk.color.a = 0.2, 0.95, 0.4, 0.9
        for p in self.b4.guard.points(q):
            mk.points.append(Point(x=float(p[0]), y=float(p[1]), z=float(p[2])))
        self.pub_arm.publish(mk)
        tg = Marker()
        tg.header.frame_id, tg.header.stamp = "world", stamp(t)
        tg.ns, tg.id, tg.type = "target", 0, Marker.SPHERE
        tg.action = Marker.ADD if screw and self.b4.state in ("to_hole", "servo", "descend", "drive", "retract") else Marker.DELETE
        hole = None
        for s in self.b4.screws:
            if s.name == screw:
                hole = s
        if tg.action == Marker.ADD and hole is not None:
            tg.pose.position.x, tg.pose.position.y, tg.pose.position.z = map(float, hole.hole)
        tg.pose.orientation.w = 1.0
        tg.scale.x = tg.scale.y = tg.scale.z = 0.05
        tg.color.r, tg.color.g, tg.color.b, tg.color.a = 0.3, 0.6, 1.0, 0.9
        self.pub_target.publish(tg)

    # ------------------------------------------------------------ the end
    def finish(self, tick: dict):
        if self.done:
            return
        self.done = True
        summary = self.b4.summary() if self.ready else {}
        st = self.stats
        result = {
            "stage": 1, "variant": "B4", "seed": self.args.seed, "date": "2026-09-19",
            "ticks_answered": st["ticks"], "pictures": st["frames"],
            "picture_ms_median": round(float(np.median(st["frame_ms"])), 1) if st["frame_ms"] else None,
            "picture_ms_max": round(float(max(st["frame_ms"])), 1) if st["frame_ms"] else None,
            "picture_parts_ms_median": {k: round(float(np.median(st[k + "_ms"])), 1) for k in ("perceive", "grid", "blind", "rviz")
                                        if st[k + "_ms"]},
            "guard_ms_median": round(float(np.median(st["guard_ms"])), 2) if st["guard_ms"] else None,
            "guard_ms_p99": round(float(np.percentile(st["guard_ms"], 99)), 2) if st["guard_ms"] else None,
            "qp_ms_median": round(float(np.median(st["solve_ms"])), 2) if st["solve_ms"] else None,
            "guard_verdicts": dict(sorted(st["verdicts"].items())),
            "planner_summary": summary,
        }
        os.makedirs(self.args.out, exist_ok=True)
        path = os.path.join(self.args.out, f"planner_B4_seed{self.args.seed}.yaml")
        yaml.safe_dump(result, open(path, "w"), sort_keys=False)
        print(f"PLANNER: {st['ticks']} ticks, {st['frames']} pictures; picture {result['picture_ms_median']} ms median / "
              f"{result['picture_ms_max']} max {result['picture_parts_ms_median']}, guard {result['guard_ms_median']} ms "
              f"median / {result['guard_ms_p99']} p99, screws {summary.get('screws_driven')}, guard {result['guard_verdicts']}",
              flush=True)
        print(f"wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/ros2/stage1")
    args = ap.parse_args(remove_ros_args(sys.argv)[1:])   # launch adds --ros-args
    rclpy.init()
    node = Planner(args)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
