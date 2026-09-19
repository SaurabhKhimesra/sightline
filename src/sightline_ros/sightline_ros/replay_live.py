"""Play a recorded run in real time into Gazebo and ROS 2 together, for the live
desktop recording: Gazebo's GUI shows the cell, rviz shows the UR5e model on the
recorded joint angles, the worker as the judge's capsules, the station's shapes, the
eyes camera's picture (bridged from Gazebo) and the judge's words.

    python ros2/replay_live.py results/ros2/run_B4_seed0_poses.npz [--mjcf results/ros2/scene/sightline_cell.xml]
        [--speed 1.0] [--start 0] [--end 999] [--no-gazebo] [--world sightline_replay]

Every frame of the recording (25 a second) sets the pose of every body model in
Gazebo (set_pose_vector, then a step of one frame's sim time) and publishes to ROS
/joint_states, /tf (world to robot_mount, world to the eyes camera), /sightline/worker
(MarkerArray), /sightline/station (MarkerArray, once), /sightline/caption_image (the judge's
words drawn as a picture) and /sightline/words (String). Gazebo and rviz render; nothing here decides anything. The joint
angles come from the recorded body orientations, exactly (tests in tests/test_ros2.py).
"""
import argparse
import os
import sys
import time

# the system's Python packages (rclpy's message modules import "em" from there) go
# last: ahead of the venv they shadow its protobuf, which the Gazebo messages need
sys.path[:] = [p for p in sys.path if "dist-packages" not in p] + [p for p in sys.path if "dist-packages" in p]

import numpy as np  # noqa: E402
import mujoco  # noqa: E402
import gz.math  # noqa: E402,F401  (registers the shared types the bindings below need)
from gz.transport import Node as GzNode  # noqa: E402
from gz.msgs.pose_v_pb2 import Pose_V  # noqa: E402
from gz.msgs.boolean_pb2 import Boolean  # noqa: E402
from gz.msgs.world_control_pb2 import WorldControl  # noqa: E402
from gz.msgs.empty_pb2 import Empty  # noqa: E402
from gz.msgs.scene_pb2 import Scene  # noqa: E402
import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.utilities import remove_ros_args  # noqa: E402
from rclpy.qos import QoSProfile, DurabilityPolicy  # noqa: E402
from sensor_msgs.msg import JointState, Image  # noqa: E402
from geometry_msgs.msg import TransformStamped  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402
from visualization_msgs.msg import Marker, MarkerArray  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from sightline_sim.episode import ROBOT_WORDS  # noqa: E402
from sightline_sim import textures  # noqa: E402

JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint",
          "wrist_3_joint"]
WORKER_COLOR = (1.0, 0.62, 0.2, 0.95)
CAPTION_W, CAPTION_H, CAPTION_PX = 960, 112, 2      # the judge's words as a picture, for an rviz Image panel
TITLES = {"B0": "B0: IGNORES HIM", "B2": "B2: KEEPS 10 CM FROM HIM", "B3": "B3: 10 CM AND OUT OF HIS VIEW",
          "B4": "B4: PLANS ON THE CAMERA GRID"}


def joint_angles(m: mujoco.MjModel, jids, xquat: np.ndarray, table=None) -> np.ndarray:
    """The angle of each hinge joint, from the recorded orientations of its two bodies:
    what the joint added to the child's zero orientation in the parent frame, taken
    about the joint axis."""
    table = table or joint_table(m, jids)
    q = np.zeros(len(jids))
    inv_p, rel, turn, vel = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(3)
    for k, (b, parent, inv0, axis) in enumerate(table):
        mujoco.mju_negQuat(inv_p, xquat[parent])
        mujoco.mju_mulQuat(rel, inv_p, xquat[b])
        mujoco.mju_mulQuat(turn, inv0, rel)
        mujoco.mju_quat2Vel(vel, turn, 1.0)
        q[k] = float(vel @ axis)
    return q


def joint_table(m: mujoco.MjModel, jids):
    """Per joint: child body, parent body, the inverse of the child's zero orientation
    in the parent frame, the axis. Read once; reading model fields per frame is slow."""
    table = []
    for j in jids:
        b = int(m.jnt_bodyid[j])
        inv0 = np.zeros(4)
        mujoco.mju_negQuat(inv0, m.body_quat[b].copy())
        table.append((b, int(m.body_parentid[b]), inv0, m.jnt_axis[j].copy()))
    return table


def set_pose(msg_pose, pos, quat_wxyz):
    msg_pose.position.x, msg_pose.position.y, msg_pose.position.z = map(float, pos)
    w, x, y, z = map(float, quat_wxyz)
    msg_pose.orientation.x, msg_pose.orientation.y, msg_pose.orientation.z, msg_pose.orientation.w = x, y, z, w


def geom_world(m, g, body_pos, body_quat):
    """World pose of a geom from its body's recorded pose."""
    pos = np.zeros(3)
    mujoco.mju_rotVecQuat(pos, m.geom_pos[g], body_quat)
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, body_quat, m.geom_quat[g])
    return body_pos + pos, quat


def shape_markers(m, g, rgba, ns, first_id):
    """rviz markers for one MuJoCo primitive, with the part each one plays: a capsule
    is a cylinder and two spheres, one at each end along its axis. Poses are set per
    frame by place_markers; building messages every frame is what costs time."""
    kind, size = int(m.geom_type[g]), m.geom_size[g].copy()

    def marker(mtype, scale, mid, along):
        mk = Marker()
        mk.header.frame_id = "world"
        mk.ns, mk.id, mk.type, mk.action = ns, mid, mtype, Marker.ADD
        mk.scale.x, mk.scale.y, mk.scale.z = map(float, scale)
        mk.color.r, mk.color.g, mk.color.b, mk.color.a = map(float, rgba)
        return mk, along

    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return [marker(Marker.SPHERE, (2 * size[0],) * 3, first_id, 0.0)]
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        r, h = float(size[0]), float(size[1])
        return [marker(Marker.CYLINDER, (2 * r, 2 * r, 2 * h), first_id, 0.0),
                marker(Marker.SPHERE, (2 * r,) * 3, first_id + 1, h),
                marker(Marker.SPHERE, (2 * r,) * 3, first_id + 2, -h)]
    if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return [marker(Marker.SPHERE, 2 * size, first_id, 0.0)]
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return [marker(Marker.CYLINDER, (2 * size[0], 2 * size[0], 2 * size[1]), first_id, 0.0)]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return [marker(Marker.CUBE, 2 * size, first_id, 0.0)]
    return []


def place_markers(shaped, pos, quat, stamp):
    """Set the world pose of one primitive's markers; `along` is the offset of each
    along the primitive's own z axis."""
    axis = np.zeros(3)
    mujoco.mju_rotVecQuat(axis, np.array([0.0, 0.0, 1.0]), quat)
    for mk, along in shaped:
        mk.header.stamp = stamp
        set_pose(mk.pose, pos + along * axis if along else pos, quat)


class LiveReplay(Node):
    def __init__(self, args):
        super().__init__("sightline_replay")
        self.args = args
        # every array read once: indexing the npz file itself decompresses the whole
        # array again at every access
        with np.load(args.poses, allow_pickle=True) as rec:
            self.rec = {key: rec[key] for key in rec.files}
        self.names = [str(n) for n in self.rec["body_names"]]
        self.t = self.rec["t"]
        self.m = mujoco.MjModel.from_xml_path(args.mjcf)
        m = self.m
        self.body_of = {name.replace("/", "__"): b for b, name in enumerate(self.names)}
        self.jids = [m.joint("ur5e__" + j).id for j in JOINTS]
        self.joint_table = joint_table(m, self.jids)
        self.base = self.body_of["ur5e__base"]
        self.eyes = self.body_of["eyes"]
        self.eyes_cam = m.camera("eyes_cam").id
        worker_bodies = {b for b in range(m.nbody) if m.body(b).name.startswith("worker")}
        # geom types as plain ints: an enum inside a tuple never matches a numpy int here
        kinds = [int(k) for k in m.geom_type]
        self.worker_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in worker_bodies and m.geom_group[g] == 3
                             and kinds[g] != int(mujoco.mjtGeom.mjGEOM_MESH)]
        robot_bodies = {b for b in range(m.nbody) if m.body(b).name.startswith("ur5e")}
        self.station_geoms = [g for g in range(m.ngeom)
                              if m.geom_bodyid[g] not in worker_bodies | robot_bodies and m.geom_group[g] == 0
                              and kinds[g] in (int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_CYLINDER))]
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_joints = self.create_publisher(JointState, "/joint_states", 10)
        self.pub_tf = self.create_publisher(TFMessage, "/tf", 10)
        self.pub_worker = self.create_publisher(MarkerArray, "/sightline/worker", 10)
        self.pub_station = self.create_publisher(MarkerArray, "/sightline/station", latched)
        self.pub_caption = self.create_publisher(Image, "/sightline/caption_image", 10)
        self.pub_words = self.create_publisher(String, "/sightline/words", 10)
        self.timing = {"gazebo": [], "ros": [], "joints_tf": [], "worker": [], "caption": []}   # where a frame's time goes
        self.gz = None
        if not args.no_gazebo:
            self.gz = GzNode()
            ok, scene = self.gz.request(f"/world/{args.world}/scene/info", Empty(), Empty, Scene, 5000)
            assert ok, "the Gazebo world did not answer: is the server up on this GZ_PARTITION?"
            models = {model.name for model in scene.model}
            self.posed = [(b, name.replace("/", "__")) for b, name in enumerate(self.names)
                          if name.replace("/", "__") in models]
            self.get_logger().info(f"{len(self.posed)} recorded bodies are models in Gazebo")
        # the messages, built once: the worker's shapes get their poses every frame
        self.worker_shapes = []
        mid = 0
        for g in self.worker_geoms:
            shaped = shape_markers(m, g, WORKER_COLOR, "worker", mid)
            mid += len(shaped)
            self.worker_shapes.append((g, int(m.geom_bodyid[g]), shaped))
        self.worker_array = MarkerArray()
        self.worker_array.markers = [mk for _, _, shaped in self.worker_shapes for mk, _ in shaped]
        self.station_array = self.build_station()
        self.publish_station()

    def build_station(self) -> MarkerArray:
        arr = MarkerArray()
        mid = 0
        d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, d)
        stamp = self.get_clock().now().to_msg()
        for g in self.station_geoms:
            mat = self.m.geom_matid[g]
            rgba = self.m.mat_rgba[mat] if mat >= 0 else self.m.geom_rgba[g]
            shaped = shape_markers(self.m, g, (rgba[0], rgba[1], rgba[2], 0.55), "station", mid)
            mid += len(shaped)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, d.geom_xmat[g])
            place_markers(shaped, d.geom_xpos[g].copy(), quat, stamp)
            arr.markers.extend(mk for mk, _ in shaped)
        return arr

    def publish_station(self):
        # sent again every couple of seconds: rviz may open after the first one
        stamp = self.get_clock().now().to_msg()
        for mk in self.station_array.markers:
            mk.header.stamp = stamp
        self.pub_station.publish(self.station_array)

    def frame(self, i: int, sim_index: int | None = None):
        """Frame i of the recording into Gazebo and ROS; sim time runs to frame sim_index
        (i itself unless told otherwise, as during the lead, which holds frame 0 while
        the clock runs so a recorder on sim time sees the hold too)."""
        xpos, xquat = self.rec["xpos"][i], self.rec["xquat"][i]
        stamp = self.get_clock().now().to_msg()
        t_start = time.perf_counter()
        sim_index = i if sim_index is None else sim_index
        if self.gz is not None:
            msg = Pose_V()
            for b, name in self.posed:
                p = msg.pose.add()
                p.name = name
                p.position.x, p.position.y, p.position.z = map(float, xpos[b])
                w, x, y, z = map(float, xquat[b])
                p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z = w, x, y, z
            ok, rep = self.gz.request(f"/world/{self.args.world}/set_pose_vector", msg, Pose_V, Boolean, 1000)
            if not ok or not rep.data:
                self.get_logger().warning(f"frame {i}: set_pose_vector failed")
            ctrl = WorldControl()
            sim_ns = int(round((sim_index + 1) * 1e9 / float(self.rec["fps"])))
            ctrl.run_to_sim_time.sec, ctrl.run_to_sim_time.nsec = sim_ns // 1_000_000_000, sim_ns % 1_000_000_000
            ok, rep = self.gz.request(f"/world/{self.args.world}/control", ctrl, WorldControl, Boolean, 1000)
            if not ok or not rep.data:
                self.get_logger().warning(f"frame {i}: world control failed")
        self.timing["gazebo"].append(time.perf_counter() - t_start)
        t_ros = time.perf_counter()
        js = JointState()
        js.header.stamp = stamp
        js.name = JOINTS
        js.position = joint_angles(self.m, self.jids, xquat, self.joint_table).tolist()
        self.pub_joints.publish(js)
        tfm = TFMessage()
        for child, b in (("robot_mount", self.base), ("eyes", self.eyes)):
            tr = TransformStamped()
            tr.header.stamp, tr.header.frame_id, tr.child_frame_id = stamp, "world", child
            tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z = map(float, xpos[b])
            w, x, y, z = map(float, xquat[b])
            tr.transform.rotation.x, tr.transform.rotation.y, tr.transform.rotation.z, tr.transform.rotation.w = x, y, z, w
            tfm.transforms.append(tr)
        self.pub_tf.publish(tfm)
        t_a = time.perf_counter()
        for g, b, shaped in self.worker_shapes:
            pos, quat = geom_world(self.m, g, xpos[b], xquat[b])
            place_markers(shaped, pos, quat, stamp)
        self.pub_worker.publish(self.worker_array)
        t_b = time.perf_counter()
        self.publish_caption(i, stamp)
        t_c = time.perf_counter()
        if i % 50 == 0:
            self.publish_station()
        self.timing["ros"].append(time.perf_counter() - t_ros)
        self.timing["joints_tf"].append(t_a - t_ros)
        self.timing["worker"].append(t_b - t_a)
        self.timing["caption"].append(t_c - t_b)

    def publish_caption(self, i, stamp):
        rec = self.rec
        screw = str(rec["screw"][i])
        doing = ROBOT_WORDS.get(str(rec["robot_state"][i]), str(rec["robot_state"][i]).upper()).format(
            s=screw.replace("_", " SCREW ").upper())
        verdict = str(rec["verdict"][i])
        if verdict == "brake":
            doing += " | HOLDING: HE IS CLOSE"
        elif verdict.startswith("escape"):
            doing += " | BACKING AWAY FROM HIM"
        title = TITLES.get(str(rec["variant"]), str(rec["variant"]))
        lines = [(f"{title}   T {self.t[i]:5.1f} S", (1.0, 1.0, 0.95)),
                 (f"ROBOT: {doing}", (0.55, 0.85, 1.0)),
                 (f"WORKER: {str(rec['worker_step'][i]).upper()}", (1.0, 0.85, 0.45)),
                 (f"SCREWS {int(rec['screws_done'][i])}/6   CONTACTS {int(rec['contacts'][i])}   "
                  f"VIEW BLOCKED {int(rec['blocked'][i])} FRAMES", (0.85, 0.95, 0.85))]
        # drawn as a picture: rviz's view facing text spread its letters across the view;
        # sent only when the words change (a 320 KB picture every frame cost 10 ms)
        key = tuple(text for text, _ in lines)
        if key == getattr(self, "_caption_key", None):
            return
        self._caption_key = key
        strip = np.full((CAPTION_H, CAPTION_W, 3), 0.10, np.float32)
        for n, (text, rgb) in enumerate(lines):
            textures.draw_text(strip, text, 12, 8 + n * 12 * CAPTION_PX, rgb, px=CAPTION_PX)
        img = Image()
        img.header.frame_id, img.header.stamp = "world", stamp
        img.height, img.width, img.encoding, img.is_bigendian, img.step = CAPTION_H, CAPTION_W, "rgb8", 0, CAPTION_W * 3
        img.data = textures.to_uint8(strip).tobytes()
        self.pub_caption.publish(img)
        self.pub_words.publish(String(data="\n".join(text for text, _ in lines)))

    def run(self):
        args = self.args
        frames = np.nonzero((self.t >= args.start) & (self.t <= args.end))[0]
        period = 1.0 / (float(self.rec["fps"]) * args.speed)
        lead = int(round(args.lead * float(self.rec["fps"])))
        t0 = time.time()
        late = 0
        for k, i in enumerate([frames[0]] * lead + list(frames)):
            if k == 0:
                self.get_logger().info(f"first frame at wall {time.time():.3f}")   # a screen grab aligns on it
            self.frame(int(i), sim_index=k)
            due = t0 + (k + 1) * period
            now = time.time()
            if now > due + period:
                late += 1
            time.sleep(max(due - now, 0.0))
            if k % 250 == 0:
                self.get_logger().info(f"frame {k}/{len(frames)} run t={self.t[i]:.1f} s, late {late}")
        gz_ms = np.median(self.timing["gazebo"]) * 1000 if self.timing["gazebo"] else 0.0
        ros_ms = np.median(self.timing["ros"]) * 1000 if self.timing["ros"] else 0.0
        parts = ", ".join(f"{k} {np.median(v) * 1000:.1f}" for k, v in self.timing.items() if v and k not in ("gazebo", "ros"))
        self.get_logger().info(f"done: {len(frames)} frames in {time.time() - t0:.1f} s wall, {late} late; per frame "
                               f"gazebo requests {gz_ms:.1f} ms, ros publishing {ros_ms:.1f} ms ({parts}) (medians)")
        print("REPLAY DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("poses")
    ap.add_argument("--mjcf", default="results/ros2/scene/sightline_cell.xml")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=1e9)
    ap.add_argument("--world", default="sightline_replay")
    ap.add_argument("--no-gazebo", action="store_true")
    ap.add_argument("--lead", type=float, default=2.0, help="seconds to hold the first frame, clock running")
    args = ap.parse_args(remove_ros_args(sys.argv)[1:])   # launch adds --ros-args
    rclpy.init()
    node = LiveReplay(args)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
