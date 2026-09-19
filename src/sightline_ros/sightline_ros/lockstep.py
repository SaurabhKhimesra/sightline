"""What the two live nodes share: topic names, the JSON messages, image conversion,
sim time stamps and the caption strip.

The cell (ros2/sim_node.py) and the planner (ros2/planner_node.py) run in lockstep
on sim time: every 10 ms the cell publishes the joint state with a tick, waits for
the command that answers that tick, and only then steps on. Every 40 ms it publishes
a depth picture first, and the tick after it names that picture, so the planner
answers no tick before it has looked at the picture the cell sent. Nothing depends
on wall time, so the run is the same run at any speed the machine manages.
"""
import json
import sys

# the system's Python packages (rclpy's message modules import "em" from there) go
# last: ahead of the venv they shadow its protobuf
sys.path[:] = [p for p in sys.path if "dist-packages" not in p] + [p for p in sys.path if "dist-packages" in p]

import numpy as np  # noqa: E402
from builtin_interfaces.msg import Time as TimeMsg  # noqa: E402
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy  # noqa: E402
from sensor_msgs.msg import Image, PointCloud2, PointField  # noqa: E402
from std_msgs.msg import String  # noqa: E402

TOPICS = {
    # what the cell knows once, sent once and kept for a late subscriber
    "calibration": "/sightline/cell/calibration",   # where the planner may believe the eyes camera is
    "background": "/eyes/background",               # the empty station's depth, taken at commissioning
    "blind": "/sightline/cell/blind_cells",         # cells the furniture hides from the camera
    "cell": "/sightline/cell/taught",               # the jig, the holes, the feeder, the park pose
    # every 10 ms
    "joints": "/joint_states",
    "tick": "/sightline/tick",
    "cmd": "/sightline/cmd",
    "clock": "/clock",
    # every 40 ms
    "depth": "/eyes/depth",
    "frame": "/sightline/frame",
    "image": "/eyes/image",
    "judge": "/sightline/judge",
    "worker": "/sightline/worker",
    "station": "/sightline/station",
    "caption": "/sightline/caption_image",
    # the planner's view of things, for rviz
    "grid": "/sightline/grid",
    "arm": "/sightline/arm_points",
    "target": "/sightline/target",
    "planner_caption": "/sightline/planner_caption",
    "ready": "/sightline/planner/ready",           # the planner has built its world and answers ticks
}
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)


def stamp(t: float) -> TimeMsg:
    sec = int(t)
    return TimeMsg(sec=sec, nanosec=int(round((t - sec) * 1e9)))


def _plain(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON: {type(o)}")


def to_json(obj) -> String:
    return String(data=json.dumps(obj, default=_plain))


def from_json(msg: String) -> dict:
    return json.loads(msg.data)


def depth_to_msg(depth: np.ndarray, t: float, frame_id: str = "eyes") -> Image:
    depth = np.ascontiguousarray(depth, dtype=np.float32)
    msg = Image()
    msg.header.stamp, msg.header.frame_id = stamp(t), frame_id
    msg.height, msg.width = depth.shape
    msg.encoding, msg.is_bigendian, msg.step = "32FC1", 0, depth.shape[1] * 4
    msg.data = depth.tobytes()
    return msg


def msg_to_depth(msg: Image) -> np.ndarray:
    assert msg.encoding == "32FC1", msg.encoding
    return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()


def rgb_to_msg(rgb: np.ndarray, t: float, frame_id: str = "eyes") -> Image:
    rgb = np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8)
    msg = Image()
    msg.header.stamp, msg.header.frame_id = stamp(t), frame_id
    msg.height, msg.width = rgb.shape[:2]
    msg.encoding, msg.is_bigendian, msg.step = "rgb8", 0, rgb.shape[1] * 3
    msg.data = rgb.tobytes()
    return msg


def caption_image(lines, t: float, width: int = 960, px: int = 2) -> Image:
    """Lines of (text, rgb) drawn on a dark strip, for an rviz Image panel."""
    from sightline_sim import textures
    height = 8 + 12 * px * len(lines)
    strip = np.full((height, width, 3), 0.10, np.float32)
    for n, (text, rgb) in enumerate(lines):
        textures.draw_text(strip, text, 12, 8 + n * 12 * px, rgb, px=px)
    return rgb_to_msg(textures.to_uint8(strip), t, "caption")


def cloud_msg(points: np.ndarray, rgb, t: float, frame_id: str = "world") -> PointCloud2:
    """Coloured points for rviz, straight from numpy: building a marker point by point
    in Python cost more than the picture itself."""
    pts = np.asarray(points, np.float32).reshape(-1, 3)
    r, g, b = (int(round(c * 255)) for c in rgb)
    colour = np.uint32((r << 16) | (g << 8) | b)
    data = np.zeros(len(pts), dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32), ("rgb", np.uint32)])
    data["x"], data["y"], data["z"], data["rgb"] = pts[:, 0], pts[:, 1], pts[:, 2], colour
    msg = PointCloud2()
    msg.header.stamp, msg.header.frame_id = stamp(t), frame_id
    msg.height, msg.width = 1, len(pts)
    msg.fields = [PointField(name=n, offset=o, datatype=PointField.FLOAT32 if n != "rgb" else PointField.UINT32, count=1)
                  for n, o in (("x", 0), ("y", 4), ("z", 8), ("rgb", 12))]
    msg.is_bigendian, msg.point_step, msg.row_step, msg.is_dense = False, 16, 16 * len(pts), True
    msg.data = data.tobytes()
    return msg
