"""Play a recorded run into the Gazebo replay world, and save what its cameras see.

The world is started paused. For every frame of the recording (25 a second) the pose
of every body model is set through the world's set_pose_vector service, the world is
stepped by one camera period, and the image each named camera stamps with that sim
time is saved as <out>/<camera>_<frame>.png. Every saved frame is therefore the exact
state the judge saw at that instant; Gazebo renders, nothing here decides anything.

    gz sim -s --headless-rendering -v 1 results/ros2/gazebo/sightline_replay.sdf &
    python ros2/replay_gazebo.py results/ros2/run_B4_seed0_poses.npz results/ros2/gazebo/replay_frames \\
        --cameras cycle_hero eyes_cam [--start 0] [--end 999]

The frames are received by ros2/frame_grabber.py, started here as a second process.
Run everything with the same GZ_PARTITION so other Gazebo sessions on the machine
stay apart.
"""
import argparse
import os
import subprocess
import sys
import time

# the system's Python packages last: ahead of the venv they shadow its protobuf
sys.path[:] = [p for p in sys.path if "dist-packages" not in p] + [p for p in sys.path if "dist-packages" in p]

import numpy as np  # noqa: E402
import gz.math  # noqa: E402,F401  (registers the shared types the bindings below need)
from gz.transport import Node  # noqa: E402
from gz.msgs.pose_v_pb2 import Pose_V  # noqa: E402
from gz.msgs.boolean_pb2 import Boolean  # noqa: E402
from gz.msgs.world_control_pb2 import WorldControl  # noqa: E402
from gz.msgs.empty_pb2 import Empty  # noqa: E402
from gz.msgs.scene_pb2 import Scene  # noqa: E402

from .frame_grabber import READY_FILE, STOP_FILE  # noqa: E402



def stamped_frames(out: str, camera: str):
    """(stamp ns, path) of every frame of this camera not yet taken, oldest first."""
    found = []
    prefix = camera + "_t"
    for entry in os.scandir(out):
        if entry.name.startswith(prefix) and entry.name.endswith(".png") and not entry.name.endswith(".tmp.png"):
            digits = entry.name[len(prefix):-4]
            if digits.isdigit():
                found.append((int(digits), entry.path))
    return sorted(found)


def take_frame(out: str, camera: str, stamp_ns: int, period_ns: int, index: int, timeout: float) -> bool:
    """Rename the frame stamped within one period from stamp_ns to its index."""
    deadline = time.time() + timeout
    while True:
        for stamp, path in stamped_frames(out, camera):
            if stamp < stamp_ns:
                os.remove(path)                       # a frame from before this pose was set
            elif stamp < stamp_ns + period_ns:
                os.replace(path, os.path.join(out, f"{camera}_{index:05d}.png"))
                return True
            else:
                print(f"{camera}: a frame at t={stamp * 1e-9:.3f} s, past the one wanted at "
                      f"{stamp_ns * 1e-9:.3f} s: the world stepped further than asked", file=sys.stderr)
                return False
        if time.time() > deadline:
            return False
        time.sleep(0.002)


def wait_for_models(node: Node, world: str, expected: int, timeout_s: float = 600.0) -> set:
    """The names of the world's models, once all of them are there: the scene service
    answers while the world is still loading."""
    seen, t0 = None, time.time()
    while True:
        ok, scene = node.request(f"/world/{world}/scene/info", Empty(), Empty, Scene, 5000)
        assert ok, "the world's scene service did not answer: is the server up on this GZ_PARTITION?"
        names = {model.name for model in scene.model}
        if names != seen:
            seen = names
            print(f"{len(names)} of {expected} models loaded after {time.time() - t0:.0f} s", flush=True)
        if len(names) >= expected:
            return names
        assert time.time() - t0 < timeout_s, f"the world stopped at {len(names)} of {expected} models"
        time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("poses")
    ap.add_argument("out")
    ap.add_argument("--cameras", nargs="*", default=["cycle_hero"])
    ap.add_argument("--rate", type=float, default=25.0, help="camera rate of the replay world")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=1e9)
    ap.add_argument("--times", type=float, nargs="*", default=None, help="only the frames nearest these times")
    ap.add_argument("--world", default="sightline_replay")
    ap.add_argument("--world-sdf", default="results/ros2/gazebo/sightline_replay.sdf",
                    help="the world file the server was given, to know how many models to wait for")
    args = ap.parse_args()
    rec = np.load(args.poses, allow_pickle=True)
    names = [str(n) for n in rec["body_names"]]
    t, xpos, xquat = rec["t"], rec["xpos"], rec["xquat"]
    frames = np.nonzero((t >= args.start) & (t <= args.end))[0]
    if args.times:
        frames = np.array(sorted({int(np.argmin(np.abs(t - when))) for when in args.times}))
    os.makedirs(args.out, exist_ok=True)
    for name in (READY_FILE, STOP_FILE):
        if os.path.exists(os.path.join(args.out, name)):
            os.remove(os.path.join(args.out, name))
    grabber = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__), "frame_grabber.py"),
                                args.out, *args.cameras])
    while not os.path.exists(os.path.join(args.out, READY_FILE)):
        time.sleep(0.05)
    node = Node()
    pose_service = f"/world/{args.world}/set_pose_vector"
    control_service = f"/world/{args.world}/control"
    # only bodies that are models of the world get a pose: a body without visuals
    # (a joint frame of the worker) has no model, and one unknown name fails the
    # whole request
    models = wait_for_models(node, args.world, open(args.world_sdf).read().count("<model name="))
    # the recording names bodies as MuJoCo does ("worker/head"); the exported model,
    # and so the world, writes "__" for the slash
    posed = [(b, name.replace("/", "__")) for b, name in enumerate(names) if name.replace("/", "__") in models]
    print(f"{len(posed)} of {len(names)} recorded bodies are models in the world", flush=True)
    period_ns = int(round(1e9 / args.rate))
    sim_ns = 0
    t0 = time.time()
    missed = 0
    try:
        for k, i in enumerate(frames):
            msg = Pose_V()
            for b, name in posed:
                p = msg.pose.add()
                p.name = name
                p.position.x, p.position.y, p.position.z = map(float, xpos[i, b])
                w, x, y, z = map(float, xquat[i, b])
                p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z = w, x, y, z
            ok, rep = node.request(pose_service, msg, Pose_V, Boolean, 2000)
            if not ok or not rep.data:
                print(f"frame {k}: set_pose_vector failed", file=sys.stderr)
            # the poses take effect on the next step; the world runs one camera period,
            # to an absolute sim time, and the cameras render at that instant
            sim_ns += period_ns
            ctrl = WorldControl()
            ctrl.run_to_sim_time.sec = sim_ns // 1_000_000_000
            ctrl.run_to_sim_time.nsec = sim_ns % 1_000_000_000
            ok, rep = node.request(control_service, ctrl, WorldControl, Boolean, 2000)
            if not ok or not rep.data:
                print(f"frame {k}: world control failed", file=sys.stderr)
            for cam in args.cameras:
                if not take_frame(args.out, cam, sim_ns, period_ns, k, 10.0):
                    missed += 1
                    print(f"frame {k}: {cam} gave no image for t={sim_ns * 1e-9:.3f}", file=sys.stderr)
            if k % 100 == 0:
                print(f"frame {k}/{len(frames)} run t={t[i]:.2f} s, {time.time() - t0:.0f} s wall", flush=True)
    finally:
        open(os.path.join(args.out, STOP_FILE), "w").close()
        try:
            grabber.wait(5)
        except subprocess.TimeoutExpired:
            grabber.kill()
        for name in (READY_FILE, STOP_FILE):
            if os.path.exists(os.path.join(args.out, name)):
                os.remove(os.path.join(args.out, name))
    print(f"{len(frames)} frames in {time.time() - t0:.0f} s wall, {missed} camera frames missed")


if __name__ == "__main__":
    main()
