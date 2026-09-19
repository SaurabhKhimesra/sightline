"""Save every image the named Gazebo cameras publish, one PNG per frame, named by
the camera and the sim time stamp in nanoseconds: <out>/<camera>_t<stamp ns>.png.
Or, with --video, encode one camera's frames straight into an mp4 as they arrive,
one frame per sim time step of the camera (a missing step repeats the last frame),
which is how the live desktop recording gets Gazebo's picture without reading a
window back through X.

Runs until it is told to stop (SIGTERM, or the stop file appearing in <out>). It is
a process of its own on purpose: a Python callback that receives images and a
blocking service request from the same process wait on each other.

    python ros2/frame_grabber.py <out> cycle_hero eyes_cam
    python ros2/frame_grabber.py <out> front --video <out>/gazebo.mp4 --fps 25
"""
import os
import signal
import sys
import threading
import time

# the system's Python packages last: ahead of the venv they shadow its protobuf
sys.path[:] = [p for p in sys.path if "dist-packages" not in p] + [p for p in sys.path if "dist-packages" in p]

import numpy as np  # noqa: E402
import gz.math  # noqa: E402,F401  (registers the shared types the bindings below need)
from gz.transport import Node  # noqa: E402
from gz.msgs.image_pb2 import Image  # noqa: E402
import imageio.v2 as imageio  # noqa: E402

READY_FILE = ".grabber_ready"
STOP_FILE = ".grabber_stop"


class Saver:
    def __init__(self, camera: str, out: str):
        self.camera, self.out, self.count = camera, out, 0

    def __call__(self, msg: Image):
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nsec
        path = os.path.join(self.out, f"{self.camera}_t{stamp:015d}.png")
        imageio.imwrite(path + ".tmp.png", image)
        os.replace(path + ".tmp.png", path)          # complete files only, under the final name
        self.count += 1


class VideoSaver:
    """One camera's frames into an mp4, in sim time order: frame k is the picture
    stamped (k + 1) camera periods, and a period with no picture repeats the last."""

    def __init__(self, camera: str, path: str, fps: float):
        self.camera, self.fps, self.count, self.filled = camera, fps, 0, 0
        self.writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=1,
                                         pixelformat="yuv420p")
        self.next_index, self.last = 0, None
        self.lock = threading.Lock()

    def __call__(self, msg: Image):
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
        stamp = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
        index = int(round(stamp * self.fps)) - 1
        with self.lock:
            while self.next_index < index and self.last is not None:
                self.writer.append_data(self.last)             # a period the camera skipped
                self.next_index += 1
                self.filled += 1
            if index >= self.next_index:
                self.writer.append_data(image)
                self.last, self.next_index = image, index + 1
                self.count += 1

    def close(self):
        with self.lock:
            self.writer.close()


def main():
    args = sys.argv[1:]
    video, fps = None, 25.0
    if "--video" in args:
        video = args[args.index("--video") + 1]
        del args[args.index("--video"):args.index("--video") + 2]
    if "--fps" in args:
        fps = float(args[args.index("--fps") + 1])
        del args[args.index("--fps"):args.index("--fps") + 2]
    out, cameras = args[0], args[1:]
    os.makedirs(out, exist_ok=True)
    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    node = Node()
    savers = {cam: (VideoSaver(cam, video, fps) if video else Saver(cam, out)) for cam in cameras}
    for cam, saver in savers.items():
        assert node.subscribe(Image, f"/replay/{cam}", saver), cam
    open(os.path.join(out, READY_FILE), "w").close()
    while not stop["now"] and not os.path.exists(os.path.join(out, STOP_FILE)):
        time.sleep(0.05)
    for cam, saver in savers.items():
        node.unsubscribe(f"/replay/{cam}")
        if isinstance(saver, VideoSaver):
            saver.close()
            print(f"{cam}: {saver.count} frames encoded, {saver.filled} periods filled with the frame before", flush=True)
        else:
            print(f"{cam}: {saver.count} frames saved", flush=True)
    time.sleep(0.2)


if __name__ == "__main__":
    main()
