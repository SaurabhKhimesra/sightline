"""Put the judge's record on a video whose frame k is frame k of the recording
(Gazebo's live camera video from ros2/frame_grabber.py --video), with a lead of
still frames first, and a line saying where the picture came from.

    python ros2/caption_video.py <poses.npz> <in.mp4> <out.mp4> [--lead 2.0] [--source "..."]
"""
import argparse
import os
import sys

import numpy as np
import imageio.v2 as imageio

from sightline_sim.episode import ROBOT_WORDS, captioned  # noqa: E402
from sightline_sim.episode import TITLES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("poses")
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--lead", type=float, default=2.0, help="seconds of the first frame held before the run")
    ap.add_argument("--source", default="PICTURE: GAZEBO SIM FRONT CAMERA, LIVE ON THE GPU")
    args = ap.parse_args()
    with np.load(args.poses, allow_pickle=True) as loaded:
        rec = {key: loaded[key] for key in loaded.files}
    t, fps, variant = rec["t"], float(rec["fps"]), str(rec["variant"])
    title = TITLES.get(variant, variant)
    lead = int(round(args.lead * fps))
    reader = imageio.get_reader(args.video)
    writer = imageio.get_writer(args.out, fps=fps, codec="libx264", quality=7, macro_block_size=1, pixelformat="yuv420p")
    n = 0
    for k, frame in enumerate(reader):
        i = min(max(k - lead, 0), len(t) - 1)
        screw = str(rec["screw"][i])
        doing = ROBOT_WORDS.get(str(rec["robot_state"][i]), str(rec["robot_state"][i]).upper()).format(
            s=screw.replace("_", " SCREW ").upper())
        verdict = str(rec["verdict"][i])
        if verdict == "brake":
            doing += " | HOLDING: HE IS CLOSE"
        elif verdict.startswith("escape"):
            doing += " | BACKING AWAY FROM HIM"
        lines = [(f"SIGHTLINE  {title} | T {t[i]:5.1f} S", (0.95, 0.95, 0.92)),
                 (f"ROBOT: {doing}", (0.55, 0.85, 1.0)),
                 (f"WORKER: {str(rec['worker_step'][i]).upper()}", (1.0, 0.85, 0.45)),
                 (f"SCREWS {int(rec['screws_done'][i])}/6 | CONTACTS {int(rec['contacts'][i])} | "
                  f"FRAMES WITH HIS VIEW BLOCKED {int(rec['blocked'][i])}", (0.85, 0.95, 0.85)),
                 (args.source, (0.75, 0.75, 0.72))]
        writer.append_data(captioned(np.asarray(frame)[:, :, :3], lines, px=3))
        n += 1
    writer.close()
    print(f"{args.out}: {n} frames, {n / fps:.1f} s")


if __name__ == "__main__":
    main()
