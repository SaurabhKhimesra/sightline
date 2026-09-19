"""Make the video of a run rendered by Gazebo: the frames ros2/replay_gazebo.py saved,
captioned with what the judge recorded at each instant (the same words as the
MuJoCo video, from the same recording), and a line saying where the picture came
from. Nothing in the picture is changed.

    python ros2/assemble_video.py results/ros2/run_B4_seed0_poses.npz \\
        results/ros2/gazebo/replay_frames_B4_seed0 results/ros2/gazebo --cameras cycle_hero eyes_cam
"""
import argparse
import os
import sys

import numpy as np
import imageio.v2 as imageio

from sightline_sim.episode import ROBOT_WORDS, captioned  # noqa: E402

TITLES = {"B0": "B0: IGNORES HIM", "B2": "B2: KEEPS 10 CM FROM HIM", "B3": "B3: 10 CM AND OUT OF HIS VIEW",
          "B4": "B4: PLANS ON THE CAMERA GRID"}
SOURCE = "PICTURE: GAZEBO SIM REPLAY OF THE RUN JUDGED IN MUJOCO"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("poses")
    ap.add_argument("frames")
    ap.add_argument("out")
    ap.add_argument("--cameras", nargs="*", default=["cycle_hero", "eyes_cam"])
    ap.add_argument("--end", type=int, default=None, help="stop after this many frames")
    args = ap.parse_args()
    with np.load(args.poses, allow_pickle=True) as loaded:
        rec = {key: loaded[key] for key in loaded.files}    # read once, not at every access
    t, fps, variant, seed = rec["t"], float(rec["fps"]), str(rec["variant"]), int(rec["seed"])
    title = TITLES.get(variant, variant)
    os.makedirs(args.out, exist_ok=True)
    for cam in args.cameras:
        short = {"cycle_hero": "hero", "eyes_cam": "eyes", "front": "front"}.get(cam, cam)
        path = os.path.join(args.out, f"{variant}_{short}_gazebo_seed{seed}.mp4")
        writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=1,
                                    pixelformat="yuv420p")
        count = 0
        for k in range(len(t)):
            if args.end is not None and k >= args.end:
                break
            frame_path = os.path.join(args.frames, f"{cam}_{k:05d}.png")
            if not os.path.exists(frame_path):
                print(f"{cam}: frame {k} missing, stopping at t={t[k]:.2f} s", file=sys.stderr)
                break
            frame = imageio.imread(frame_path)[:, :, :3]
            screw = str(rec["screw"][k])
            doing = ROBOT_WORDS.get(str(rec["robot_state"][k]), str(rec["robot_state"][k]).upper()).format(
                s=screw.replace("_", " SCREW ").upper())
            verdict = str(rec["verdict"][k])
            if verdict == "brake":
                doing += " | HOLDING: HE IS CLOSE"
            elif verdict.startswith("escape"):
                doing += " | BACKING AWAY FROM HIM"
            seg = str(rec["worker_step"][k]).upper()
            done, contacts, blocked = int(rec["screws_done"][k]), int(rec["contacts"][k]), int(rec["blocked"][k])
            where = "EYES CAMERA | " if cam == "eyes_cam" else ""
            lines = [(f"SIGHTLINE  {title} | {where}T {t[k]:5.1f} S", (0.95, 0.95, 0.92)),
                     (f"ROBOT: {doing}", (0.55, 0.85, 1.0)),
                     (f"WORKER: {seg}", (1.0, 0.85, 0.45)),
                     (f"SCREWS {done}/6 | CONTACTS {contacts} | FRAMES WITH HIS VIEW BLOCKED {blocked}",
                      (0.85, 0.95, 0.85)),
                     (SOURCE, (0.75, 0.75, 0.72))]
            px = 2 if cam == "eyes_cam" else 3
            writer.append_data(captioned(frame, lines, px=px))
            count += 1
        writer.close()
        print(f"{path}: {count} frames, {count / fps:.1f} s")


if __name__ == "__main__":
    main()
