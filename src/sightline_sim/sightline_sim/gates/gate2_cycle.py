"""Gate 2: the worker builds one box while the robot stays parked.

Plays the scripted cycle, measures it, and records the hero and overhead views.

Usage: MUJOCO_GL=egl python scripts/gate2_cycle.py [out_dir]
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from sightline_sim.data import park_pose  # noqa: E402

from sightline_sim import measure, pose, script, station  # noqa: E402

FPS = 30.0
# file name -> camera, width, height. Fog is off: at 3.4 m the hall fog washes the
# worker out, and gate 2 is about seeing his hands.
SHOTS = {"cycle_hero": ("cycle_hero", 1600, 900), "cycle_eyes_cam": ("eyes_cam", 1280, 720)}


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "results/gate2")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    scene = station.build()
    m = scene.model
    d = mujoco.MjData(m)

    # park the robot where gate 1 put it
    park = park_pose()
    d.qpos[pose.robot_qadr(m)] = park
    mujoco.mj_forward(m, d)

    script.parts_to_start(scene, d)  # empty jig, parts in their totes
    print("solving the key poses")
    pb = script.build_playback(scene, d)
    print(f"  {len(pb.segments)} segments, {pb.duration:.1f} s")
    worst_reach = max((max([v for k, v in e.items() if k != "segment"], default=0.0) for e in pb.reach_errors_mm), default=0.0)
    print(f"  worst reach error {worst_reach:.1f} mm")
    for e in pb.reach_errors_mm:
        if max([v for k, v in e.items() if k != "segment"], default=0.0) > 5.0:
            print("   ", e)

    opt = mujoco.MjvOption()
    for g in range(6):
        opt.geomgroup[g] = 1 if g in (0, 1, 2) else 0
    renderers = {name: mujoco.Renderer(m, h, w) for name, (_, w, h) in SHOTS.items()}
    writers = {name: imageio.get_writer(out / f"{name}.mp4", fps=FPS, codec="libx264",
                                        quality=8, macro_block_size=1, pixelformat="yuv420p")
               for name in SHOTS}
    cam_frame = measure.camera_frame(m, d)
    seen_log = []

    def on_frame(t, index, seg):
        for name, r in renderers.items():
            r.update_scene(d, camera=SHOTS[name][0], scene_option=opt)
            r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = False
            writers[name].append_data(r.render())
        if index % 15 == 0:  # twice a second
            hands = measure.surface_points(m, d, ("lhand", "rhand", "lfingers", "rfingers", "lthumb", "rthumb"), 9)
            v = measure.visibility(m, d, scene.points["eyes_pos"], hands, measure.eyes_body(m), cam_frame)
            seen_log.append({"t": round(t, 2), "segment": seg.name, "hands_seen_pct": round(v["seen_pct"], 1)})

    print("playing the cycle")
    stats = script.run(scene, d, pb, on_frame=on_frame, fps=FPS)
    for r in renderers.values():
        r.close()
    for w in writers.values():
        w.close()

    worst = min(seen_log, key=lambda s: s["hands_seen_pct"]) if seen_log else {}
    stats["hands_seen_pct_worst"] = worst
    stats["hands_seen_pct_mean"] = round(float(np.mean([s["hands_seen_pct"] for s in seen_log])), 1) if seen_log else None
    stats["segments"] = [{"name": s.name, "seconds": s.duration} for s in pb.segments]
    stats["reach_errors_mm"] = pb.reach_errors_mm
    stats["worst_reach_error_mm"] = round(worst_reach, 1)
    stats["hands_seen_trace"] = seen_log
    (out / "gate2.yaml").write_text(yaml.safe_dump(stats, sort_keys=False))
    print({k: v for k, v in stats.items() if k not in ("segments", "hands_seen_trace")})
    print(f"wrote {out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
