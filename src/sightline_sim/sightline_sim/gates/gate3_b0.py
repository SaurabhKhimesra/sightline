"""Gate 3: B0, the baseline that ignores the worker.

Plain visual servoing to every screw while the worker builds the box. The planner
sees only what docs/design.md section 8.1 allows. The judge measures R1 and R2 from the
simulator's own state.

**Kill test (docs/design.md section 12):** if B0 shows no contacts and no blocked frames on
the development seeds, there is nothing to avoid and the task needs rethinking.

Usage: MUJOCO_GL=egl python scripts/gate3_b0.py [out_dir] [seed] [--video]
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

from sightline_sim.judge import R1Judge, R2Judge  # noqa: E402
from sightline_planner import B0, SensorFrame, Taught, ToolKinematics  # noqa: E402
from sightline_sim import measure, pose, script, station  # noqa: E402

FPS = 30.0
MOTION_HZ = 100.0
MAX_EPISODE_S = 120.0
CAM_W, CAM_H = 640, 480


def jittered_cycle(scene: station.Scene, rng: np.random.Generator) -> list:
    """The same cycle with a seed's worth of variation, so episodes can be paired."""
    segs = script.cycle(scene)
    for s in segs:
        s.duration = float(s.duration * rng.uniform(0.88, 1.12))
        if s.stance is not None:
            s.stance = (s.stance[0] + rng.normal(0, 0.015), s.stance[1] + rng.normal(0, 0.015), s.stance[2])
        s.lean = float(np.clip(s.lean + rng.normal(0, 0.02), 0.10, 0.36))
    return segs


def jig_signal_times(pb: script.Playback) -> list:
    """When the cell can tell the robot a set of screws is ready.

    A real jig has a clamp switch. This is not a look at the person.
    """
    out = []
    clamped = any(getattr(seg, "signal", None) == "cover_ready" for seg in pb.segments)
    for seg, end in zip(pb.segments, pb.seg_times[1:]):
        for part, home in seg.release:
            if part == "part_rail" and home == "rail_home":
                out.append((end, "rail_ready"))
            # with the clamps, the cover placed ends the rail screws (they are under it)
            # and the clamps closing starts the cover screws; without, one switch does both
            if part == "part_cover" and home == "cover_home":
                out.append((end, "cover_on" if clamped else "cover_ready"))
        if getattr(seg, "signal", None):
            out.append((end, seg.signal))
    return sorted(out)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = Path(args[0] if args else "results/gate3")
    seed = int(args[1]) if len(args) > 1 else 0
    want_video = "--video" in sys.argv
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(seed)

    scene = station.build()
    m = scene.model
    d = mujoco.MjData(m)
    qadr = pose.robot_qadr(m)
    park = park_pose()
    d.qpos[qadr] = park
    mujoco.mj_forward(m, d)

    script.parts_to_start(scene, d)
    print(f"seed {seed}: solving the worker's key poses")
    pb = script.build_playback(scene, d, segments=jittered_cycle(scene, rng))
    signals = jig_signal_times(pb)
    print(f"  worker cycle {pb.duration:.1f} s, jig signals at {[(round(t, 1), s) for t, s in signals]}")

    # what the planner is given: its own robot model and the static cell numbers
    zw = scene.worktop_z
    kin = ToolKinematics(station.robot_only_model())
    jx, jy = scene.layout.jig_xy
    base = np.array(scene.points["box_home"], float)
    taught = Taught(part_pose=tuple(base),
                    rail_holes=[tuple(np.array(h, float) - base) for h in scene.points["rail_holes"]],
                    cover_holes=[tuple(np.array(h, float) - base) for h in scene.points["cover_holes"]],
                    feeder_pick=tuple(np.array(scene.points["feeder_pick"], float) + [0, 0, 0.004]))
    b0 = B0.build(kin, taught, park_tip=(-0.45, 0.33, zw + 0.30))
    print(f"  turn angles the planner chose: {b0.summary()['turn_angles_deg']}")

    r1 = R1Judge(m, qadr)
    r2 = R2Judge(m, qadr, "eyes_cam", CAM_W, CAM_H)
    wrist = mujoco.Renderer(m, CAM_H, CAM_W)
    opt = mujoco.MjvOption()
    for g in range(6):
        opt.geomgroup[g] = 1 if g in (0, 1, 2) else 0

    writers = {}
    renderers = {}
    if want_video:
        for name, cam, w, h in (("b0_hero", "cycle_hero", 1280, 720), ("b0_eyes", "eyes_cam", 960, 540)):
            renderers[name] = (mujoco.Renderer(m, h, w), cam)
            writers[name] = imageio.get_writer(out / f"{name}_seed{seed}.mp4", fps=FPS, codec="libx264",
                                               quality=7, macro_block_size=1, pixelformat="yuv420p")

    dt = pb.dt
    motion_every = max(1, int(round(1.0 / (MOTION_HZ * dt))))
    q = np.array(park, float)
    qd = np.zeros(6)
    history = []            # robot joints, for the judge's one-motion-cycle rewind
    cycle_back = max(1, int(round(0.010 / dt)))
    wrist_img = None
    state = {"step": 0, "signal": "empty", "hand_speed": 0.0, "prev_hand": None, "frames": 0, "robot_wait_s": 0.0}

    def on_step(t: float) -> None:
        nonlocal q, qd, wrist_img
        i = state["step"]
        while signals and t >= signals[0][0]:
            state["signal"] = signals.pop(0)[1]
        if i % motion_every == 0:
            frame = SensorFrame(t=t, q=q.copy(), qd=qd.copy(), wrist_rgb=wrist_img,
                                jig_signal=state["signal"])
            qd = b0.update(frame)
            if b0.state in ("wait", "drive"):
                state["robot_wait_s"] += motion_every * dt
        q = kin.clamp(q + qd * dt)
        d.qpos[qadr] = q
        state["step"] = i + 1

    def after_step(t: float) -> None:
        history.append(q.copy())
        before = history[-cycle_back] if len(history) > cycle_back else None
        hands = [d.xpos[m.body(station.HUMAN_PREFIX + s + "hand").id].copy() for s in ("l", "r")]
        if state["prev_hand"] is not None:
            state["hand_speed"] = max(float(np.linalg.norm(a - b)) / dt
                                      for a, b in zip(hands, state["prev_hand"]))
        state["prev_hand"] = hands
        r1.step(d, t, before, person_speed=state["hand_speed"],
                measure_distance=(state["step"] % motion_every == 0))

    def on_frame(t: float, index: int, seg) -> None:
        nonlocal wrist_img
        wrist.update_scene(d, camera="wrist_cam", scene_option=opt)
        wrist_img = wrist.render()
        before = history[-cycle_back] if len(history) > cycle_back else None
        frame_back = max(1, int(round((1.0 / FPS) / dt)))
        last_frame = history[-frame_back] if len(history) > frame_back else None
        r2.frame(d, t, before, last_frame)
        state["frames"] += 1
        for name, (r, cam) in renderers.items():
            r.update_scene(d, camera=cam, scene_option=opt)
            r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = False
            writers[name].append_data(r.render())

    print("running the episode")
    # one worker cycle. Running on with the worker frozen would count blocked frames
    # against a person who is not there any more.
    duration = min(MAX_EPISODE_S, pb.duration)
    stats = script.run(scene, d, pb, on_frame=on_frame, fps=FPS,
                       on_step=on_step, after_step=after_step, duration=duration)

    wrist.close()
    for r, _ in renderers.values():
        r.close()
    for w in writers.values():
        w.close()
    r2.close()

    result = {
        "gate": 3, "variant": "B0", "seed": seed, "date": "2026-09-18",
        "note": "visual servoing, no avoidance. Robot and worker both played back "
                "kinematically, so the robot tracks its commands exactly.",
        "episode_s": round(duration, 2),
        "worker_cycle_s": round(pb.duration, 2),
        "wall_time_s": round(time.time() - t0, 1),
        "planner": b0.summary(),
        "robot_idle_s": round(state["robot_wait_s"], 1),
        "R1": r1.results(),
        "R2": r2.results(),
        "worker": {k: stats[k] for k in ("peak_hand_speed_m_s", "peak_hand_accel_m_s2", "worst_hand_push_mm")},
    }
    (out / f"b0_seed{seed}.yaml").write_text(yaml.safe_dump(result, sort_keys=False))
    r1r, r2r = result["R1"], result["R2"]
    print(f"\nscrews driven {b0.summary()['screws_driven']}/{b0.summary()['screws_total']}, "
          f"robot idle {result['robot_idle_s']} s")
    print(f"R1: {r1r['contacts']} contacts {r1r['contacts_by_who_moved_in']}, "
          f"closest {r1r['min_distance_mm']} mm, deepest {r1r['worst_contact_depth_mm']} mm")
    print(f"R2: {r2r['blocked_frames']}/{r2r['frames']} frames blocked "
          f"({r2r['blocked_frames_pct']} %) in {r2r['blocked_events']} events "
          f"{r2r['blocked_events_by_who_moved_in']} (one frame back: "
          f"{r2r['blocked_events_by_who_moved_in_one_frame_back']}), longest {r2r['longest_block_s']} s, "
          f"worst {r2r['worst_blocked_pixels']} px")
    print(f"wrote {out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
