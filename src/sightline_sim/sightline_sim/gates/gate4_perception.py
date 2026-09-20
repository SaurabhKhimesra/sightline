"""Gate 4: what the robot can work out from its two cameras.

Runs the same episode as gate 3, with the eyes pipeline on every camera frame and
the wrist pipeline whenever the robot is lining up on a hole. Measures what docs/design.md
section 12 asks for: person recall and precision, delay, box pose error, hole error,
and how well blocking is predicted before it happens.

**What would kill this gate:** person recall so low that the planner would be
planning around a person it cannot see, or a box and hole error too large to put a
screw in. Both are reported straight.

Usage: MUJOCO_GL=egl python scripts/gate4_perception.py [out_dir] [seed] [--video]
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

from sightline_sim.judge import PerceptionTruth, match  # noqa: E402
from sightline_planner import B0, SensorFrame, Taught, ToolKinematics  # noqa: E402
from sightline_planner.perceive import Calibration, Perception, blocks_line, find_hole  # noqa: E402
from sightline_sim import pose, script, sensors, station  # noqa: E402

FPS = 30.0
MOTION_HZ = 100.0
CAM_W, CAM_H = 640, 480
HOLE_D = 0.0046          # the cover hole, from the part drawing
CALIB_POS_MM = 2.0       # how far out the cell's own calibration is (scene choice)
CALIB_DEG = 0.15
SIGHT_RADIUS = 0.03      # how close a person voxel has to be to a sight line to block it


def rotation_error(rng, degrees: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.radians(degrees)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def draw_points(img: np.ndarray, uv: np.ndarray, colour, size: int = 1) -> None:
    h, w = img.shape[:2]
    for u, v in uv:
        u0, v0 = int(u) - size, int(v) - size
        u1, v1 = int(u) + size + 1, int(v) + size + 1
        if u1 < 0 or v1 < 0 or u0 >= w or v0 >= h:
            continue
        img[max(v0, 0):min(v1, h), max(u0, 0):min(u1, w)] = colour


def draw_line(img: np.ndarray, a, b, colour) -> None:
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
    if n > 4000:
        return
    us = np.linspace(a[0], b[0], n).astype(int)
    vs = np.linspace(a[1], b[1], n).astype(int)
    ok = (us >= 0) & (us < img.shape[1]) & (vs >= 0) & (vs < img.shape[0])
    img[vs[ok], us[ok]] = colour


def project(points: np.ndarray, cam_pos, cam_R, f: float, w: int, h: int):
    rel = (np.atleast_2d(points) - cam_pos) @ cam_R
    front = rel[:, 2] < -1e-6
    uv = np.full((rel.shape[0], 2), -1e6)
    uv[front, 0] = w / 2 - 0.5 + f * (rel[front, 0] / -rel[front, 2])
    uv[front, 1] = h / 2 - 0.5 - f * (rel[front, 1] / -rel[front, 2])
    return uv, front


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = Path(args[0] if args else "results/gate4")
    seed = int(args[1]) if len(args) > 1 else 0
    want_video = "--video" in sys.argv
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(seed)

    scene = station.build()
    m, d = scene.model, mujoco.MjData(scene.model)
    qadr = pose.robot_qadr(m)
    park = park_pose()
    d.qpos[qadr] = park
    mujoco.mj_forward(m, d)

    script.parts_to_start(scene, d)
    print(f"seed {seed}: solving the worker's key poses")
    from sightline_sim.gates.gate3_b0 import jig_signal_times, jittered_cycle          # the same episode as gate 3
    pb = script.build_playback(scene, d, segments=jittered_cycle(scene, rng))
    signals = jig_signal_times(pb)

    kin = ToolKinematics(station.robot_only_model())
    base = np.array(scene.points["box_home"], float)
    taught = Taught(part_pose=tuple(base),
                    rail_holes=[tuple(np.array(h, float) - base) for h in scene.points["rail_holes"]],
                    cover_holes=[tuple(np.array(h, float) - base) for h in scene.points["cover_holes"]],
                    feeder_pick=tuple(np.array(scene.points["feeder_pick"], float) + [0, 0, 0.004]))
    b0 = B0.build(kin, taught, park_tip=(-0.45, 0.33, scene.worktop_z + 0.30))

    eyes = sensors.DepthCamera(m, "eyes_cam", CAM_W, CAM_H, seed=seed, delay_frames=1)
    wrist = sensors.Camera(m, "wrist_cam", CAM_W, CAM_H)
    truth = PerceptionTruth(m, "eyes_cam", CAM_W, CAM_H)
    cam_pos, cam_R = sensors.camera_pose(m, d, "eyes_cam")
    calib = Calibration(pos=cam_pos + rng.normal(0, CALIB_POS_MM / 1000, 3),
                        R=rotation_error(rng, CALIB_DEG) @ cam_R,
                        width=CAM_W, height=CAM_H, fovy_deg=float(m.cam_fovy[m.cam("eyes_cam").id]))
    per = Perception(calib, kin, taught)
    per.set_background(sensors.empty_station_depth(eyes, m, d))
    print(f"  background taken, calibration off by {CALIB_POS_MM} mm and {CALIB_DEG} deg")

    hero = None
    writer = None
    if want_video:
        hero = sensors.Camera(m, "cycle_hero", 1280, 720)
        writer = imageio.get_writer(out / f"perception_seed{seed}.mp4", fps=FPS, codec="libx264",
                                    quality=7, macro_block_size=1, pixelformat="yuv420p")

    dt = pb.dt
    motion_every = max(1, int(round(1.0 / (MOTION_HZ * dt))))
    q = np.array(park, float)
    qd = np.zeros(6)
    state = {"step": 0, "signal": "empty", "wrist_img": None}
    rows, wrist_rows, block_rows = [], [], []
    centroids = []

    def on_step(t: float) -> None:
        nonlocal q, qd
        i = state["step"]
        while signals and t >= signals[0][0]:
            state["signal"] = signals.pop(0)[1]
        if i % motion_every == 0:
            qd = b0.update(SensorFrame(t=t, q=q.copy(), qd=qd.copy(), wrist_rgb=state["wrist_img"],
                                       jig_signal=state["signal"]))
        q = kin.clamp(q + qd * dt)
        d.qpos[qadr] = q
        state["step"] = i + 1

    def on_frame(t: float, index: int, seg) -> None:
        depth = eyes.capture(d)
        if index == 0:
            return           # the camera is one frame late, so there is no picture yet
        t_start = time.perf_counter()
        model = per.update(depth, q, t)
        compute_ms = (time.perf_counter() - t_start) * 1000

        true_voxels = truth.person_voxels(d)
        recall, precision = match(model.voxels, true_voxels, per.voxel)
        row = {"t": round(t, 2), "voxels": int(len(model.voxels)), "true_voxels": int(len(true_voxels)),
               "unseen": int(len(model.unseen)), "recall": None if np.isnan(recall) else round(recall, 4),
               "precision": None if np.isnan(precision) else round(precision, 4),
               "compute_ms": round(compute_ms, 1), "speed_max": round(model.speed_max, 3)}
        tb = truth.box_pose(d, scene.layout.jig_xy)
        if model.box_pose is not None and tb is not None:
            row["box_err_mm"] = [round((model.box_pose[k] - tb[k]) * 1000, 1) for k in range(3)]
            row["box_yaw_err_deg"] = round(float(np.degrees(model.box_pose[3] - tb[3])), 1)
        rows.append(row)
        centroids.append((t,
                          model.voxels.mean(axis=0) if len(model.voxels) else None,
                          true_voxels.mean(axis=0) if len(true_voxels) else None))

        # the wrist camera, whenever the robot is lining up or going in
        state["wrist_img"] = wrist.rgb(d)
        if b0.state in ("servo", "descend") and b0.index < len(b0.screws):
            screw = b0.screws[b0.index]
            hole = np.asarray(screw.hole, float)
            if model.box_pose_held is not None:
                held = model.box_pose_held
                hole = hole + [held[0] - base[0], held[1] - base[1], 0.0]
            cpos, cR = kin.camera_pose(q)
            rel = cR.T @ (hole - cpos)
            owner = truth.hole_owner(d, screw.hole)
            entry = {"t": round(t, 2), "screw": screw.name, "state": b0.state, "owner": owner}
            if rel[2] < -1e-6:
                f = (CAM_H / 2) / np.tan(np.radians(kin.camera_fovy()) / 2)
                u = CAM_W / 2 - 0.5 + f * (rel[0] / -rel[2])
                v = CAM_H / 2 - 0.5 - f * (rel[1] / -rel[2])
                dist = float(np.linalg.norm(rel))
                got = find_hole(state["wrist_img"], (u, v), HOLE_D * f / dist)
                entry["found"] = got is not None
                if got is not None:
                    cu, cv, area = got
                    ray = cR @ np.array([(cu - CAM_W / 2 + 0.5) / f, -(cv - CAM_H / 2 + 0.5) / f, -1.0])
                    world = cpos + ray * ((screw.hole[2] - cpos[2]) / ray[2])
                    entry["error_mm"] = round(float(np.linalg.norm(world[:2] - np.asarray(screw.hole)[:2])) * 1000, 2)
            else:
                entry["found"] = False
            wrist_rows.append(entry)
            # blocking prediction: does the planner think the person is on the sight line?
            person_only = blocks_line(model.voxels, cpos, np.asarray(screw.hole, float), SIGHT_RADIUS)
            with_unseen = person_only or blocks_line(model.unseen, cpos, np.asarray(screw.hole, float),
                                                     SIGHT_RADIUS)
            actually = owner.startswith(station.HUMAN_PREFIX)
            block_rows.append({"t": round(t, 2), "screw": screw.name, "predicted": bool(person_only),
                               "predicted_with_unseen": bool(with_unseen), "actual": bool(actually)})

        if writer is not None:
            frame = hero.rgb(d)
            hpos, hR = sensors.camera_pose(m, d, "cycle_hero")
            f_hero = (720 / 2) / np.tan(np.radians(float(m.cam_fovy[m.cam("cycle_hero").id])) / 2)
            if len(model.unseen):
                uv, ok = project(model.unseen[::3], hpos, hR, f_hero, 1280, 720)
                draw_points(frame, uv[ok], (250, 170, 40), 0)
            if len(model.voxels):
                sample = model.voxels[::10]
                uv_cam, _ = project(np.atleast_2d(cam_pos), hpos, hR, f_hero, 1280, 720)
                uv_v, ok = project(sample, hpos, hR, f_hero, 1280, 720)
                for point in uv_v[ok][::6]:
                    draw_line(frame, uv_cam[0], point, (90, 170, 255))
                uv, ok = project(model.voxels, hpos, hR, f_hero, 1280, 720)
                draw_points(frame, uv[ok], (40, 220, 90), 1)
            writer.append_data(frame)

    print("running the episode")
    script.run(scene, d, pb, on_frame=on_frame, fps=FPS, on_step=on_step, duration=pb.duration)

    eyes.close()
    wrist.close()
    truth.close()
    if hero is not None:
        hero.close()
    if writer is not None:
        writer.close()

    # Delay: how many frames back the truth has to be shifted to fit best. Only frames
    # where the person is actually moving say anything: standing still, every lag ties.
    moving = []
    for i in range(1, len(centroids)):
        a, b = centroids[i - 1][2], centroids[i][2]
        if a is not None and b is not None and float(np.linalg.norm(b - a)) > 0.004:
            moving.append(i)
    lags = {}
    for lag in range(0, 5):
        errs = [float(np.linalg.norm(centroids[i][1] - centroids[i - lag][2]))
                for i in moving if i - lag >= 0 and centroids[i][1] is not None
                and centroids[i - lag][2] is not None]
        if errs:
            lags[lag] = round(float(np.mean(errs)) * 1000, 1)
    best_lag = min(lags, key=lags.get) if lags else None

    seen = [r for r in rows if r["recall"] is not None and r["true_voxels"] > 20]
    box = [r for r in rows if "box_err_mm" in r]
    found = [w for w in wrist_rows if w.get("found")]
    result = {
        "gate": 4, "seed": seed, "date": "2026-09-18",
        "note": "eyes pipeline every camera frame during the gate 3 B0 episode, wrist "
                "pipeline while lining up on a hole.",
        "frames": len(rows),
        "wall_time_s": round(time.time() - t0, 1),
        "person": {
            "frames_with_person": len(seen),
            "recall_mean_pct": round(float(np.mean([r["recall"] for r in seen])) * 100, 1) if seen else None,
            "recall_worst_pct": round(float(np.min([r["recall"] for r in seen])) * 100, 1) if seen else None,
            "precision_mean_pct": round(float(np.mean([r["precision"] for r in seen if r["precision"] is not None])) * 100, 1) if seen else None,
            "voxels_mean": int(np.mean([r["voxels"] for r in seen])) if seen else 0,
            "unseen_voxels_mean": int(np.mean([r["unseen"] for r in seen])) if seen else 0,
            "compute_ms_mean": round(float(np.mean([r["compute_ms"] for r in rows])), 1),
            "compute_ms_worst": round(float(np.max([r["compute_ms"] for r in rows])), 1),
        },
        "delay": {"mean_centroid_error_mm_by_lag_frames": lags, "best_lag_frames": best_lag,
                  "frames_with_motion": len(moving)},
        "box": {
            "frames_fitted": len(box),
            "frames_fitted_pct": round(100.0 * len(box) / max(len(rows), 1), 1),
            "xy_error_mm_median": round(float(np.median([np.hypot(*r["box_err_mm"][:2]) for r in box])), 1) if box else None,
            "xy_error_mm_worst": round(float(np.max([np.hypot(*r["box_err_mm"][:2]) for r in box])), 1) if box else None,
            "z_error_mm_median": round(float(np.median([abs(r["box_err_mm"][2]) for r in box])), 1) if box else None,
            "yaw_error_deg_worst": round(float(np.max([abs(r["box_yaw_err_deg"]) for r in box])), 1) if box else None,
        },
        "hole": {
            "approach_frames": len(wrist_rows),
            "found": len(found),
            "found_pct": round(100.0 * len(found) / max(len(wrist_rows), 1), 1),
            "error_mm_median": round(float(np.median([w["error_mm"] for w in found])), 2) if found else None,
            "error_mm_worst": round(float(np.max([w["error_mm"] for w in found])), 2) if found else None,
            "what_is_at_the_hole": {},
        },
        "blocking_prediction": {},
        "trace": rows[::3],
        "wrist_trace": wrist_rows[:60],
    }
    owners: dict = {}
    for w in wrist_rows:
        owners[w["owner"]] = owners.get(w["owner"], 0) + 1
    result["hole"]["what_is_at_the_hole"] = dict(sorted(owners.items(), key=lambda kv: -kv[1]))
    if block_rows:
        for key, label in (("predicted", "person_voxels_only"), ("predicted_with_unseen", "with_unseen_space")):
            tp = sum(1 for b in block_rows if b[key] and b["actual"])
            fp = sum(1 for b in block_rows if b[key] and not b["actual"])
            fn = sum(1 for b in block_rows if not b[key] and b["actual"])
            tn = sum(1 for b in block_rows if not b[key] and not b["actual"])
            result["blocking_prediction"][label] = {
                "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
                "accuracy_pct": round(100.0 * (tp + tn) / len(block_rows), 1),
                "frames": len(block_rows)}
    (out / f"gate4_seed{seed}.yaml").write_text(yaml.safe_dump(result, sort_keys=False))
    p = result["person"]
    print(f"\nperson: recall {p['recall_mean_pct']} % mean, {p['recall_worst_pct']} % worst, "
          f"precision {p['precision_mean_pct']} %, {p['voxels_mean']} voxels, "
          f"{p['unseen_voxels_mean']} unseen, {p['compute_ms_mean']} ms")
    print(f"delay: {result['delay']}")
    print(f"box: {result['box']}")
    print(f"hole: {result['hole']}")
    print(f"blocking: {result['blocking_prediction']}")
    print(f"wrote {out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
