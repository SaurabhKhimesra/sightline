"""Gate 5: the rules as hard constraints, on the same episode as the baseline.

Four variants of one controller. B0 drives its taught path and never looks. B2 puts
R1 in the motion layer's constraints: it will not close on anything it thinks is a
person, or on the space it cannot see behind them. B3 adds R2: it also keeps every
link off the sight lines from the eyes camera to the person. B4 plans against the
camera's grid instead (viewgrid.py and guard.py): every direction in which the camera
sees him is off limits out to just behind him, rebuilt with every picture, and every
10 ms the arm's next moves are checked against it; it also chooses which hole to
drive, and how, from what is clear. All four share the same joint speed,
acceleration and position limits.

The eyes camera, and everything built from it, runs at 25 frames a second: the grid
is rebuilt every 40 ms (the user's choice), and the motion cycle is 10 ms.

The planner sees only what section 8.1 allows: the eyes depth image (one frame late,
noisy), its own joints, its own robot model, the cell drawing. The judge scores R1
and R2 from the simulator's state.

**What would kill this gate:** B2 or B3 still hitting the worker when the robot moved
in, or a cycle time so long the cell is useless. Both reported straight.

Usage: MUJOCO_GL=egl python scripts/gate5_planner.py [out_dir] [seed] [B0|B2|B3|B4] [--video] [--seconds N]
       [--vh measured|iso]
"""

from __future__ import annotations

import os
import sys
import time
from collections import Counter, deque
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from sightline_sim.data import park_pose  # noqa: E402

from sightline_sim.judge import R1Judge, R2Judge  # noqa: E402
from sightline_planner import B0, B4, SensorFrame, Taught, ToolKinematics  # noqa: E402
from sightline_planner.motion import MotionLayer, Obstacles  # noqa: E402
from sightline_planner.perceive import Calibration, KnownWorld, Perception, PersonModel, pack, unpack  # noqa: E402
from sightline_planner.viewgrid import ViewGrid  # noqa: E402
from sightline_sim import pose, script, sensors, station, textures  # noqa: E402

from sightline_planner.settings import (FPS, MOTION_HZ, CAM_W, CAM_H, CALIB_POS_MM, CALIB_DEG,  # noqa: E402
                                        HOLD_ASK_S, HELD_FRACTION, HAND_REACH_M)
from ..episode import ROBOT_WORDS, blind_cells, captioned, rotation_error  # noqa: E402


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = Path(args[0] if args else "results/gate5")
    seed = int(args[1]) if len(args) > 1 else 0
    variant = args[2] if len(args) > 2 else "B2"
    want_video = "--video" in sys.argv
    seconds = None
    if "--seconds" in sys.argv:
        seconds = float(sys.argv[sys.argv.index("--seconds") + 1])
    unseen_depth = None       # --unseen D: how far behind a seen voxel counts as occupied, for sensitivity runs
    if "--unseen" in sys.argv:
        unseen_depth = float(sys.argv[sys.argv.index("--unseen") + 1])
    probe_at = None           # --probe T: save the planner's whole world at time T, for diagnosis
    if "--probe" in sys.argv:
        probe_at = float(sys.argv[sys.argv.index("--probe") + 1])
    poses_file = None         # --record-poses F: every body's pose at every frame, for replaying the run elsewhere
    if "--record-poses" in sys.argv:
        poses_file = Path(sys.argv[sys.argv.index("--record-poses") + 1])
    record = None             # --record F: save every picture's person pixels and his true hands, for tuning offline
    if "--record" in sys.argv:
        record = Path(sys.argv[sys.argv.index("--record") + 1])
    vh = "measured"           # --vh iso: the constant 2000 mm/s person speed instead of the measured one
    if "--vh" in sys.argv:
        vh = sys.argv[sys.argv.index("--vh") + 1]
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
    print(f"{variant} seed {seed}: solving the worker's key poses")
    from sightline_sim.gates.gate3_b0 import jig_signal_times, jittered_cycle
    pb = script.build_playback(scene, d, segments=jittered_cycle(scene, rng))
    signals = jig_signal_times(pb)

    zw = scene.worktop_z
    kin = ToolKinematics(station.robot_only_model())
    base = np.array(scene.points["box_home"], float)
    taught = Taught(part_pose=tuple(base),
                    rail_holes=[tuple(np.array(h, float) - base) for h in scene.points["rail_holes"]],
                    cover_holes=[tuple(np.array(h, float) - base) for h in scene.points["cover_holes"]],
                    feeder_pick=tuple(np.array(scene.points["feeder_pick"], float) + [0, 0, 0.004]))
    eyes = sensors.DepthCamera(m, "eyes_cam", CAM_W, CAM_H, seed=seed, delay_frames=1)
    cam_pos, cam_R = sensors.camera_pose(m, d, "eyes_cam")
    calib = Calibration(pos=cam_pos + rng.normal(0, CALIB_POS_MM / 1000, 3),
                        R=rotation_error(rng, CALIB_DEG) @ cam_R,
                        width=CAM_W, height=CAM_H, fovy_deg=float(m.cam_fovy[m.cam("eyes_cam").id]))
    known = KnownWorld(station.robot_only_model(), calib)
    planes = [(np.array([0.0, 0.0, zw]), np.array([0.0, 0.0, 1.0]))] if variant != "B0" else []
    grid = None
    if variant == "PARKED":
        # the arm stays at park the whole cycle: a clean record of him for tuning
        b0 = B0.build(kin, taught, park_tip=(-0.45, 0.33, zw + 0.30), home_q=park)
    elif variant == "B4":
        # the planner's own belief of where the camera is, calibration error and all
        grid = ViewGrid(calib, speed_mode=vh)
        b0 = B4.build(kin, taught, park_tip=(-0.45, 0.33, zw + 0.30), home_q=park,
                      eye=np.asarray(calib.pos, float), grid=grid, planes=planes)
    elif variant != "PARKED":
        b0 = B0.build(kin, taught, park_tip=(-0.45, 0.33, zw + 0.30), home_q=park)
    # Every variant runs through the same motion layer, so they share the joint speed,
    # acceleration and position limits of a real controller. B0 has the rules off and
    # sees no obstacles; B2 adds R1; B3 adds R2. B4 keeps the rules with its grid and
    # guard instead, so its motion layer keeps only the limits and the worktop.
    b0.motion = MotionLayer(kin, use_r1=(variant in ("B2", "B3")), use_r2=(variant == "B3"))
    per = Perception(calib, kin, taught, known=known) if unseen_depth is None else \
        Perception(calib, kin, taught, known=known, unseen_depth=unseen_depth)
    empty = sensors.empty_station_depth(eyes, m, d)
    per.set_background(empty)
    if grid is not None:
        grid.set_background(empty)
    # the jig and the box in it, from the part's drawing, 2 cm round and 3 cm over
    jig = (base, np.array(taught.cover_size) / 2 + 0.02, base[2] + taught.cover_top + 0.03)
    hidden = blind_cells(scene, cam_pos, jig=jig) if variant != "B0" else np.zeros((0, 3))
    print(f"  static knowledge: {len(hidden)} blind cells on the worker's side, worktop plane at {zw:.3f} m")

    r1 = R1Judge(m, qadr)
    r2 = R2Judge(m, qadr, "eyes_cam", CAM_W, CAM_H)
    renderers, writers = {}, {}
    if want_video:
        for name, cam, w, h in ((f"{variant}_hero", "cycle_hero", 1280, 720), (f"{variant}_eyes", "eyes_cam", 960, 720)):
            renderers[name] = (sensors.Camera(m, cam, w, h))
            writers[name] = imageio.get_writer(out / f"{name}_seed{seed}.mp4", fps=FPS, codec="libx264",
                                               quality=7, macro_block_size=1, pixelformat="yuv420p")

    dt = pb.dt
    motion_every = max(1, int(round(1.0 / (MOTION_HZ * dt))))
    q = np.array(park, float)
    qd = np.zeros(6)
    history: list = []
    cycle_back = max(1, int(round(0.010 / dt)))
    frame_back = max(1, int(round((1.0 / FPS) / dt)))
    model = PersonModel(t=0.0)
    state = {"step": 0, "signal": "empty", "held_since": None, "asks": [], "held_s": 0.0,
             "hand_speed": 0.0, "prev_hand": None, "task_zero": 0, "blind_active": [], "yield_s": 0.0,
             "verdict": "", "hands": deque(maxlen=int(round(0.5 / pb.dt)))}
    per_cycle = {"rows": [], "solve_ms": [], "fallback": 0, "min_dist": [], "min_guard": []}
    trace: list = []
    speeds: list = []            # per picture: what the grid measured against what his hands did
    recorded: list = []          # --record: (t, pixels, depths, true hands)
    poses: list = []             # --record-poses: (t, xpos, xquat, robot state, worker step, screws done, contacts, blocked)
    verdicts_by_state: Counter = Counter()

    # B2 and B3 keep their voxels for the rules; the grid here only says which blind
    # cells are live, the same way for every variant that uses them
    live_grid = grid if variant == "B4" else (ViewGrid(calib) if variant in ("B2", "B3") else None)
    state["live"] = np.zeros((0, 3))

    def live_blind() -> np.ndarray:
        """A blind cell counts only if a seen part of him is close enough to have put a
        hand in it; the far end of the bench is not a hiding place for a hand at the rack."""
        person = live_grid.seen_points() if live_grid is not None else np.zeros((0, 3))
        if not (person.shape[0] and hidden.shape[0]):
            state["blind_active"].append(0)
            return np.zeros((0, 3))
        near = np.zeros(len(hidden), dtype=bool)
        for chunk in range(0, len(person), 400):
            block = person[chunk:chunk + 400]
            gap = np.linalg.norm(hidden[:, None, :] - block[None, :, :], axis=2).min(axis=1)
            near |= gap < HAND_REACH_M
        state["blind_active"].append(int(near.sum()))
        return hidden[near]

    def obstacles_now() -> Obstacles:
        if variant == "B0":
            return Obstacles()
        if variant == "B4":
            return Obstacles(planes=planes)
        unseen = model.unseen
        active = state["live"]
        if active.shape[0]:
            unseen = np.vstack([unseen, active]) if unseen.shape[0] else active
        return Obstacles(person=model.voxels, person_velocity=model.velocity, unseen=unseen,
                         sight_from=calib.pos, planes=planes)

    def true_hands(t_from: float, t_to: float):
        """Each hand: where it was at t_to, its mean speed over the window, its peak speed in it."""
        rows = [h for h in state["hands"] if t_from - 1e-9 <= h[0] <= t_to + 1e-9]
        if len(rows) < 2:
            return []
        span = max(rows[-1][0] - rows[0][0], 1e-9)
        out = []
        for k in (1, 2):
            steps = [float(np.linalg.norm(rows[i + 1][k] - rows[i][k])) / max(rows[i + 1][0] - rows[i][0], 1e-9)
                     for i in range(len(rows) - 1)]
            out.append((rows[-1][k], float(np.linalg.norm(rows[-1][k] - rows[0][k])) / span, max(steps)))
        return out

    def on_step(t: float) -> None:
        nonlocal q, qd
        i = state["step"]
        while signals and t >= signals[0][0]:
            state["signal"] = signals.pop(0)[1]
        if i % motion_every == 0:
            frame = SensorFrame(t=t, q=q.copy(), qd=qd.copy(), jig_signal=state["signal"])
            wanted = b0._task(frame) if variant != "PARKED" else np.zeros(6)
            if b0.motion is not None:
                obs = obstacles_now()
                if probe_at is not None and t >= probe_at and not state.get("probed"):
                    state["probed"] = True
                    extra = {}
                    if variant == "B4":
                        import pickle
                        (out / f"probe_{variant}_seed{seed}_t{probe_at:.1f}.pkl").write_bytes(pickle.dumps(
                            {"grid": grid, "blind": b0.guard.blind, "q": q, "t": t, "signal": state["signal"],
                             "index": b0.index, "screws": b0.screws, "state": b0.state}))
                        extra = dict(grid_rows=grid.rows, grid_cols=grid.cols, grid_far=grid.far,
                                     grid_near=grid.near, grid_hidden=grid.hidden,
                                     grid_seen_at=grid.seen_at, grid_speed=grid.cell_speed,
                                     grid_taken_at=grid.taken_at, blind=b0.guard.blind,
                                     person_px=model.person_px, person_z=model.person_z)
                    np.savez(out / f"probe_{variant}_seed{seed}_t{probe_at:.1f}.npz",
                             q=q, wanted=wanted, person=obs.person, unseen=model.unseen,
                             guard=obs.guard_points(), hidden=hidden, sim_qpos=d.qpos.copy(), t=t, **extra)
                rep = b0.motion.solve_joint(q, wanted, obs)
                qd = rep.qd
                if variant == "B4":
                    before_state = b0.state
                    qd = b0.safe_command(frame, rep.qd)
                    b0.motion.last = qd                     # the next ramp starts from what was sent
                    state["verdict"] = b0.guard.last
                    verdicts_by_state[f"{before_state} {b0.guard.last}"] += 1
                if i % (motion_every * 5) == 0:          # 20 Hz trace for diagnosis
                    caps = b0.motion._world_capsules(q)
                    lowest = min(min(p0[2], p1[2]) - r for _, p0, p1, r in caps)
                    tip, _ = kin.fk(q)
                    trace.append({"t": round(t, 2), "state": b0.state, "tip": [round(float(v), 3) for v in tip],
                                  "lowest_link_z": round(float(lowest), 3), "fallback": rep.fallback,
                                  "guard": state["verdict"],
                                  "flagged": b0.guard.explain(q, t) if variant == "B4" else None,
                                  "rows": rep.constraints, "speed_ratio": round(float(np.linalg.norm(rep.qd)) /
                                                                               max(float(np.linalg.norm(wanted)), 1e-6), 3)})
                per_cycle["rows"].append(rep.constraints)
                per_cycle["solve_ms"].append(rep.solve_ms)
                per_cycle["fallback"] += rep.fallback != ""
                if np.isfinite(rep.min_person_distance):
                    per_cycle["min_dist"].append(rep.min_person_distance)
                if np.isfinite(rep.min_guard_distance):
                    per_cycle["min_guard"].append(rep.min_guard_distance)
            else:
                qd = wanted
            # held: the task wanted to move and the rules let almost none of it through
            want_speed = float(np.linalg.norm(wanted))
            if want_speed > 0.02 and float(np.linalg.norm(qd)) < HELD_FRACTION * want_speed:
                state["held_s"] += motion_every * dt
                if state["held_since"] is None:
                    state["held_since"] = t
                elif variant in ("B2", "B3") and t - state["held_since"] >= 0.3:
                    b0.step_back(t)                          # do not wait over the box; B4 decides for itself
            else:
                state["held_since"] = None
            # waiting at park for the hole to clear: after a while, ask him to move his hand
            if b0.state == "yield" and t - b0.yield_since >= HOLD_ASK_S and \
                    (not state["asks"] or state["asks"][-1] < b0.yield_since):
                state["asks"].append(round(t, 2))           # "please move your hand"
            if b0.state == "yield":
                state["yield_s"] += motion_every * dt
        q = kin.clamp(q + qd * dt)
        d.qpos[qadr] = q
        state["step"] = i + 1

    def after_step(t: float) -> None:
        history.append(q.copy())
        before = history[-cycle_back] if len(history) > cycle_back else None
        hands = [d.xpos[m.body(station.HUMAN_PREFIX + s + "hand").id].copy() for s in ("l", "r")]
        if state["prev_hand"] is not None:
            state["hand_speed"] = max(float(np.linalg.norm(a - b)) / dt for a, b in zip(hands, state["prev_hand"]))
        state["prev_hand"] = hands
        state["hands"].append((t, hands[0], hands[1], state["hand_speed"]))
        r1.step(d, t, before, person_speed=state["hand_speed"], measure_distance=(state["step"] % motion_every == 0))

    def on_frame(t: float, index: int, seg) -> None:
        nonlocal model
        depth = eyes.capture(d)
        # the picture is one frame late, so the arm in it is where the arm was a frame ago
        seen_q = state.get("q_last_frame", q.copy())
        state["q_last_frame"] = q.copy()
        if index > 0:
            model = per.update(depth, q, t, b0.parts_in_jig(state["signal"], per.last_box is not None),
                               q_at_capture=seen_q)
            if record is not None:
                recorded.append((t, model.person_px.astype(np.int32), model.person_z.astype(np.float32),
                                 np.array([[h[0], *h[1], *h[2]] for h in list(state["hands"])[-int(round(0.2 / dt)):]])))
            taken_at = t - 1.0 / FPS                         # the picture is one frame late
            if variant in ("B2", "B3"):
                b0.note_clear(t, model.voxels)
                live_grid.update(model.person_px, model.person_z, taken_at)
                state["live"] = live_blind()
            if variant == "B4":
                b0.see(model.person_px, model.person_z, taken_at, seen_q)
                state["live"] = live_blind()
                b0.guard.blind = state["live"]
                # what the grid gives the cells at each hand, against what that hand did:
                # its mean over the 120 ms the measurement spans, and its peak since then
                for (where, mean, _), (_, _, peak) in zip(true_hands(taken_at - 3.0 / FPS, taken_at),
                                                          true_hands(taken_at - 3.0 / FPS, t + 0.04)):
                    speeds.append((round(t, 3), round(grid.speed_near(where), 3), round(mean, 3),
                                   round(peak, 3), grid.remembered))
        before = history[-cycle_back] if len(history) > cycle_back else None
        last = history[-frame_back] if len(history) > frame_back else None
        r2.frame(d, t, before, last)
        if poses_file is not None:
            poses.append((t, d.xpos.copy(), d.xquat.copy(), b0.state,
                          b0.screws[b0.index].name if b0.index < len(b0.screws) else "",
                          pb.segments[script.segment_index(pb, t)].name, sum(s_.driven for s_ in b0.screws),
                          len(r1.contacts), r2.blocked_frames if hasattr(r2, "blocked_frames") else len(r2.blocked),
                          state.get("verdict", "")))
        if renderers:
            seg = pb.segments[script.segment_index(pb, t)].name.upper()
            screw = b0.screws[b0.index].name if b0.index < len(b0.screws) else ""
            doing = ROBOT_WORDS.get(b0.state, b0.state.upper()).format(s=screw.replace("_", " SCREW ").upper())
            verdict = state.get("verdict", "")
            if verdict == "brake":
                doing += " | HOLDING: HE IS CLOSE"
            elif verdict.startswith("escape"):
                doing += " | BACKING AWAY FROM HIM"
            by_robot = sum(c.who_moved_in == "robot" for c in r1.contacts)
            by_him = sum(c.who_moved_in == "person" for c in r1.contacts)
            views = sum(e.who_moved_in_frame == "robot" for e in r2.events)
            done = sum(s_.driven for s_ in b0.screws)
            alert = ""
            if r1._open:
                alert = "TOUCHING HIM (" + ("HE MOVED IN" if all(c.who_moved_in == "person" for c in r1._open.values())
                                           else "THE ROBOT MOVED IN") + ")"
            elif r2._open is not None:
                alert = "HIS VIEW IS BLOCKED"
            title = {"B0": "B0: IGNORES HIM", "B2": "B2: KEEPS 10 CM FROM HIM", "B3": "B3: 10 CM AND OUT OF HIS VIEW",
                     "B4": "B4: PLANS ON THE CAMERA GRID"}.get(variant, variant)
            lines = [(f"SIGHTLINE  {title}   SIMULATION  T {t:5.1f} S", (0.95, 0.95, 0.92)),
                     (f"ROBOT: {doing}", (0.55, 0.85, 1.0)),
                     (f"WORKER: {seg}", (1.0, 0.85, 0.45)),
                     (f"SCREWS {done}/6 | CONTACTS: BY ROBOT {by_robot} BY HIM {by_him} | "
                      f"VIEW BLOCKED BY ROBOT {views}", (0.85, 0.95, 0.85))]
            for name, cam in renderers.items():
                frame = cam.rgb(d)
                if name.endswith("_eyes") and grid is not None and grid.rows.size:
                    # the cells the grid holds as him, drawn on the eyes camera's own picture
                    img = frame.astype(np.float32)
                    k = frame.shape[0] / CAM_H * grid.cell
                    for hidden_cells, color in ((False, (235, 40, 30)), (True, (255, 150, 0))):
                        pick = grid.hidden == hidden_cells
                        for rr, cc in zip(grid.rows[pick], grid.cols[pick]):
                            y0, x0 = int(rr * k), int(cc * k)
                            y1, x1 = int((rr + 1) * k), int((cc + 1) * k)
                            img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * 0.55 + np.array(color) * 0.45
                    frame = img.astype(np.uint8)
                    eye_lines = [(f"SIGHTLINE  {title} | EYES CAMERA | T {t:5.1f} S", (0.95, 0.95, 0.92)),
                                 lines[1],
                                 ("RED: WHERE THE CAMERA SEES HIM. THE ARM STAYS OFF THOSE SIGHT LINES",
                                  (1.0, 0.55, 0.5)),
                                 ("ORANGE: HIDDEN BY THE ARM FOR A MOMENT, SO REMEMBERED".replace(",", ""),
                                  (1.0, 0.75, 0.4))]
                    writers[name].append_data(captioned(frame, eye_lines, px=2, alert=alert))
                else:
                    writers[name].append_data(captioned(frame, lines, px=3, alert=alert))

    duration = pb.duration if seconds is None else min(seconds, pb.duration)
    print(f"running {duration:.1f} s")
    stats = script.run(scene, d, pb, on_frame=on_frame, fps=FPS, on_step=on_step,
                       after_step=after_step, duration=duration)
    eyes.close()
    known.close()
    r2.close()
    for cam in renderers.values():
        cam.close()
    for w in writers.values():
        w.close()

    if poses_file is not None:
        np.savez_compressed(poses_file, t=np.array([p_[0] for p_ in poses]),
                            xpos=np.array([p_[1] for p_ in poses]), xquat=np.array([p_[2] for p_ in poses]),
                            body_names=np.array([m.body(i).name for i in range(m.nbody)]),
                            robot_state=np.array([p_[3] for p_ in poses]), screw=np.array([p_[4] for p_ in poses]),
                            worker_step=np.array([p_[5] for p_ in poses]), screws_done=np.array([p_[6] for p_ in poses]),
                            contacts=np.array([p_[7] for p_ in poses]), blocked=np.array([p_[8] for p_ in poses]),
                            verdict=np.array([p_[9] for p_ in poses]), fps=FPS, variant=variant, seed=seed)
        print(f"recorded {len(poses)} frames of body poses to {poses_file}")
    if record is not None:
        np.savez_compressed(record, t=np.array([r[0] for r in recorded]),
                            px=np.array([r[1] for r in recorded], dtype=object),
                            z=np.array([r[2] for r in recorded], dtype=object),
                            hands=np.array([r[3] for r in recorded], dtype=object),
                            calib_pos=calib.pos, calib_R=calib.R, empty=empty, worktop=zw, q_park=np.array(park))
        print(f"recorded {len(recorded)} pictures to {record}")
    summary = b0.summary()
    # each screw that went in, and what he was doing at that moment: the point of the
    # re-ordered cycle is that the robot works while he works
    driven_while = []
    for when, what, name in summary["state_log"]:
        if what == "done":
            seg = pb.segments[script.segment_index(pb, when)].name
            hands = [h for h in state["hands"] if abs(h[0] - when) < 0.05]
            driven_while.append({"t": when, "screw": name, "worker": seg})
    result = {
        "gate": 5, "variant": variant, "seed": seed, "date": "2026-09-18",
        "episode_s": round(duration, 2), "worker_cycle_s": round(pb.duration, 2),
        "wall_time_s": round(time.time() - t0, 1),
        "screws_driven": summary["screws_driven"],
        "screws_driven_while_he": driven_while,
        "screws_missed": summary.get("missed", []),
        "held_by_rules_s": round(state["held_s"], 2),
        "stepped_back_s": round(state["yield_s"], 2),
        "step_backs": summary.get("yields", []),
        "asked_worker_to_move": state["asks"],
        "blind_cells_used": int(len(hidden)),
        "blind_cells_active_median": float(np.median(state["blind_active"])) if state["blind_active"] else 0.0,
        "motion": {
            "cycles": len(per_cycle["rows"]),
            "rows_median": float(np.median(per_cycle["rows"])) if per_cycle["rows"] else 0,
            "rows_max": int(max(per_cycle["rows"])) if per_cycle["rows"] else 0,
            "solve_ms_median": round(float(np.median(per_cycle["solve_ms"])), 2) if per_cycle["solve_ms"] else 0,
            "solve_ms_max": round(float(max(per_cycle["solve_ms"])), 2) if per_cycle["solve_ms"] else 0,
            "fallbacks": per_cycle["fallback"],
            "planner_min_distance_mm": round(float(min(per_cycle["min_dist"])) * 1000, 1) if per_cycle["min_dist"] else None,
            "planner_min_guard_distance_mm": round(float(min(per_cycle["min_guard"])) * 1000, 1) if per_cycle["min_guard"] else None,
        },
        "R1": r1.results(), "R2": r2.results(),
        "worker": {k: stats[k] for k in ("peak_hand_speed_m_s", "peak_hand_accel_m_s2")},
        "fps": FPS,
        "planner_state_log": summary["state_log"],
        "motion_trace": trace,
    }
    if variant == "B4":
        sp = np.array([row[1:4] for row in speeds], float) if speeds else np.zeros((0, 3))
        seen = np.isfinite(sp[:, 0]) if len(sp) else np.zeros(0, dtype=bool)
        sp = sp[seen]
        moving = sp[:, 1] > 0.05 if len(sp) else np.zeros(0, dtype=bool)
        short = sp[:, 2] - sp[:, 0] if len(sp) else np.zeros(0)       # the hand's peak until the next picture, above what the grid gave it
        result["B4"] = {
            "person_speed_setting": vh,
            "guard": summary.get("guard"),
            "guard_verdicts_by_state": dict(sorted(verdicts_by_state.items())),
            "choices": summary.get("choices"), "swaps": summary.get("swaps"), "gave_up": summary.get("gave_up"),
            "options_per_screw": summary.get("options_per_screw"),
            "speed_check": {
                "hand_pictures_seen": int(len(sp)), "hand_pictures_not_seen": int((~seen).sum()),
                "grid_minus_true_mean_median_m_s": round(float(np.median(sp[moving, 0] - sp[moving, 1])), 3) if moving.any() else None,
                "grid_minus_true_mean_p5_m_s": round(float(np.percentile(sp[moving, 0] - sp[moving, 1], 5)), 3) if moving.any() else None,
                "grid_minus_true_mean_p95_m_s": round(float(np.percentile(sp[moving, 0] - sp[moving, 1], 95)), 3) if moving.any() else None,
                "grid_below_true_peak_pct": round(float((short > 0).mean() * 100), 1) if len(short) else None,
                "grid_below_true_peak_p95_m_s": round(float(np.percentile(short, 95)), 3) if len(short) else None,
                "grid_below_true_peak_max_m_s": round(float(short.max()), 3) if len(short) else None,
                "remembered_cells_max": int(max(row[4] for row in speeds)) if speeds else 0,
            },
            "speed_trace": [list(row) for row in speeds[::5]],
        }
    (out / f"{variant}_seed{seed}.yaml").write_text(yaml.safe_dump(result, sort_keys=False))
    r1r, r2r, mo = result["R1"], result["R2"], result["motion"]
    print(f"\n{variant}: screws {result['screws_driven']}/6 "
          f"({', '.join(d['screw'] + ' while he: ' + d['worker'] for d in driven_while) or 'none'}), "
          f"held by the rules {result['held_by_rules_s']} s, "
          f"missed {len(result['screws_missed'])}, "
          f"stepped back {len(result['step_backs'])} times for {result['stepped_back_s']} s, "
          f"asked him to move {len(state['asks'])} times")
    print(f"R1: {r1r['contacts']} contacts {r1r['contacts_by_who_moved_in']}, deepest {r1r['worst_contact_depth_mm']} mm, "
          f"judge min distance {r1r['min_distance_mm']} mm, planner thought {mo['planner_min_distance_mm']} mm")
    print(f"R2: {r2r['blocked_frames']}/{r2r['frames']} frames ({r2r['blocked_frames_pct']} %), "
          f"{r2r['blocked_events']} events, one frame back {r2r['blocked_events_by_who_moved_in_one_frame_back']}")
    print(f"QP: {mo['rows_median']:.0f} rows median / {mo['rows_max']} max, {mo['solve_ms_median']} ms median / "
          f"{mo['solve_ms_max']} max, {mo['fallbacks']} fallbacks in {mo['cycles']} cycles")
    if variant == "B4":
        g, sc = result["B4"]["guard"], result["B4"]["speed_check"]
        print(f"guard: {g['verdicts']}, {g['ms_median']} ms median / {g['ms_p99']} p99 / {g['ms_max']} max, "
              f"{g['points_on_the_arm']} points on the arm")
        print(f"choices {len(result['B4']['choices'])}, swaps on the way {len(result['B4']['swaps'])}, "
              f"screws left for now {len(result['B4']['gave_up'])}")
        print(f"person speed ({vh}), at his hands: grid minus true median {sc['grid_minus_true_mean_median_m_s']} m/s "
              f"(p5 {sc['grid_minus_true_mean_p5_m_s']}, p95 {sc['grid_minus_true_mean_p95_m_s']}); below the hand's peak "
              f"in {sc['grid_below_true_peak_pct']} % of pictures, by up to {sc['grid_below_true_peak_max_m_s']} m/s "
              f"(p95 {sc['grid_below_true_peak_p95_m_s']}); hand not seen in {sc['hand_pictures_not_seen']} of "
              f"{sc['hand_pictures_seen'] + sc['hand_pictures_not_seen']}")
    print(f"wrote {out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
