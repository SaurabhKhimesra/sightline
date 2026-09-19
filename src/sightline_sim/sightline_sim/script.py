"""The worker's assembly cycle: key poses, minimum jerk timing, kinematic playback.

The worker is played back, not simulated: every step writes his joint positions
from the script, so he can neither fall nor be pushed. Carried parts ride on the
hand frame; placed parts sit where they were put. SPEC.md section 7.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import measure, pose, station

PARTS = ("part_base", "part_rail", "part_cover")


def min_jerk(x: float) -> float:
    """Minimum jerk blend, 0 to 1 (Flash and Hogan 1985: 10x^3 - 15x^4 + 6x^5)."""
    x = min(1.0, max(0.0, x))
    return x * x * x * (10 - 15 * x + 6 * x * x)


@dataclass
class Segment:
    """One move of the cycle. Hand targets are world points; None leaves the arm as it is."""
    name: str
    duration: float
    left: tuple | None = None
    right: tuple | None = None
    lean: float = 0.24
    gaze: tuple | None = None
    left_fingers: tuple = (0.35, 1.0, 0.0)
    right_fingers: tuple = (-0.3, 1.0, 0.0)
    grab: tuple = ()        # parts picked up at the start of this segment: (part, hand)
    snap: dict = field(default_factory=dict)  # part: how far below the hand it snaps when grabbed
    stance: tuple | None = None  # (x, y, yaw) where the worker stands; None keeps the default
    release: tuple = ()     # parts put down at the end: (part, home point name)
    # (metres, seconds): the hands rise straight up off the last targets before this
    # segment's move. A hand that slides sideways off a part cuts straight through it,
    # because the key poses blend joint by joint and nothing keeps the path clear.
    lift: tuple | None = None
    note: str = ""
    signal: str | None = None   # the jig switch this segment ends by setting, if any


WAIT_FOR_COVER_SCREWS_S = 11.0    # how long he waits for the robot's four cover screws (scene choice)


def cycle(scene: station.Scene) -> list[Segment]:
    """One box, from an empty jig to a finished box in the tray.

    Durations and stances are scene choices. The rack is out of arm's reach from
    the bench, so the worker takes a side step to it and back, which is what
    SPEC.md section 7 allows. Every hand target here is inside the model's reach;
    build_playback reports the error if one is not.

    The cover is held by the jig's two toggle clamps, not by his hand (the user's
    decision, 2026-09-18): with his hand on the cover no planner could drive a cover
    screw 5 cm clear of him.

    His tasks are ordered so that the robot's two screwing windows fall while he has
    work to do beside it, the way a line is balanced (the user's decision, 2026-09-18):
    the rail screws while he clips blocks on the next rail and fetches the cover, the
    cover screws while he finishes that rail. The jig's switches tell the robot when
    the rail is pressed in, when the cover is on, when the clamps are closed and when
    they are opened again. He never looks at the robot.
    """
    L = scene.layout
    zw = scene.worktop_z
    jx, jy = L.jig_xy
    holes = scene.points["cover_holes"]
    cover_z = holes[0][2]
    prep = (L.prep_xy[0], L.prep_xy[1], zw + 0.05)
    tray = (L.tray_xy[0], L.tray_xy[1], zw + 0.10)
    over_jig = (jx - 0.02, jy - 0.02, zw + 0.26)
    on_cover = (jx - 0.062, jy - 0.050, cover_z + 0.024)
    at_bench = (L.worker_xy[0], L.worker_xy[1], np.pi / 2)
    # He steps over to work at the prep area and steps back from the bench to wait.
    # At the bench, reaching 42 cm to his left for the prep area or looking down at
    # the jig while waiting, his head sat over the jig's front edge 1.4 m up, 9 to 11
    # cm from the robot's wrist at any cover hole.
    at_prep = (L.prep_xy[0] + 0.02, L.worker_xy[1], np.pi / 2)
    # waiting, he steps back 30 cm from the bench and watches the robot with his hands
    # at his sides. Against the bench his head sat 23 cm from the front cover holes,
    # and the planner's rule that a hand can hide 15 cm behind anything it sees, plus
    # its margin, keeps the arm about 33 cm along the camera's line of sight behind his
    # head: the two left cover holes were inside that for the whole wait.
    at_wait = (L.worker_xy[0], L.worker_xy[1] - 0.30, np.pi / 2)
    watch = (jx, jy + 0.30, zw + 0.45)
    # hands resting on the worktop's front edge while he waits or steps along the
    # bench. Hanging them at his sides there made the key poses blend through wide
    # arcs, one of them over the jig into the robot and one through the bench edge.
    prep_left = (prep[0] - 0.04, -0.37, zw + 0.06)
    prep_right = (prep[0] + 0.12, -0.37, zw + 0.06)
    # hands down at his sides at the prep stance: where they hang, 2 cm up (the model's
    # hanging hands sit 0.245 m out and 0.045 m forward of the root at 0.84 m; 6 cm
    # higher or 4 cm further forward and the solve misses by 25 to 130 mm)
    at_side_left = (at_prep[0] - 0.245, at_prep[1] + 0.045, 0.86)
    at_side_right = (at_prep[0] + 0.245, at_prep[1] + 0.045, 0.86)
    bench_side_left = (at_wait[0] - 0.245, at_wait[1] + 0.045, 0.86)
    bench_side_right = (at_wait[0] + 0.245, at_wait[1] + 0.045, 0.86)
    # Coming back from the rack he turns to face the bench first, part held in front of
    # him, other hand at his side, and only then walks. Blending the turn and the walk
    # into one move swung the free hand in an arc over the jig's back edge, 6 mm from
    # the robot's shoulder (the rack faces the other way from the bench).
    turned = (-0.32, -0.64, np.pi / 2 + 0.15)
    held_in_front = (turned[0] - 0.12, turned[1] + 0.26, 1.24)
    turned_side_right = (turned[0] + 0.245, turned[1] + 0.045, 0.86)
    # Both rack stances are in front of the bench. The cover tote's stance was
    # (-0.55, -0.20), 25 cm inside the bench's footprint (the top runs to x = -0.8,
    # y = -0.4): he stood in the bench end, unnoticed since gate 2 because he does not
    # collide with it, and his shoulder there sat 36 cm from the first rail screw. The
    # rack moved 30 cm toward him to make the far totes reachable from here.
    at_rack_low = (-0.55, -0.66, np.pi + 0.15)      # for the base and rail totes
    at_rack_high = (-0.60, -0.50, np.pi - 0.35)     # for the cover tote further along
    base_pick = scene.points["pick_base"]
    rail_pick = scene.points["pick_rail"]
    cover_pick = scene.points["pick_cover"]
    # hands rest at the front edge of the worktop, clear of the jig and of the box
    # in it. Resting them closer in made the hand sweep through the box on the way out.
    left_rest = (-0.28, -0.37, zw + 0.06)
    right_rest = (0.26, -0.36, zw + 0.06)
    # the grips of the jig's two toggle clamps, on its right side (station.build_jig_and_parts)
    w, d_box, _ = L.box_size
    clamp_x = jx + w / 2 + 0.045 + 0.027
    clamp_z = zw + L.jig_plate[2] + 0.09
    front_clamp = (clamp_x, jy - 0.045, clamp_z)
    rear_clamp = (clamp_x, jy + 0.045, clamp_z)

    def at_rack(name, seconds, stance, target=None, lean=0.20, gaze=None, **kw):
        """At the rack the right arm hangs: the bench is out of its reach from there."""
        return Segment(name, seconds, left=target, right=None, lean=lean, gaze=gaze,
                       left_fingers=(-0.6, 0.8, 0.0), stance=stance, **kw)

    return [
        at_rack("step to the rack", 2.2, at_rack_low, lean=0.14, gaze=base_pick),
        at_rack("take a base from the tote", 1.8, at_rack_low, base_pick, gaze=base_pick),
        Segment("turn with the base", 2.2, left=held_in_front, right=turned_side_right, lean=0.16,
                gaze=over_jig, stance=turned, lift=(0.12, 0.5),
                grab=(("part_base", "l"),), snap={"part_base": 0.075}),
        Segment("carry the base to the bench", 1.6, left=over_jig, right=right_rest, lean=0.20,
                gaze=over_jig, stance=at_bench),
        Segment("set the base in the jig", 1.6, left=(jx - 0.06, jy - 0.02, zw + 0.12), right=right_rest,
                lean=0.30, gaze=(jx, jy, zw + 0.05), stance=at_bench,
                release=(("part_base", "box_home"),)),
        at_rack("step back for a rail", 3.4, at_rack_low, lean=0.14, gaze=rail_pick, lift=(0.10, 0.4)),
        at_rack("take a rail", 1.9, at_rack_low, rail_pick, gaze=rail_pick),
        Segment("turn with the rail", 2.2, left=held_in_front, right=turned_side_right, lean=0.16,
                gaze=(jx, jy, zw + 0.08), stance=turned, lift=(0.12, 0.5),
                grab=(("part_rail", "l"),), snap={"part_rail": 0.04}),
        Segment("fit the rail in the box", 1.8, left=(jx - 0.01, jy - 0.01, zw + 0.15), right=right_rest,
                lean=0.28, gaze=(jx, jy, zw + 0.08), stance=at_bench),
        Segment("press the rail down", 1.4, left=(jx + 0.02, jy + 0.01, zw + 0.12), right=right_rest,
                lean=0.30, gaze=(jx, jy, zw + 0.08), stance=at_bench,
                release=(("part_rail", "rail_home"),)),
        # the robot drives the rail screws now. He preps the next rail beside it
        Segment("step over to the prep area", 2.4, left=prep_left, right=prep_right, lean=0.14, gaze=prep,
                stance=at_prep, lift=(0.10, 0.5)),
        Segment("clip blocks on the next rail", 3.4, left=(prep[0] - 0.02, prep[1] - 0.02, prep[2]),
                right=(prep[0] + 0.08, prep[1] - 0.04, prep[2]), lean=0.30, gaze=prep, stance=at_prep),
        Segment("clip the rest of the blocks", 3.4, left=(prep[0] + 0.04, prep[1] - 0.03, prep[2]),
                right=(prep[0] + 0.10, prep[1] - 0.05, prep[2]), lean=0.30, gaze=prep, stance=at_prep),
        # the arms come down to his sides before he turns to the rack: turning with them
        # out swung his right hand behind the jig into the root of the robot's arm. Down
        # by targets, not by hanging: the blend from reaching to the hanging pose swung
        # the right hand 45 cm sideways over the jig at 1.9 m/s
        Segment("drop the hands", 1.6, left=at_side_left, right=at_side_right, lean=0.14, gaze=cover_pick,
                stance=at_prep),
        at_rack("step across for a cover", 3.0, at_rack_high, lean=0.14, gaze=cover_pick),
        at_rack("take a cover", 1.7, at_rack_high, cover_pick, gaze=cover_pick),
        Segment("turn with the cover", 2.0, left=held_in_front, right=turned_side_right, lean=0.16,
                gaze=over_jig, stance=turned, lift=(0.12, 0.5),
                grab=(("part_cover", "l"),), snap={"part_cover": 0.03}),
        Segment("bring the cover to the box", 1.6, left=over_jig, right=right_rest, lean=0.22,
                gaze=over_jig, stance=at_bench),
        Segment("press the cover on", 1.8, left=(jx - 0.05, jy - 0.03, cover_z + 0.03), right=right_rest,
                lean=0.32, gaze=(jx, jy, cover_z), stance=at_bench,
                release=(("part_cover", "cover_home"),)),
        Segment("close the front clamp", 1.3, left=left_rest, right=front_clamp, lean=0.24,
                gaze=front_clamp, stance=at_bench, lift=(0.08, 0.4)),
        Segment("close the rear clamp", 0.9, left=left_rest, right=rear_clamp, lean=0.26,
                gaze=rear_clamp, stance=at_bench, signal="cover_ready"),
        # the robot drives the cover screws now. He finishes the next rail beside it
        Segment("step over to the prep area", 1.8, left=prep_left, right=prep_right, lean=0.14, gaze=prep,
                stance=at_prep, lift=(0.08, 0.4)),
        Segment("fit the end stops", 3.0, left=(prep[0] - 0.07, prep[1] - 0.01, prep[2]),
                right=(prep[0] + 0.06, prep[1] - 0.03, prep[2]), lean=0.30, gaze=prep, stance=at_prep),
        Segment("fit the jumper bars", 3.0, left=(prep[0] - 0.03, prep[1] - 0.02, prep[2]),
                right=(prep[0] + 0.09, prep[1] - 0.04, prep[2]), lean=0.30, gaze=prep, stance=at_prep),
        Segment("label the blocks", 3.4, left=(prep[0] + 0.02, prep[1] - 0.03, prep[2]),
                right=(prep[0] + 0.11, prep[1] - 0.05, prep[2]), lean=0.30, gaze=prep, stance=at_prep),
        # he waits with his hands at his sides: on the bench edge they were 26 cm from
        # the front cover holes, inside the planner's margin, and it waited 15 s for nothing
        Segment("drop the hands", 1.6, left=at_side_left, right=at_side_right, lean=0.14, gaze=watch,
                stance=at_prep),
        Segment("step back from the prep area", 1.8, left=bench_side_left, right=bench_side_right, lean=0.04,
                gaze=watch, stance=at_wait),
        # long enough for the robot's four cover screws. A real cell ends this on the
        # stack light; the script cannot, so the time is a scene choice, stated in
        # MODEL_NOTES, and a robot that is slower loses the rest of the screws
        Segment("wait for the green light", WAIT_FOR_COVER_SCREWS_S, left=bench_side_left, right=bench_side_right,
                lean=0.04, gaze=watch, stance=at_wait),
        Segment("step in and open the clamps", 2.6, left=left_rest, right=front_clamp, lean=0.24,
                gaze=front_clamp, stance=at_bench, signal="done"),
        Segment("lift the finished box", 2.6, left=left_rest, right=(jx + 0.06, jy - 0.04, zw + 0.16),
                lean=0.26, gaze=(jx, jy, cover_z), stance=at_bench, lift=(0.08, 0.4),
                grab=(("part_base", "r"), ("part_rail", "r"), ("part_cover", "r"))),
        Segment("put it in the tray", 2.6, left=left_rest, right=tray, lean=0.22, gaze=tray,
                stance=at_bench,
                release=(("part_base", "tray"), ("part_rail", "tray"), ("part_cover", "tray"))),
        Segment("back to rest", 2.2, left=left_rest, right=right_rest, lean=0.18,
                gaze=(jx, jy, zw + 0.10), stance=at_bench, lift=(0.10, 0.4)),
    ]


@dataclass
class Playback:
    """Key frames and part handling for one cycle, ready to step through."""
    times: list        # one per interpolation key, including lift-off via poses
    qpos_keys: list
    segments: list
    worker_qadr: np.ndarray
    free_adr: int
    seg_times: list = field(default_factory=list)  # segment boundaries, len(segments) + 1
    duration: float = 0.0
    dt: float = 0.002
    reach_errors_mm: list = field(default_factory=list)


def _worker_addresses(m: mujoco.MjModel):
    free = m.joint("worker_free")
    adr = [free.qposadr[0] + i for i in range(7)]
    for j in range(m.njnt):
        name = m.joint(j).name
        if name.startswith(station.HUMAN_PREFIX):
            adr.append(int(m.jnt_qposadr[j]))
    return np.array(adr), int(free.qposadr[0])


def _part_pose(m: mujoco.MjModel, d: mujoco.MjData, part: str) -> np.ndarray:
    adr = m.joint(part + "_free").qposadr[0]
    return d.qpos[adr:adr + 7].copy()


def _set_part_pose(m: mujoco.MjModel, d: mujoco.MjData, part: str, pose7) -> None:
    adr = m.joint(part + "_free").qposadr[0]
    d.qpos[adr:adr + 7] = pose7


def parts_to_start(scene: station.Scene, d: mujoco.MjData) -> None:
    """Put the three parts in their totes, so the cycle starts with an empty jig."""
    m = scene.model
    for part, key, lift in (("part_base", "pick_base", 0.0), ("part_rail", "pick_rail", 0.0),
                            ("part_cover", "pick_cover", 0.0)):
        if key not in scene.points:
            continue
        p = np.asarray(scene.points[key], float) + [0.0, 0.0, lift]
        _set_part_pose(m, d, part, np.concatenate([p, [1.0, 0.0, 0.0, 0.0]]))
    mujoco.mj_forward(m, d)


def build_playback(scene: station.Scene, d: mujoco.MjData, segments=None) -> Playback:
    """Solve the key pose at the end of every segment.

    Every key pose starts from the standing rest pose at that segment's stance:
    continuing from the previous pose let a failed reach poison the next one.
    """
    m = scene.model
    segments = segments or cycle(scene)
    poser = pose.WorkerPoser(m, d)
    qadr, free_adr = _worker_addresses(m)
    from sightline_sim.human import RIGHT_ARM_REACH

    default_stance = (scene.layout.worker_xy[0], scene.layout.worker_xy[1], np.pi / 2)
    poser.stand(default_stance[:2], default_stance[2])
    poser.lean(0.18)
    poser.settle_on_floor()
    good: dict = {}      # each arm's joints after its last reach that landed

    def arm(side: str) -> dict:
        return {f"{side}{n}": poser.get(f"{side}{n}") for n in pose.ARM_CHAIN}

    def from_hanging(side: str, target, fingers) -> float:
        """The last resort: start with the arm hanging and solve the position alone. A
        target down at his side is a short solve from there; asked to point the fingers
        forward as well, the wrist ran into a joint limit and the hand stopped 16 cm short."""
        for n in pose.ARM_CHAIN:
            poser.set(f"{side}{n}", 0.0)
        poser.seed_arm({k: v for k, v in pose._arms_at_rest().items() if k.startswith(side)})
        return poser.reach(side, target, palm_down=False)

    def solve(seg, left, right):
        """Stand where the segment says, reach both hands, look. Returns the errors in m."""
        stance = seg.stance or default_stance
        poser.stand(stance[:2], stance[2], hang=[s for s, target in (("l", left), ("r", right)) if target is None])
        poser.lean(seg.lean)
        err = {}
        if left is not None:
            err["l"] = poser.reach("l", left, finger_dir=seg.left_fingers)
            if err["l"] > 0.001 and "l" in good:
                # a hanging arm is a poor starting guess: start again from its last good reach
                poser.seed_arm(good["l"])
                err["l"] = poser.reach("l", left, finger_dir=seg.left_fingers)
            if err["l"] > 0.001:
                err["l"] = from_hanging("l", left, seg.left_fingers)
            if err["l"] <= 0.001:
                good["l"] = arm("l")
        if right is not None:
            # solve from the rest pose first. Seeding the arm in a reaching shape solves
            # far targets, but for a target near the body it leaves the elbow out, and the
            # blend into the next key then swings the hand in a wide arc over the bench.
            before = d.qpos.copy()
            err["r"] = poser.reach("r", right, finger_dir=seg.right_fingers)
            if err["r"] > 0.001:
                d.qpos[:] = before  # a failed solve leaves the arm folded up, a bad seed
                poser.seed_arm(RIGHT_ARM_REACH)
                err["r"] = poser.reach("r", right, finger_dir=seg.right_fingers)
            if err["r"] > 0.001 and "r" in good:
                d.qpos[:] = before
                poser.seed_arm(good["r"])
                err["r"] = poser.reach("r", right, finger_dir=seg.right_fingers)
            if err["r"] > 0.001:
                d.qpos[:] = before
                err["r"] = from_hanging("r", right, seg.right_fingers)
            if err["r"] <= 0.001:
                good["r"] = arm("r")
        if seg.gaze is not None:
            poser.look_at(seg.gaze)
        poser.settle_on_floor()
        return err

    keys = [d.qpos[qadr].copy()]
    times = [0.0]
    seg_times = [0.0]
    errors = []
    t = 0.0
    prev = None
    for seg in segments:
        if seg.lift is not None and prev is not None:
            dz, secs = seg.lift
            up = lambda p: None if p is None else (p[0], p[1], p[2] + dz)
            # the via pose belongs to where the hands were, so it uses the last stance
            solve(prev, up(prev.left), up(prev.right))
            t += secs
            times.append(t)
            keys.append(d.qpos[qadr].copy())
            rest = max(seg.duration - secs, 0.2)
        else:
            rest = seg.duration
        err = solve(seg, seg.left, seg.right)
        errors.append({"segment": seg.name, **{k: round(v * 1000, 1) for k, v in err.items()}})
        t += rest
        times.append(t)
        seg_times.append(t)
        keys.append(d.qpos[qadr].copy())
        prev = seg
    pb = Playback(times, keys, segments, qadr, free_adr, seg_times=seg_times, duration=t)
    pb.reach_errors_mm = errors
    return pb


def _slerp(q0, q1, s: float) -> np.ndarray:
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    if q0 @ q1 < 0:
        q1 = -q1
    dot = float(np.clip(q0 @ q1, -1.0, 1.0))
    if dot > 0.9995:
        out = q0 + s * (q1 - q0)
    else:
        theta = np.arccos(dot)
        out = (np.sin((1 - s) * theta) * q0 + np.sin(s * theta) * q1) / np.sin(theta)
    return out / np.linalg.norm(out)


def worker_state(pb: Playback, t: float) -> np.ndarray:
    """Worker joint vector at time t, blended with a minimum jerk profile."""
    i = int(np.searchsorted(pb.times, t, side="right")) - 1
    i = max(0, min(i, len(pb.qpos_keys) - 2))
    span = pb.times[i + 1] - pb.times[i]
    s = min_jerk(min((t - pb.times[i]) / span, 1.0) if span > 0 else 1.0)
    a, b = pb.qpos_keys[i], pb.qpos_keys[i + 1]
    out = a + s * (b - a)
    out[3:7] = _slerp(a[3:7], b[3:7], s)
    return out


def segment_index(pb: Playback, t: float) -> int:
    bounds = pb.seg_times or pb.times
    i = int(np.searchsorted(bounds, t, side="right")) - 1
    return max(0, min(i, len(pb.segments) - 1))


def run(scene: station.Scene, d: mujoco.MjData, pb: Playback, on_frame=None, fps: float = 30.0,
        on_step=None, after_step=None, duration: float | None = None) -> dict:
    """Play the cycle. on_frame(t, index, segment) is called at the frame rate, after mj_forward.

    on_step(t) runs every step before mj_forward, which is where another controller
    writes its own joints. after_step(t) runs every step once the state is current,
    which is where a judge looks. duration runs the clock past the end of the cycle,
    with the worker held at his last pose, so a slower robot can finish.

    Returns the measurements gate 2 asks for: hand speeds and accelerations, how
    far the hands push into anything, and how much the feet slide.
    """
    m = scene.model
    hands = {side: m.body(f"{station.HUMAN_PREFIX}{side}hand").id for side in ("l", "r")}
    hand_geoms = {g for g in range(m.ngeom)
                  if m.body(m.geom_bodyid[g]).name.startswith(
                      tuple(station.HUMAN_PREFIX + p for p in ("lhand", "rhand", "lfingers", "rfingers", "lthumb", "rthumb")))}
    foot_geoms = [g for g in range(m.ngeom)
                  if m.body(m.geom_bodyid[g]).name.startswith(
                      tuple(station.HUMAN_PREFIX + p for p in ("lfoot", "rfoot", "ltoes", "rtoes")))
                  and m.geom_group[g] == station.GROUP_COLLISION]
    homes = {"box_home": scene.points["box_home"], "rail_home": scene.points["rail_home"],
             "cover_home": scene.points["cover_home"],
             "tray": (scene.layout.tray_xy[0], scene.layout.tray_xy[1], scene.worktop_z + 0.006)}

    held: dict = {}
    fixed: dict = {}
    for part in PARTS:
        try:
            m.joint(part + "_free")
        except KeyError:
            continue
        fixed[part] = _part_pose(m, d, part)

    steps = int(round((pb.duration if duration is None else duration) / pb.dt))
    frame_every = max(1, int(round(1.0 / (fps * pb.dt))))
    prev_pos = {side: None for side in hands}
    prev_speed = {side: 0.0 for side in hands}
    peak_speed = {side: 0.0 for side in hands}
    peak_accel = {side: 0.0 for side in hands}
    worst_push = 0.0
    worst_push_what = ""
    worst_push_when = ""
    pushes: dict = {}   # what the hands clip in transit, and how deep, over the whole cycle
    foot_xy = []
    frame_index = 0
    prev_idx = None

    for step in range(steps + 1):
        t = step * pb.dt
        d.qpos[pb.worker_qadr] = worker_state(pb, t)
        idx = segment_index(pb, t)
        if idx != prev_idx:
            mujoco.mj_forward(m, d)
            if prev_idx is not None:
                for part, home in pb.segments[prev_idx].release:
                    held.pop(part, None)
                    fixed[part] = _home_pose(m, d, part, homes[home], home)
            for part, hand in pb.segments[idx].grab:
                fixed.pop(part, None)
                drop = pb.segments[idx].snap.get(part)
                if drop is not None:  # a part taken from a tote snaps into the hand
                    hp = d.xpos[hands[hand]]
                    _set_part_pose(m, d, part, np.concatenate([hp + [0.0, 0.0, -drop], [1.0, 0.0, 0.0, 0.0]]))
                    mujoco.mj_forward(m, d)
                held[part] = (hand, _hold_offset(m, d, part, hands[hand]))
            prev_idx = idx
        for part, (hand, offset) in held.items():
            _set_part_pose(m, d, part, _carried_pose(m, d, hands[hand], offset))
        for part, pose7 in fixed.items():
            _set_part_pose(m, d, part, pose7)
        if on_step is not None:
            on_step(t)
        mujoco.mj_forward(m, d)
        if after_step is not None:
            after_step(t)

        for side, bid in hands.items():
            p = d.xpos[bid].copy()
            if prev_pos[side] is not None:
                speed = float(np.linalg.norm(p - prev_pos[side])) / pb.dt
                peak_speed[side] = max(peak_speed[side], speed)
                peak_accel[side] = max(peak_accel[side], abs(speed - prev_speed[side]) / pb.dt)
                prev_speed[side] = speed
            prev_pos[side] = p
        for i in range(d.ncon):
            g1, g2 = int(d.contact.geom[i][0]), int(d.contact.geom[i][1])
            if (g1 in hand_geoms) != (g2 in hand_geoms):
                other = g2 if g1 in hand_geoms else g1
                name = m.body(m.geom_bodyid[other]).name
                if name.startswith(station.HUMAN_PREFIX) or name in held:
                    continue
                dist = float(d.contact.dist[i])
                pushes[name] = min(pushes.get(name, 0.0), dist)
                if dist < worst_push:
                    worst_push = dist
                    worst_push_what = name
                    worst_push_when = pb.segments[idx].name
        if step % frame_every == 0:
            foot_xy.append(np.array([d.geom_xpos[g][:2] for g in foot_geoms]))
            if on_frame is not None:
                on_frame(t, frame_index, pb.segments[idx])
            frame_index += 1
    for part, home in pb.segments[-1].release:
        held.pop(part, None)
        fixed[part] = _home_pose(m, d, part, homes[home], home)

    foot_xy = np.array(foot_xy)
    slide = float(np.max(np.linalg.norm(foot_xy - foot_xy[0], axis=2))) if len(foot_xy) else 0.0
    return {
        "cycle_s": round(pb.duration, 2),
        "peak_hand_speed_m_s": {k: round(v, 3) for k, v in peak_speed.items()},
        "peak_hand_accel_m_s2": {k: round(v, 1) for k, v in peak_accel.items()},
        "worst_hand_push_mm": round(-worst_push * 1000, 2),
        "worst_hand_push_into": worst_push_what,
        "worst_hand_push_during": worst_push_when,
        "hand_push_mm": {k: round(-v * 1000, 1) for k, v in sorted(pushes.items(), key=lambda kv: kv[1])},
        "foot_slide_mm": round(slide * 1000, 2),
        "frames": frame_index,
    }


def _hold_offset(m, d, part: str, hand_body: int) -> np.ndarray:
    """Part pose in the hand frame at the moment it is picked up."""
    pose7 = _part_pose(m, d, part)
    hp, hR = d.xpos[hand_body], d.xmat[hand_body].reshape(3, 3)
    pos = hR.T @ (pose7[:3] - hp)
    quat = np.zeros(4)
    hq = np.zeros(4)
    mujoco.mju_mat2Quat(hq, hR.flatten())
    mujoco.mju_negQuat(hq, hq)
    mujoco.mju_mulQuat(quat, hq, pose7[3:7])
    return np.concatenate([pos, quat])


def _carried_pose(m, d, hand_body: int, offset: np.ndarray) -> np.ndarray:
    hp, hR = d.xpos[hand_body], d.xmat[hand_body].reshape(3, 3)
    hq = np.zeros(4)
    mujoco.mju_mat2Quat(hq, hR.flatten())
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, hq, offset[3:7])
    return np.concatenate([hp + hR @ offset[:3], quat])


def _home_pose(m, d, part: str, home_xyz, home_name: str) -> np.ndarray:
    """Where a released part lands: its home point, keeping the height it was built at."""
    pose7 = _part_pose(m, d, part).copy()
    pose7[:3] = np.asarray(home_xyz, float)
    pose7[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    if home_name == "tray":
        base = {"part_base": 0.0, "part_rail": 0.021, "part_cover": 0.082}
        pose7[2] += base.get(part, 0.0)
    return pose7
