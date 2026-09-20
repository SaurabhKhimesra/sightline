# SightLine: scope, aim and rules

Working name. Written 2026-09-17, after CobotSafe v0.1 (my earlier project on
what happens when a robot that cannot see people runs a cell). This file is the agreed
design; every change since is logged in section 19.

## 1. Why

CobotSafe v0.1 measured what happens when a robot that cannot see people runs a
fixed program next to one: every row of its headline table fails ISO/TS 15066.
The industrial people I showed its video to all said the same thing: a
real robot should see the person, move around them and keep working. This
project builds that in simulation and measures it.

## 2. Aim

A person and a UR5e build a control box together at one screwdriving station.
The person places parts, the robot drives the screws. The robot plans its motion
in real time from its own cameras, and two rules are hard:

- **R1. Never touch the person.**
- **R2. Never block the eyes camera's view of the person.**

When the person is in the way of what the robot needs to see or reach, the robot
first looks around, the way a person moves their head: it turns around the
screwdriver bit, changes its arm posture or picks another screw. It waits only
when nothing works.

For a normal person: *the robot watches the worker, moves around them, never
blocks its own view of them, and keeps screwing.*

## 3. What it is and what it is not

It is:
- a simulation study of vision-based, real-time, person-aware motion planning on
  a real manufacturing task;
- measured by an independent judge (CobotSafe) against ground truth the planner
  never sees;
- a demo video cut from real closed-loop runs.

It is not:
- a safety function. The cameras are not safety-rated sensors, the planner is not
  a safety-rated controller, and nothing here replaces a risk assessment or a
  validated speed and separation monitoring system;
- a claim about real hardware. Results hold only for the modelled worker, sensors
  and robot;
- certification of anything.

## 4. The two hard rules, exactly

"Hard" has to be testable, so each rule has a planner side and a judge side.

### R1. Never touch the person

- **Judge:** at every physics step, the distance between every robot geom (arm,
  screwdriver, wrist camera, the screw on the bit) and every human geom is above
  zero. Report the minimum distance per run.
- **Every contact counts.** Each one is labelled with who moved in: put the robot
  back where it was one motion cycle (10 ms) earlier and keep the person where
  they are now. If the contact is still there, the person moved in. Otherwise the
  robot did.
- A contact's force is measured the CobotSafe way (section 10) and reported
  against the ISO/TS 15066 limits for information. The run has broken R1 whatever
  the force.
- **Planner:** hard constraints in every motion cycle (section 8.5) and an
  independent separation monitor (section 8.6).

### R2. Never block the eyes camera's view of the person

- **Judge:** for every eyes camera frame, render a segmentation image with the
  robot and one with the robot hidden. A pixel that shows the person without the
  robot and the robot with it is blocked. The rule holds when every frame has zero
  blocked pixels.
- Blocked frames get the same who-moved-in label as contacts, with one change
  decided 2026-09-18: the robot is put back one camera frame, not one motion cycle.
  Ten milliseconds is 2.5 mm of arm travel, and a camera only catches an overlap
  once it is wider than that, so the 10 ms test called 15 of 16 events the person's
  doing when the video showed the robot arriving. Both labels are still recorded.
  The frame was 33 ms at 30 Hz and is 40 ms since the cameras went to 25 Hz
  (section 19, gate 5).
- Only the camera's own view counts. Body parts the camera cannot see anyway (legs
  under the bench, a hand behind the worker's own body) are outside R2.
- **Planner:** hard constraints keep every robot link out of the sight lines from
  the camera to the person, with a margin (section 8.5). The margin also covers
  the space the person can reach before the robot could clear the view, so a hand
  moving no faster than the assumed speed cannot get into a blind spot the robot
  made.

### What no rule can promise

A person faster than the speed the margins assume, or one who walks into a robot
that has already stopped. That is why every violation carries its who-moved-in
label and the person's speed at the time, and why the worker's own peak speeds are
reported next to the speed the planner assumes.

## 5. The task: building a control box

The product is a scene choice, not a real part number.

Parts:
- base: plastic enclosure about 200 × 150 × 80 mm, two threaded bosses for the
  rail, four corner bosses for the cover;
- a 35 mm top-hat rail, 160 mm long, two screw slots, pre-fitted by the worker
  with 4 terminal blocks;
- cover with 4 corner holes;
- 6 screws: 2 for the rail, 4 for the cover.

One cycle:
1. Worker: base from its bin into the jig. The jig is loose: the base can sit up
   to ±3 mm and ±3° off (scene choice), so the robot has to find it by vision.
2. Worker: pre-fitted rail onto the bosses, often steadying it with one hand for
   a moment.
3. Robot: 2 rail screws.
4. Worker, at the same time: next rail and 4 terminal blocks from the bins,
   clipped together in the prep area, sometimes reaching across the shared zone
   for a part.
5. Worker, when the stack light says the rail screws are done: cover onto the
   base, pressing it down and often keeping a hand on it.
6. Robot: 4 cover screws, order chosen live.
7. Worker, when the stack light says the box is done: box to the out tray, next
   base into the jig.

The robot fetches every screw from the feeder, so it crosses the shared zone 6
times per box.

## 6. The station

Everything here is a scene choice unless a source is given. Make it look like a
real cell.

Layout (world frame: z up, origin on the floor under the bench centre, +y from the
worker toward the robot):
- **Floor:** grey epoxy, yellow and black tape around the shared area.
- **Bench:** aluminium profile frame, worktop 1.6 × 0.8 m with an ESD mat.
  Height set from the humanoid's measured elbow height: **1.118 m**, which is the
  measured elbow height 1.193 m minus 7.5 cm (CCOHS, "Working in a Standing
  Position", updated 2022-11-30: light work sits 5 to 10 cm below elbow height;
  the middle of that band is a scene choice). A top frame 1.1 m above the worktop
  carries the LED light bars, the stack light and the station screen (2026-09-18).
- **Robot:** UR5e from MuJoCo Menagerie on the worktop at the rear centre (confirm
  with the reachability map in gate 1).
- **Screwdriver:** the OnRobot Screwdriver's shape, from its datasheet v1.7
  (p. 2 and the drawing on p. 34), without branding: 322 mm from the top to the
  bit holder tip, body 86 × 114.1 mm, nose Ø49 mm reaching 60.4 mm below the body,
  bit holder 13.5 mm, robot flange on the **back face** with its centre 166.6 mm
  above the tip and the screw axis about 93.5 mm off the flange face. The side
  mount is the important part: turning around the screw moves the whole arm, not
  just the wrist, so reach depends on the turn angle (2026-09-18).
- **Wrist camera:** Intel RealSense D405 (87 × 58°, 42 × 42 × 23 mm, ideal range
  7 to 50 cm, Intel product specification) on a bracket 75 mm off the screw axis
  and 52 mm above the bit tip, aimed 35 mm below the tip so the nose does not hide
  the hole. The offset is what makes looking around work: turning around the screw
  swings the camera around the hole (2026-09-18).
- **Eyes camera:** Intel RealSense D455 (86 × 57°, 124 × 29 × 26 mm, range 0.6 to
  6 m, Intel product specification) on a pole beside the bench at x 1.15 m,
  y -0.90 m, 3.25 m above the floor, looking down at 56° across the bench. It must
  see the worker's hands and arms anywhere over the worktop, the head and
  shoulders when the worker leans in, and the whole shared zone. The position was
  chosen in gate 1 by comparing three mounts (results.md): from the front of
  the cell the worker's own head and back hide the box, so the camera looks in
  from the side instead (2026-09-18).
- **Jig:** aluminium nest plate with side stops and two toggle clamps (clamps
  visual only).
- **Screw feeder:** presenter type on the robot side, a screw ready at a fixed pick
  point.
- **Bins and trays:** 4 parts bins on a sloped rail on the worker's left, out tray
  on the right, prep area in front of the worker.
- **Stack light:** tells the worker when the robot has finished a step.
- **Parts with real holes:** meshes generated in numpy, so the wrist camera sees
  actual holes. Holes are visual only; collision uses simple shapes. Hole
  interiors get a dark material so they read as holes on camera (a rendering
  choice; notes.md says so).
- **Screws:** placed geoms shown when picked or driven. A screw on the bit is part
  of the robot for both rules.
- **Worker:** the dm_control CMU humanoid (CMUHumanoidPositionControlledV2020), the
  model CobotSafe uses, so CobotSafe's body regions and limits apply. As shipped it
  stands 2.05 m, so it is scaled by 0.8531 to **1.75 m**, a height inside the
  published range for adult men (NCD-RisC 2016). Densities are scaled with it, so
  the mass stays 70 kg (2026-09-18). Work clothes colours (blue jacket, dark
  trousers), safety glasses, gloves.
- **Lighting:** warm white key light from the light bar, soft fill, shadows on.
- **Film cameras:** hero three-quarter view, side, top, plus the robot's two camera
  feeds as picture-in-picture.

The robot knows everything that does not move (bench, frame, jig, feeder, bins),
like a real cell's CAD. It does not know the worker, the parts the worker carries,
or where the box sits in the jig.

## 7. The worker

- **Kinematic playback.** Every physics step writes the humanoid's joint positions
  and velocities from the script. No balance or gait controller, so nothing can
  fall or fly.
- Poses come from IK over both arms, trunk lean and head gaze, with minimum-jerk
  timing between key poses (Flash and Hogan 1985). CobotSafe's `reach.py` is a
  starting point. Trunk, neck and head are free here; CobotSafe braced them only
  because its physics humanoid folded.
- Feet stay planted: small weight shifts and single side steps only.
- Carried parts are attached to the hand while held.
- Human geoms collide only with robot geoms (contype and conaffinity bits), so the
  body does not fight the bench. Gate 2 checks hands against parts and bench with
  a penetration number.
- **The worker ignores the robot's motion completely.** It reacts only to task
  state (the stack light). This is the strictest test and needs no politeness
  assumptions.
- Every episode is drawn from a seeded generator: timing ±20 %, reach speed scale,
  which hand, where a steadying hand rests on the cover (sometimes within 3 cm of
  a hole), how long it rests, and occasional extras: leaning in to look, reaching
  across the jig, resting both hands on the bench edge.
- Stress scripts on top: a hand resting next to the next hole, a reach across the
  jig at the generator's top speed, the head leaning over the jig, the worker
  lifting the box while the robot approaches a screw.
- Reported per episode: peak hand speed and peak hand acceleration.
- **The jig's toggle clamps hold the cover while the robot screws it** (decided with
  decided 2026-09-18, section 19). He closes the clamps, which is the jig switch the
  robot starts the cover screws on, preps the next rail at the prep area, then waits
  upright at the bench until the screws are in and opens the clamps. The cycle is
  about 70 s instead of 44 s: four cover screws take the robot about 22 s.

## 8. The robot

### 8.1 What the planner may see, enforced in code

The planner may use only:
- eyes camera colour and depth images (with noise and delay);
- wrist camera colour images;
- its own joint positions and velocities;
- the static station model;
- its own robot model (for self-filtering and collision capsules);
- the product model (holes relative to the part, not the part's pose);
- the task state it keeps itself.

It never gets human geometry or state from the simulator, the worker script or
its seed, segmentation images, the true box pose, or anything from the future.

In code: the planner package imports nothing from the simulation or judge
packages and receives a `SensorFrame`. A test fails if such an import appears. A
replay test feeds a logged sensor stream back in and must reproduce the logged
commands exactly.

### 8.2 Sensors

- **Eyes camera:** RealSense D455, 640 × 480 colour and depth at 25 Hz with one frame of delay
  (a design choice, 2026-09-18; it was 30 Hz). Depth noise from Nguyen, Izadi and Lovell 2012: axial
  σ_z = 0.0012 + 0.0019 (z − 0.4)² m for surface angles 10° to 60° (their Eq. 3;
  their Eq. 4 adds a term for steeper angles), lateral
  σ_L = 0.8 + 0.035·θ/(π/2 − θ) px (their Eq. 1). Fitted on a Kinect between 0.5
  and 2.75 m; used here as a stand-in for an RGB-D camera, and notes.md says
  so. Missing depth at object edges (scene choice).
- **Wrist camera:** RealSense D405, 640 × 480 colour at 25 Hz, the same as the eyes
  camera (it was 30 Hz).
- **Joint state:** every 2 ms, exact. A UR e-Series controller runs its real-time
  loop at 500 Hz (UR RTDE guide).
- **Robot limits:** UR5e maximum joint speed ±180°/s on all six joints, maximum
  TCP speed 4 m/s, reach 850 mm (UR5e technical specification, updated May 2025).
  The planner uses lower, stated limits near the person.

### 8.3 Perception

Eyes pipeline, every frame:
1. Depth to points in the world frame, using the camera calibration with a small
   stated error.
2. Remove the background, using a depth map of the empty station taken at start.
3. Robot self-filter: render the robot's own model at its current joint angles
   from the eyes camera and remove points close to it. Real cells do this with
   the robot's URDF and joint state.
4. Workpiece filter: remove the box parts the robot has already localized, the
   same way.
5. Everything left is "maybe the person". A loose part left on the bench counts
   as person too. That is conservative, and the extra waiting it causes is
   measured.
6. Person voxels (2 cm, scene choice), each with a velocity from frame-to-frame
   tracking.
7. Unseen space behind the person: voxels behind observed person points along each
   camera ray, up to a stated depth, count as occupied for R1 (the depth space
   idea of Flacco et al. 2012).
8. Box pose: fit the box's top face and edges in the depth points. Coarse, a few
   mm.

Wrist pipeline, near a hole:
1. Predict where the hole is in the image from the coarse box pose.
2. Find the dark ellipse in a window around the prediction with numpy
   thresholding and image moments (no OpenCV).
3. Accept it only if its area, shape and position fit the prediction. Otherwise
   the hole is "not visible".
4. Predict blocking before it happens: if person voxels cut the line from the
   wrist camera to the hole, the view counts as blocked.

### 8.4 Layers

| layer | rate (scene choice) | job |
|---|---|---|
| task | 5 Hz | which screw next, and the mode: WORK, LOOK AROUND, REORDER, WAIT, PARK |
| look around | 10 Hz | pick the approach pose for the chosen screw |
| motion | 100 Hz | velocity QP with the hard constraints, visual servoing near the hole |
| joint servo | 500 Hz | integrate the QP's velocities into targets for the Menagerie position actuators |
| monitor | 500 Hz | independent separation check and protective stop |

**Task layer.** Scores each remaining hole by how clear it is (distance to the
person, predicted blocking, whether any approach passes both rules) and how far
away it is. Hysteresis, so it does not flip between screws.

**Look-around layer.** The screw fixes the bit axis, which uses 5 of the 6 axes.
The sixth, turning around the bit, is free. With the side-mounted screwdriver a
turn moves the whole arm, so each turn angle is a different arm pose and reach has
to be checked per angle (2026-09-18). Candidates: every 15° of turn (scene
choice) × IK branches (elbow up or down, wrist flip) × two stand-off heights. Each
candidate is checked for joint limits and reach, R1 clearance of the whole arm
with margin, R2 clearance of the whole arm with margin, a clear line from the
wrist camera to the hole, and how far the joints have to move. Switch only when
the current pose fails or another is clearly better.
- Triggers: wrist view of the hole lost or predicted blocked; the current approach
  breaks R1 or R2; no progress for a stated time (stuck).
- Responses, in order: turn around the bit, change posture, another screw, back
  off to a stand-off that keeps both rules and wait, park.
- Every trigger and response is logged.

### 8.5 Motion layer: the per-cycle QP

- Variables: the 6 joint velocities. Solve it in numpy with the dual active-set
  method of Goldfarb and Idnani 1983, tested by checking the optimality conditions
  on random problems.
- Objective: follow the task velocity (move toward the chosen pose, or visual
  servoing near the hole), plus smoothness.
- **Hard constraints, never relaxed** (only the task term gets slack):
  - joint position, velocity and acceleration limits;
  - **R1:** for each robot collision capsule (the Menagerie UR5e capsules plus
    screwdriver, camera and screw) and its nearby person voxels, a velocity damper
    (Faverjon and Tournassoud 1987) keeps the distance above a safety distance
    d_s. It includes the voxel's own velocity;
  - **R2:** for each robot capsule and each sight line from the eyes camera centre
    to a person voxel, the same kind of damper keeps the capsule-to-line distance
    above a margin r_vis;
  - static obstacles (bench, frame, jig, feeder) and self-collision.
- d_s and r_vis cover delay × person speed, voxel size, depth noise and robot
  tracking error. Their formula, and every number in it, goes into notes.md
  with a source or "scene choice".
- If the QP is infeasible: solve for the rules alone (move away). If that also
  fails, brake to a stop within the acceleration limits. Count every fallback.
- **Visual servoing:** image-based control on the hole centre (Chaumette and
  Hutchinson 2006) for the two sideways directions; bit axis held from the box
  pose; the turn around the bit left to the look-around layer. Descend along the
  bit axis only when aligned. If the hole is lost: stop descending, hold, trigger
  look-around.
- **Screw drive, simulated:** when aligned within tolerance (0.5 mm and 2°, scene
  choice), the screw moves down its length over the drive time and becomes part of
  the box. Threads and torque are not simulated; notes.md says so.

### 8.6 The monitor

Separate code from the planner, same eyes camera. notes.md says a real cell
would give it its own safety-rated sensor.

- Every 2 ms it computes the protective separation distance in the ISO/TS 15066
  form as reproduced by Marvel and Norcross 2017, Eq. 1: the person's travel
  during the robot's reaction and stopping time (S_H), the robot's travel during
  its reaction time (S_R), the robot's travel while stopping (S_S), the intrusion
  distance C, and the position uncertainties of the person (Z_S) and the robot
  (Z_R). It uses the robot's stopping behaviour measured in this simulation, not
  UR's data.
- If any robot capsule is closer to a person voxel than that distance: protective
  stop. If the planner works it should almost never fire. Every firing is logged.
- **Person speed v_H, two settings of the same planner, run separately:**
  - (a) constant 2000 mm/s, the ISO 13855 worst case that Marvel and Norcross
    suggest using (Sec. 3, pp. 146 and 148; ISO 13855 allows 1600 mm/s beyond
    500 mm);
  - (b) measured person speed plus a bounded acceleration term. The ISO/TS 15066
    form allows v_H to be measured directly (Marvel and Norcross, Sec. 3). The
    acceleration bound must come from published data on human arm movement and
    is never tuned to our worker script.
  - In each run the planner's d_s and r_vis use the same setting as the monitor.
- **Intrusion distance C:** Marvel and Norcross (Sec. 7, p. 152) note that ISO
  13855's C was written for stationary machinery and may not fit robots. Here C is
  replaced by a stated term for the camera's detection limits (voxel size and
  depth noise), and the report says so.

### 8.7 Real time

- Here "real time" means the planner uses only what it has at that moment, and
  each cycle's compute fits its period on this machine. Report the 99th percentile
  and the worst compute time against 10 ms (motion) and 40 ms (perception).
- The simulation itself may run slower than wall clock. Any speed-up or slow-down
  in the video is labelled.
- The judge's renders are not counted in the planner's time.

## 9. What gets measured

- **Rules:** R1 contacts (count, who moved in, person speed, force from the
  judge), minimum robot to person distance; R2 blocked frames and pixels (count,
  who moved in).
- **Work:** boxes done, cycle time per box, idle share, time per mode, protective
  stops, QP fallbacks.
- **Look around:** blocking events and what resolved each (turn, posture, other
  screw, wait, park), time lost per event.
- **Rule cost:** share of motion cycles where an R2 constraint was active. If R2 is
  never active it is decorative, and the camera placement has to change.
- **Accuracy:** final alignment error per screw against the true hole (mm,
  degrees), missed screws.
- **Perception:** person point recall and precision against ground truth, box pose
  error, hole detection error, delay.
- **Compute:** per-cycle compute times.
- **Worker:** peak hand speed and acceleration per episode.

## 10. The judge: CobotSafe

- Ground truth for R1 and R2 comes from a separate evaluation package that the
  planner cannot import.
- A contact is measured the CobotSafe way: the latest state with no robot-person
  overlap anywhere, reruns with the person held still (free root and welded root)
  until every contact ends, and the plausibility check against what the arm can
  hold.
- CobotSafe's reruns replay its waypoint program, so this needs a small adapter
  that replays the logged joint commands instead. Install CobotSafe into the same
  environment and import it. **Do not edit CobotSafe without asking.**
- Force limits come from CobotSafe's data files with their sources, as in v0.1.
  They are for information only: a contact breaks R1 at any force.

## 11. Experiments

Every planner variant runs on the same worker episodes, so comparisons are paired.

| variant | what it is | why |
|---|---|---|
| B0 | visual servoing, no avoidance | shows the problem: contacts and blocked frames |
| B1 | monitor only, no replanning | what many cells do today: stop and wait |
| B2 | R1 constraints only | what R2 costs and what it prevents |
| B3 | R1 + R2, no looking around | pauses when blocked |
| B4 | R1 + R2 + looking around | the proposal |

- B1 to B4 run under both v_H settings.
- Seeds: tuning uses development seeds only. The final tables use held-out
  evaluation seeds that nobody tunes on.
- Start with 20 episodes per variant and setting. Size it in gate 3 from the
  measured episode time so the matrix runs overnight with at most 3 workers.
- Stress scripts run on B3 and B4.

**Headline acceptance for B4 on the evaluation seeds:**
- R1: no contact where the robot moved in. Contacts where the person moved into a
  stopped robot are reported with count, force and situation, and are reported.
- R2: no blocked frame where the robot moved in. Frames where the person moved
  behind the robot are reported with how long the robot took to clear the view.
- Every screw driven within tolerance, or each miss explained.
- Cycle time reported against B1 and B3, whatever it is.
- If any of these fail, report the numbers as they are and decide then.
  No retuning on evaluation seeds.

## 12. Gates

Each gate ends with renders or numbers and waits for review. Before
building a gate, state what result would kill it.

1. **Station.** Build the scene. Show six stills: hero, side, top, eyes camera view
   with the worker at the bench, wrist camera view of a hole, close-up of the parts
   and holes. Numbers: humanoid elbow height and the bench height set from it; eyes
   camera coverage of the worker over a set of work poses; reachability for each
   hole and the feeder (share of turn angles around the bit with a valid IK
   solution, per branch); a park pose that blocks no view of the worker in any
   work pose.
2. **Worker.** The scripted assembly with the robot parked. Show one full cycle
   from the hero and eyes cameras. Numbers: hand and part penetration, peak hand
   speed and acceleration, foot sliding.
3. **Baseline B0.** Plain visual servoing while the worker works. Show a clip and
   the judge's numbers, plus wall time per episode. **Kill test:** if B0 shows no
   contacts and no blocked frames on the development seeds, there is nothing to
   avoid; rethink the task before going on.
4. **Perception.** Overlays of person voxels, unseen space and sight lines on the
   eyes view, and hole detection on the wrist view. Numbers: recall, precision,
   delay, box pose error, hole error, blocking prediction accuracy.
5. **Planner R1, then R2** (B2, B3) on development seeds, with clips of close
   passes.
6. **Look around** (B4). Blocking events and how each was resolved; time lost, B3
   against B4. **Kill test:** if looking around does not cut the time lost against
   B3, the feature is dead and we say so.
7. **Evaluation** on held-out seeds: every variant, both v_H settings, stress
   scripts. results.md and a README draft.
8. **Video.**

## 13. The video (after gate 7)

- About 2 minutes, captions on screen, rendered offscreen (MUJOCO_GL=egl, ffmpeg
  through imageio).
- Story for a normal person: the station and the job; the robot's eyes view with
  the worker highlighted; both working at once; a close pass where the robot moves
  around a hand; a hand blocks the wrist camera and the robot turns around the bit
  to see the hole; the sight lines the robot keeps clear; the numbers from gate 7
  (did the rules hold, cycle time against stop-and-wait); what this is not.
- Every clip is a real closed-loop run from the evaluation set, with its seed on
  screen. Speed changes are labelled. A clip picked because it looks good says
  "selected", and the numbers card covers the whole set. If failures exist, the
  video does not claim zero.
- The author's rule: **the video may be produced well but never fabricated.**

## 14. Repo and environment

- A ROS 2 workspace of its own, separate from CobotSafe: the judge should not live
  inside what it judges.
- Python: the packages in `requirements.txt` (MuJoCo 3.13, dm_control 1.0.46, numpy,
  pyyaml, imageio with ffmpeg), in the interpreter that runs the ROS 2 nodes. No
  OpenCV, no scipy, no QP library: the QP is 200 lines of numpy and easier to check.
- The UR5e model comes from a checkout of google-deepmind/mujoco_menagerie; its path
  goes in `MUJOCO_MENAGERIE` (default `~/mujoco_menagerie`).
- Rendering is offscreen through EGL (`MUJOCO_GL=egl`). Renders and tests run one at
  a time on the 7 GB laptop this was built on.
- House rules for the documents: every number taken from a standard or a paper
  carries its source, from a document actually opened; everything else is marked as
  a scene choice; nothing claims to certify a cell or to replace a physical
  measurement; old numbers are kept and marked superseded, never deleted; every bug
  gets the test that catches it (CHANGELOG.md).

## 15. Known risks

- Reactive control gets stuck in local minima. The look-around layer must notice
  no progress.
- An overhead camera may leave little room when hands are at the jig. Camera
  placement in gate 1 decides much of this, so it is chosen from numbers.
- Space the robot hides far from the person is not covered by R2. A hand can only
  get there through visible space, and the margin is what makes that safe. This is
  measured, not assumed.
- The kinematic worker cannot be pushed. Contacts are detected, and their force
  comes only from the judge's reruns.
- Depth alone cannot tell a person from a loose part, so loose parts count as
  person.
- Planner and monitor share one camera that is not safety-rated.
- Rendering cost: every eyes frame needs a depth render and a self-filter render,
  and the judge adds two segmentation renders.
- The screw drive is not physical.

## 16. Changed from what was said earlier in the chat

- Blind Spot's FeatureGuard is out of scope for now. The final alignment uses one
  hole (two image features), where its degeneracy test does not apply. Blind Spot
  also measured its area signal firing on a healthy oblique view (70°), and an
  off-axis wrist camera always sees the hole at an angle, so it would need
  re-measuring first. Revisit only if several holes are used as features at once.
- The work lives in its own repo, not inside CobotSafe (reason in section 14).

## 17. Decisions (open ones do not block gate 1)

1. Name and location: SightLine, `~/sightline`.
2. Which v_H setting leads the headline. **Decided 2026-09-17: measured speed (b)
   leads**, shown next to the constant 2000 mm/s (a). Before it can lead, gate 4
   must measure the error of the measured speed against ground truth: 2 cm voxels
   at 30 Hz quantise speed in 600 mm/s steps, and Marvel and Norcross (p. 148)
   warn that a few mm of position noise adds hundreds of mm/s. Setting (b) also
   still needs a published source for its acceleration bound.
3. Videos in git or as release files, for both projects. I recommend release
   files.
4. CobotSafe v0.1 is still unpushed.
5. The CobotSafe v0.1 video's search-grid curve still uses pre-fix search 5 scores
   (its 2.85 overstates). Fix that before the video is published.

## 19. Changes agreed after the scope was written

**2026-09-17, decisions**
- In the worst case the robot holds. Holding is the fallback, never a reason to
  stop the project; how often it happens is reported.
- Watch the person's measured speed rather than assume one top speed for everyone
  (section 17.2).
- Everything practical: every piece of equipment is a real product with numbers
  from its datasheet, and the planner only gets what a real camera and a real UR5e
  controller would give it. A true digital twin mirrors a cell that exists, so the
  README calls this a simulated cell built from real parts.
- The project is pitched to manufacturers: succeed in simulation, then raise funds
  for a hardware test.

**2026-09-18, built and decided during gate 1**
- Worker scaled to 1.75 m, mass kept at 70 kg (section 6).
- Bench height fixed at 1.118 m from the measured elbow height (section 6).
- Real products modelled: OnRobot Screwdriver shape, RealSense D455 and D405
  (sections 6 and 8.2). The enclosure, rail, terminal blocks and feeder are still
  scene choices and need matching to products.
- The screwdriver mounts from the side, so turning around the screw moves the whole
  arm and reach depends on the turn angle (sections 6 and 8.4).
- Eyes camera moved to a pole beside the bench, 3.25 m up, 56° down (section 6).
- The station screen and teach pendant moved to the left upright: they sat in the
  camera's line of sight.
- Design-time visibility is measured by ray casting to the worker's surface, which
  names the blocker. The R2 judge still uses rendered segmentation (section 4).

**2026-09-18, built during gates 2 and 3**
- The worker takes two side steps to the parts rack during his cycle: the totes are
  0.82 to 0.96 m from his shoulder and his arm reaches 0.77 m, so he cannot work the
  rack from the bench (section 7).
- R1's collision model now covers the screwdriver nose, the bit, the screw and the
  wrist camera. Before that the bit tip sat 74 mm outside every collision shape and
  a poke with the screw would have scored as clear (section 4).
- The enclosure's collision shape is a hollow box, not a solid one, because the rail
  screws are driven inside it. With the tool modelled, reaching a rail screw is clear
  of everything at 17 of 24 turn angles, not 23 (section 5).
- B0 drives from taught positions. Its wrist detector finds a hole 7 to 16 mm off,
  which is worse than the taught position, so corrections over 4 mm are refused.
  Gate 4 builds the detector properly and B0 can be re-run then (sections 8.3, 12).
- A gate 3 episode is one worker cycle. Running on with the worker frozen would count
  blocked frames against a person who is not working any more.
- **Open:** section 4 labels a violation by putting the robot back one
  motion cycle, 10 ms. For contacts that works. For a blocked camera frame it does
  not, because 10 ms is 2.5 mm of arm travel and a 30 Hz camera only catches an
  overlap once it is wider than that, so every event reads "the person moved in".
  Asking the same question one camera frame back, 33 ms, matches what the video
  shows. Gate 3 reports both. One line of section 4 needs to change.
- **Still open from gate 2:** the parts rack is a blind spot. While the worker carries
  a part back from it the eyes camera sees 0 % of his hands, most of it hidden by the
  rack's own shelves. Move the rack, lower it, add a second camera, or let the robot
  hold while he is in there. Holding is the default under our own rule (section 8.3).

**2026-09-18, built during gate 4**
- The depth noise model is held flat past the 2.75 m Nguyen et al. fitted it over,
  and the depth image is cut off at 4 m (section 8.2). Extrapolating it made the far
  floor noisier than the background test could tolerate.
- The box pose fit reports a centre and a height only. Yaw is not observable from the
  eyes camera at 3.25 m, so the jig sets it (section 8.3, step 8).
- Unseen space is used for R1 but not for deciding whether the wrist camera can see a
  hole: on the sight line it raises a false alarm on 43 % of frames. Two uses, two
  rules (sections 8.3 step 7 and 8.5).
- The cover no longer carries two pre-driven screws. That was a posing choice for the
  gate 1 stills and it meant the robot drove screws into filled holes.

**2026-09-18, decided during gate 5**
- The eyes camera moves to (1.70, -1.50, 3.60) m, 47.6 degrees down, picked by
  `sightline_sim/gates/eyes_optimise.py` from 76 pole positions for two things at once: it sees
  86.6 % of the space the arm and a person can share above the worktop (the old pole
  84.5 %) and 82.0 % of the space behind the robot where a second person could walk
  up (was 79.4 %). Section 6.
- What that camera still cannot see, 13.4 % of the shared space and almost all of it
  the bench top's shadow, is handled by the robot and not by more hardware: the arm
  stays above the worktop; while a person is in the cell the hidden cells count as
  occupied for R1; the wrist camera can look into them (gate 6); and if the worker's
  hand keeps the robot waiting for more than 2 s the cell asks him to move it and
  the wait is counted. The estimate is that this bites about a tenth of the
  time. Gate 5 measures it. Sections 8.3 and 8.5.
- R2 violations are labelled one camera frame back (section 4).
- The safety machinery only binds when the arm can reach the person in time. The
  full reach envelope is 1324 mm and covers the whole cell, including the rack, so
  "out of reach" never happens here; the useful form is the damper itself, which does
  nothing until a pair is within 35 cm.
- **The robot plans against the camera's own grid (the grid algorithm).** Picture
  the camera's view as a grid of directions. Every direction in which it sees the
  person is off limits from the camera out to just behind him: in front of him the
  arm would block the view (R2), at him it would touch him (R1). The margin is the
  safety distance, the size of the part of the arm being checked, and how far he
  could have moved since the picture at his measured speed. The grid is rebuilt with
  every picture and the arm checks every joint, the tool tip and points along every
  link against it every 10 ms. B4 is built on it (sections 8.4, 8.5).
- **Both cameras, the grid and the judge's R2 frames run at 25 Hz, 40 ms** (the
  user: "make it 40 ms, it's okay"). The motion cycle stays 10 ms (sections 4, 8.2,
  8.7).
- Choices stated for the grid and not objected to: 15 cm behind a seen
  surface is off limits too (a limb's thickness); points along every link are
  checked, not only the joints and the tool tip; the fixed blind area counts as
  occupied when a seen part of him is within 35 cm of it.

**2026-09-18, decided after the bound for any planner**
- With his hand holding the cover and then resting beside the next hole, no planner
  could drive a cover screw 5 cm clear of him and off the camera's view of him (0 % of
  that window, perfect knowledge of the worker). The choice was: **the jig's toggle
  clamps hold the cover**, and he works elsewhere while the robot screws (section 7).

**2026-09-18, decided after the first clamp-cycle results**
- The pitch needs the robot and the worker visibly working at the same time: **at
  least two screws must go in while he is working beside the robot**, really driven
  under the rules; the video may show it well but never fake it. So his tasks are
  re-ordered the way a line is balanced (section 7): the rail screws while he clips
  blocks on the next rail 40 cm from the jig and then fetches the cover; the cover
  screws while he finishes that rail. His order is a scene choice, and results.md
  says it was chosen for this.
- The jig gives four signals: rail pressed in, cover placed (which ends the rail
  screws, they are under it), clamps closed, clamps opened (which ends the cover
  screws). Without the second the robot would have driven a rail screw through a
  cover it had not been told about; without the fourth it would have screwed an
  empty jig.
- The prep area moves 15 cm further from the jig, to x = -0.55 m: at -0.40 m his
  upper arm was 17 to 25 cm from the robot's wrist at a rail screw, inside the
  planner's margin (section 6, notes.md).
- The parts rack moves 30 cm toward the worker's side (section 6): against the
  bench's left end its far totes were reachable only from inside the bench, where the
  script had him standing since gate 2. He waits with his hands at his sides, not on
  the bench edge, 26 cm from the front cover holes.
- Waiting for the cover screws he steps back 30 cm from the bench with his hands at
  his sides. Against the bench his head was 23 cm from the front cover holes, and the
  15 cm hidden zone behind anything seen plus the margin keeps the arm about 33 cm
  along the camera's line of sight behind his head, which covered the two left holes
  for the whole wait.
- Coming back from the rack he turns in place first, then walks (section 7): turning
  and walking as one blended move swung his free hand 6 mm from the robot's shoulder.
- The overhead camera's mount will be searched again once the cycle is settled, the
  cell frame included, with the cost that matters now added: how often the robot's
  work sits on the camera's lines of sight to him. Gate 1's search counted only how
  much of him each mount sees.
- **Hardware path (2026-09-18):** the planner takes
  a `SensorFrame` in and a joint velocity out every 10 ms, and the grid takes a depth
  image in every 40 ms. On hardware each becomes a ROS 2 node: the depth image from
  the RealSense driver, the joint state and velocity command through the UR ROS 2
  driver, and rviz for the live 3D view that would have shown the worker-script
  faults of gate 5 at a glance. Nothing in the planner depends on the simulator, so
  the wrap is small; it is not done until the simulation gates are.

- **ROS 2 and Gazebo, on the instruction (2026-09-18):** "run all of this on
  ROS 2, Gazebo, rviz and record from there." Done first as a replay: the run the
  judge scores in MuJoCo is recorded (every body pose at 25 Hz) and played into
  Gazebo Sim and ROS 2 frame for frame, for the Gazebo-rendered videos and the desktop
  recording of Gazebo's GUI and rviz2 (README "Gazebo and ROS 2"). MuJoCo stays the
  simulator and the judge; Gazebo renders. The live version, the planner and the
  simulator as ROS 2 nodes talking over topics, is what the hardware path above
  needs anyway. Tools added for it, the request: ROS 2 Lyrical, Gazebo 10,
  rviz2, `sdformat-mjcf`, `ros_gz_bridge`, `ur_description`.
- **Stage 1, done 2026-09-19:** the cell (MuJoCo, the worker, the eyes camera, the jig,
  the judge) and the planner (B4) are two ROS 2 nodes in lockstep on sim time
  (notes.md "The planner live on ROS 2"). The lockstep is a decision: the run
  must not depend on the machine's speed, so the cell waits for each command and
  the planner for each picture. Parity with the in-process runner is exact. The
  judge stays with the truth, in the cell.

**2026-09-18, the worker's script corrected with the clamp change. Not agreed yet**
- A hand with nothing to reach now hangs whole; before, only the shoulder was reset,
  and at the rack his right hand stood 1.42 m up behind him in the robot's side.
- He steps over to the prep area to work there, instead of reaching 42 cm to his
  left from the jig with his head over it.
- Waiting, he stands upright at the bench with his hands on its edge, watching the
  robot, instead of leaning over the jig.
- Moves are long enough that his hands stay at gate 2's pace (1.3 m/s at most); the
  first version of the new moves reached 8.9 m/s.

**2026-09-18, built during gate 5**
- Each part of him grows by its own measured speed, not by his fastest part's. With
  one number for all of him, parts coming into view at the rack read as 2.4 to 2.7
  m/s while his hands moved at 0.03 to 0.7 m/s, and the margin round every part of
  him grew with it (section 8.6 (b)).
- Where the arm itself hides part of the view, the grid keeps what was last seen
  there, with its age, until the camera sees into it again (section 8.3).
- Points of the arm that no joint can move, the shoulder on its own axis, are left
  out of the guard's decisions: nothing the arm does changes them, and he works 18 cm
  from it whenever he puts a part in the jig. A contact there is labelled by the
  judge like any other. Points that stay within 0.2 m of the base's axis, the root of
  the upper arm, must never move closer to him but do not send the arm fleeing.
- The guard's hard rule is "no point of the arm deeper inside the off-limits space
  than holding still would leave it", the same as the motion layer's "do not close".
- The fixed blind area leaves out the jig and the box (fingers there hang under a
  wrist the grid sees), and a blind cell is live when a cell of him is within 20 cm,
  a hand's length, instead of any point of him within 35 cm (section 8.3).
- B4 fetches the next screw while he works and waits at the feeder with it, at a
  feeder pose that keeps the arm behind the jig's back edge (section 8.4).
- B4 plans against the grid as it will be 0.15 s ahead, the guard's own stopping
  horizon, gives up a hole only when its way stays closed for two pictures, and does
  not choose a hole again for 1 s after giving it up (section 8.4).
- The measured person speed is capped at the ISO 2000 mm/s: a faster reading is
  noise, and the constant setting uses that value anyway (section 8.6 (b)).

## 18. Sources opened for this file (2026-09-17)

- J. A. Marvel and R. Norcross, "Implementing speed and separation monitoring in
  collaborative robot workcells", Robotics and Computer-Integrated Manufacturing
  44 (2017) 144 to 155. Eq. 1 (p. 144): the ISO/TS 15066 protective separation
  distance. Sec. 3 (pp. 146, 148): v_H, the ISO 13855 values 2000 mm/s and
  1600 mm/s beyond 500 mm, and v_H may be measured. Sec. 7 (p. 152): C. Open copy:
  https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=914783
- C. V. Nguyen, S. Izadi and D. Lovell, "Modeling Kinect Sensor Noise for Improved
  3D Reconstruction and Tracking", 3DIMPVT 2012, Eqs. 1 to 4 (pp. 526 to 527).
  Open copy:
  https://users.cecs.anu.edu.au/~nguyen/papers/conferences/Nguyen2012-ModelingKinectSensorNoise.pdf
- Universal Robots, "UR5e Technical specification", updated May 2025: reach
  850 mm, maximum TCP speed 4 m/s, every joint ±360° and ±180°/s.
  https://www.universal-robots.com/manuals/EN/TechSheets/UR5e_techsheet_pdf_online/UR5e_techsheet_en.pdf
- Universal Robots, RTDE guide: real-time control loop at 500 Hz on e-Series.
  https://docs.universal-robots.com/tutorials/communication-protocol-tutorials/rtde-guide.html

Opened later, during gate 1 (2026-09-18):
- OnRobot, "Screwdriver datasheet v1.7": specification table (p. 2) and mechanical
  drawing (p. 34).
  https://onrobot.com/storage/datasheets/screwdriver/datasheet_screwdriver_v1.7_en.pdf
- Intel, RealSense D455 and D405 product specification pages: depth fields of view,
  housing sizes and ranges.
- CCOHS, "Working in a Standing Position - Basic Information" (updated 2022-11-30):
  work surface 5 to 10 cm below elbow height for light work.
  https://www.ccohs.ca/oshanswers/ergonomics/standing/standing_basic.html
- NCD Risk Factor Collaboration, "A century of trends in adult human height",
  eLife 2016;5:e13410: the range of mean adult male height.

Method references (no numbers taken from them yet; check the full citation before
it goes into a paper):
- F. Chaumette and S. Hutchinson, "Visual servo control, Part I: Basic
  approaches", IEEE Robotics and Automation Magazine, 2006.
- B. Faverjon and P. Tournassoud, "A local based approach for path planning of
  manipulators with a high number of degrees of freedom", ICRA 1987.
- F. Flacco, T. Kröger, A. De Luca and O. Khatib, "A depth space approach to
  human-robot collision avoidance", ICRA 2012.
- T. Flash and N. Hogan, "The coordination of arm movements: an experimentally
  confirmed mathematical model", Journal of Neuroscience, 1985.
- D. Goldfarb and A. Idnani, "A numerically stable dual method for solving
  strictly convex quadratic programs", Mathematical Programming, 1983.
- Kirschner et al., arXiv:2203.02706, as used by CobotSafe.
