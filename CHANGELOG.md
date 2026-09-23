# Changelog

Every bug found, and the test that now catches it. Newest first.

## 2026-09-20

### The planner ported to C++

- **`sightline_planner_cpp`**: the whole planner in C++17 on Eigen and the MuJoCo C
  API, with no ROS and no simulator in it. QP solver, kinematics, view grid, motion
  layer, guard, perception and the B0/B4 task layers, the same constants and the same
  order of operations as the Python. `sightline_ros_cpp` is the planner node and the
  topic contract; `stage1.launch.py planner:=cpp` runs it in place of the Python one.
- **The known world came back flat at the near plane, every pixel, every frame.**
  MuJoCo's own renderer sets `mjrContext.readDepthMap = mjDEPTH_ZEROFAR` right after
  `mjr_setBuffer`, and the depth it hands back is undone with the reversed
  coefficients. Left at the default `mjDEPTH_ZERONEAR` the buffer is a standard depth
  map, the reversed coefficients map all of it to `znear`, and nothing reads as
  further away than 12 mm. Checked against the Python renderer on the same model and
  camera: 2.3240 m nearest, the same 303066 pixels past the far plane.
- **The same render saw nothing on `EGL_DEFAULT_DISPLAY`.** On a machine with two
  graphics devices the default display initialises and renders an empty scene.
  MuJoCo's headless renderer goes through `eglQueryDevicesEXT` and
  `eglGetPlatformDisplayEXT` and leaves the context surfaceless, and
  `MUJOCO_EGL_DEVICE_ID` picks the device. The C++ `KnownWorld` now does the same, and
  refuses to start if `mjr_setBuffer` did not give it an offscreen framebuffer.
- **`spec.to_xml()` refused the robot-only model: "no support for buffer textures".**
  `station.export_robot_mjcf`, which writes the model the C++ planner loads, builds it
  with an assets directory so every texture goes to a PNG file, the way `export_mjcf`
  already did for the whole cell. Unlike `export_mjcf` it leaves the slashes in names
  alone: the planner looks its bodies up as `ur5e/shoulder_link`.
- **Checked against the Python on the way:** the C++ guard puts the same 63 points on
  the arm, 58 of them moving and 8 rooted on the base's axis, with the same 4.009591 m
  of radii; forward kinematics, the wrist camera pose and the Jacobian norm agree to
  four decimals; and the exported model has the same bodies, sites, cameras, mocap
  parts, collision shapes, joint limits and field of view as `robot_only_model`. The
  solver and the geometry helpers have gtest cases in `sightline_planner_cpp/test`.

## 2026-09-19

### Stage 1, the planner live on ROS 2

- **The first two ticks timed out.** The planner builds B4 (75 s of IK) only once the
  cell's knowledge arrives, and the cell started ticking before that. The planner
  now says `/sightline/planner/ready` and the cell's first tick waits for it.
- **A picture cost the planner 164 ms, the perception 95 of them.** The rest was the
  rviz grid built as a line list point by point in Python. The grid is three point
  clouds straight from numpy (`lockstep.cloud_msg`, 9 ms for all rviz output).
- **Parity check added:** `results/ros2/stage1/parity` holds the runner's poses over
  the same 6 s; the ROS 2 run matched them to 0.0000 mm, with the same states and
  verdicts in every frame. `sightline_ros/test/test_ros2.py` covers the message helpers.
- **rviz opened empty on the stage 1 layout.** A display name with a colon in it
  ("Grid: where the camera sees him") is not valid YAML unquoted, and rviz falls
  back to its default layout without a word. Names quoted; both layouts are checked
  with a YAML parser before use.
- **`ros2 bag record` took no topics.** This release wants `--topics`; the
  positional list was refused and the first full run had no bag.

### The desktop recording (the Gazebo half lagged)

- **The Gazebo window repainted at 7.6 fps in the desktop recording.** Measured from
  the video (70 % of frames unchanged); rviz beside it did 24.8. The GUI rendered on
  the Intel GPU that drives the display; the NVIDIA GPU only had the headless server.
  With PRIME offload the GUI renders fast but its window reads back through X at 1 to
  13 fps, with stale frames; the VideoRecorder plugin has no remote trigger here and
  GNOME's screencast refuses scripts. The Gazebo half is now the server's front camera
  encoded live on the GPU (`frame_grabber.py --video`), 25 fps with no repeats, and the
  page and results log say so.
- **The rviz half ran 1.5 s behind the Gazebo half.** The grab started 0.5 s after the
  driver process, which spends 2 s loading before its first frame. The driver logs the
  wall time of its first frame, the grab starts first, and the difference is trimmed.
- **A 320 KB caption picture every frame cost the driver 10 ms.** Sent only when its
  words change.

### The Gazebo replay

Found while taking the judged run into Gazebo Sim (ros2/). In the order found.

- **The worker and the robot were invisible in Gazebo.** The converter takes geom
  group 0 as visual and 3 as collision and drops the rest; the worker's skin was
  group 1 and the robot's group 2. `export_mjcf` folds both into group 0.
  `sightline_ros/test/test_ros2.py` checks the export holds only groups 0 and 3.
- **The converter crashed on the worker's clothing.** Its 39 meshes were vertex lists
  inside the model; the converter wants a file per mesh. The export writes them as
  OBJ files and clears the inline data (MuJoCo refuses a mesh with both). The test
  checks no ` vertex=` attribute remains and every mesh keeps its vertex count.
- **The replay world would not load: "Non-unique name[hall]".** A light and a model
  shared the hall's name. Lights are prefixed `light_`.
- **The replay posed only 10 of 40 bodies, twice for different reasons.** First the
  poses named bodies that have no model (the worker's joint frames), and one unknown
  name fails the whole request; now only names the world lists are sent. Then the
  recording's names carry MuJoCo's slash (`worker/head`) while the world's carry the
  export's `__`; the test file had been built from the export, so it never showed.
  Mapped at the request.
- **Every service request took 2 s and most timed out.** A Python callback receiving
  images and a blocking request in the same process wait on each other for the
  interpreter lock. The grabber is a process of its own (`sightline_sim/gazebo/frame_grabber.py`).
- **The replay deleted its own frames.** Frames saved by stamp and frames renamed by
  index matched the same pattern, so each step removed the previous step's result.
  Stamped names carry a `t`; `sightline_ros/test/test_ros2.py` covers taking a frame by stamp.
- **One extra picture appeared after the last frame.** Stepping by a count of
  iterations accumulated; the world is now run to an absolute sim time per frame,
  and a picture a whole period past the wanted one is reported, never used.
- **A stale server answered the new one's requests.** `kill` on the `gz` wrapper
  leaves `gz-sim-main` running; both are stopped by PID, and every replay runs on its
  own `GZ_PARTITION`.
- **`set -u` killed the recorder inside ROS's setup script.** It is sourced before
  the strict mode.
- **The live driver fell 14 % behind real time.** 17 of its 24 ms a frame were
  `np.load`'s lazy file: every `rec["xpos"][i]` decompressed the whole array again.
  The arrays are read into memory once (3 ms a frame now, `assemble_video.py` too).
- **rviz showed no station shapes.** `geom_type[g] in (BOX, CYLINDER)` is the gate 5
  capsule bug again: an enum in a tuple never matches a numpy int in these bindings.
  Compared as plain ints; 351 shapes now.

## 2026-09-18

### Gate 5

Found while running the first B2 and B3 episodes. In the order they were found; each
one hid the next.

- **The planner was blind exactly where the arm meets a hand.** The self filter
  deleted every point within 5 cm of the arm, and the workpiece filter deleted
  everything inside the jig volume, where his hands are while the robot works. With
  the tool beside his hand the truth had 36 to 55 person voxels within 10 cm of the
  arm and the planner had none; its closest was 137 mm away while his hand was 7 mm
  away. Replaced by design.md 8.3's method: render what the planner knows is there (its
  own arm, the parts at their fitted pose) as a depth image from the calibrated
  camera, and call only what is clearly in front of that the person. The render
  matches the simulator's camera to 0.00 mm. Now 22 to 24 voxels within 10 cm are
  seen, closest 20 mm, and recall rose from 89 to 93 %.
- **The robot froze on its own ghost.** The camera is a frame late, so the picture
  shows the arm where it was 33 ms ago; subtracting it where it is now left a band
  2 to 3 cm wide along every moving link, 34 to 52 voxels a frame, that read as a
  person hugging the arm. B2 sat still for 18 s with 548 fallbacks. Fixed by
  subtracting the arm at the joint angles from when the picture was taken, plus a
  2 pixel margin for the camera's sideways noise: 0 ghosts in six trials.
- **The joints did not move together.** The acceleration limit was applied per
  joint, so the small wrist joints finished while the shoulder was still starting its
  swing, and the tool reached out and down 40 cm under its path before it turned.
  B0 now moves every joint on one synchronised profile: ramp, cruise, and a stop
  exactly at the goal. As a side effect the cycle got much faster: six screws in
  33.7 s where the old exponential approach took about 70 s.
- **The damper could demand a stop the arm could not make.** A link entering a 10 cm
  zone at 0.6 m/s needs 9 m/s^2 to stop; it could not, and went 105 mm under the
  worktop. The bound is now "could still stop in time", approach speed at most
  sqrt(2 a (d - d_s)) with a = 3 m/s^2, and each zone is sized so the arm's top
  speed is allowed at its edge.
- **"Nearest pose" clamped a joint and changed the pose.** Shifting the taught feeder
  pose's shoulder by a whole turn put it past its limit, and clamping it produced a
  different pose 397 mm away, over the jig. `nearest()` now picks among whole-turn
  copies inside the limits and never clamps.
- **Solving each target afresh flipped the arm's configuration.** After backing away
  from the worker by 15 mm, "the solution nearest to here" was on another branch and
  the next move dived under the path. B0 now teaches every pose once, as a chain from
  home, like a programmer teaching a cell, and uses them exactly as taught. The chain
  also decides which way the shoulder swings: round the back, never over the worker.
- **The brake reversed the joints.** The last-resort fallback set the speed to minus
  the previous one, clipped, which throws each joint into reverse. It now slows each
  joint toward zero.
- **Waiting in place is the worst place to wait.** Held by the rules over the box, the
  arm was walked into 54 times and blocked the camera 44 % of the time. When held for
  0.3 s it now steps back to a stand-off for that screw, 25 cm back and 30 cm up,
  waits until the screwing pose is clear of what the camera sees of him by the safety
  distance plus 5 cm, asks him to move his hand after 2 s, and goes back in.
- **A missed screw stalled the rest of the cycle.** Once the cover went on, the rail
  screw under it could not be driven, but the arm kept trying it and never started the
  cover screws. A screw whose stage has passed is now recorded as missed, with the
  reason, and the next one starts.
- **The R2 constraint looped in Python over every voxel for every link every cycle.**
  Two seconds per 10 ms cycle; B3 ran for 86 minutes before I stopped it. Vectorised
  and cone-filtered: identical results on 1040 pairs, 417 times faster.

- **The damper made every cycle infeasible when a pair started inside the safety
  distance.** Faverjon and Tournassoud's bound goes positive there, ordering a
  retreat, and with several pairs the orders conflict, so the QP had no answer and
  the arm froze where it was. Fixed by capping the constraint at "do not get closer",
  which standing still always satisfies against a still person, and asking for the
  retreat in the objective instead. Tests: the still person, the empty cell, the
  walker, the plane (`test_motion.py`).
- **The person's velocity had the wrong sign in the damper.** d_dot is the person's
  speed along the normal minus the robot's; it was added instead of subtracted, so a
  hand coming in would have loosened the constraint. Caught while writing the walker
  test.
- **Comparing a numpy array against a Python enum cost 160 ms per image.**
  `objtype == mujoco.mjtObj.mjOBJ_GEOM` falls back to element by element Python.
  Against a plain int it is under 2 ms. This was most of why gate 3 and 4 episodes
  took ten minutes; per frame cost went from 269 ms to 78 ms with two more fixes:
  row-wise `np.unique` replaced by packed integer keys (24x), and the judge only
  unprojecting the person's pixels instead of all 300 000.
- **The gate 2 blind spot was blamed on the wrong thing.** I wrote that the rack's
  shelves hide the worker's hands while he carries a part back. Ray casting names
  the blocker: the bench top, 492 of 662 points; the rack hides none. Corrected in
  results.md. The fix that follows is different too: no rack change helps, the
  camera position does a little, and the rest is a planner rule.

Found while building B4 on the camera grid (the grid algorithm, 25 Hz):

- **Every link of the arm was a ball.** `geom_type[g] in (mjGEOM_CAPSULE,
  mjGEOM_CYLINDER)` is always False in these bindings (`==` works, `in` does not), so
  the motion layer, B0's clear check, B4's wrist view check and the first guard all
  modelled each link as a ball of its radius at its centre. The ends of the 400 mm
  upper arm and 380 mm forearm were not in the model. All four now use
  `kin.shape_line`. Every earlier gate 5 number is superseded, and the motion test's
  "person in the way" had to move: with full-length links the arm started 34 mm
  inside it.
- **The grid was grown once per picture, at the wrong depth and as a square.** Growing
  it by the margin at his nearest depth (his head), rounded up to 2 cm, with a square
  filter, turned a 12 cm margin into 19 to 26 cm. Replaced by the exact test per point
  and cell, measured at the point's own depth.
- **Depth noise made cells of him all over the room**, 18.9 a picture. A cell now
  needs a neighbour: 1.75 a picture, none of his lost.
- **One speed for all of him.** Parts coming into view read as jumps of 2.4 to 2.7
  m/s, and the margin round every part of him grew with it: the parked arm was off
  limits for seconds at a time. Each part now grows by its own speed.
- **The worktop check flagged the robot's own shoulder**, which is bolted at worktop
  height, and the points on the shoulder's axis, which no joint moves, froze the arm
  whenever he put a part in the jig 18 cm away. Both left out of the decisions.
- **Building the arm's 63 points in a Python loop** cost half of each check. Now a
  few numpy operations, identical to 2e-16, and a check costs 0.2 ms.
- **"Nothing but a way out while inside the margin" froze the arm.** He works 20 cm
  from the root of the upper arm, which no way out moves far. The hard rule is now
  the motion layer's own: no point of the arm deeper inside than holding still would
  leave it. A first version allowed 2 mm of slack, and a slow move crept down onto
  his hand 2 mm a cycle; the slack is zero.
- **The memory of what the arm hides grew without bound**, first in time (53 m by
  the end of the cycle) and then in space, spreading from his hands at the jig
  through the parked arm's whole shadow. It now keeps only cells of him the arm
  covers in the next picture, at their own depth and speed, for 0.5 s.
- **The speed estimate read still hands at 0.92 m/s.** A cell straddling two
  surfaces of him flipped its nearest pixel between them. Each cell now also keeps
  its farthest pixel, a reading needs a neighbour that agrees, and the cap is the
  ISO 2000 mm/s: still hands read 0.26 m/s.
- **The blind area's cells inside the jig, and stray noise near the feeder, held the
  arm off its own feeder.** Cells inside the jig are left out (a hand there is under
  a wrist in plain view, which the grid covers); liveness is counted from the grid's
  filtered cells within a hand's length, 20 cm, not from raw points within 35 cm:
  the cells beside the feeder went from live in every picture to 15 %.
- **The arm waited at park and swung 180 degrees round the base once each part was
  in**, while he was still placing it. B4 now fetches the next screw while he works
  and waits at the feeder with it.
- **B0's feeder pose hung the elbow 11 cm over the jig.** B4 teaches its own, of all
  turn angles and arm configurations the one furthest behind the jig's back edge:
  6 cm behind it, and the way there from park stays behind it too.
- **Only 3 of the 23 turn angles that reach the first rail screw were kept**, all with
  the wrist toward him, because the options were walked round from one starting pose.
  Each angle is now solved from several.
- **B4 chose the same hole and dropped it twelve times in 12 s.** Planning asked the
  grid as it was; the guard asked it grown by his motion over the arm's stopping
  time, so a hole passed the one and failed the other. Planning now asks the grid
  0.15 s ahead, a hole is dropped only if its way stays closed for two pictures, and
  a dropped hole waits 1 s before it can be chosen again.
- **The worker's new cover stage moved his hands at up to 8.9 m/s.** Short moves
  between very different arm poses, and lift-offs taking half of them. Each move is
  now long enough for gate 2's pace, 1.3 m/s at most; the key-pose blend is unchanged.
- **Hanging his arms at the bench blended them through wide arcs**, one over the jig
  into the robot driving a rail screw, one through the bench edge. At the bench his
  hands now rest on its front edge while he waits or steps along it; they hang only at
  the rack, as in gate 2. A general fix, adding keys wherever a hand strayed from a
  straight line, was tried and dropped: straight lines cut through the bench edge
  where the arcs went over it, and the added keys doubled the rack trips' hand speed.
- **The bound for any planner under-reported the clearance.** Its distance started
  from the bounding-sphere estimate and refined only pairs under 15 cm, so a true
  25 cm read as 5. Every pair under 30 cm is measured exactly now, the rest is bounded
  by the spheres. The earlier bound tables were conservative, never optimistic.
- **The robot could have driven a rail screw through the cover.** The jig signalled
  "cover ready" only when the clamps closed, 2 s after the cover was placed, and
  nothing told the planner the rail screws were under it. The jig now signals the
  cover placed and the clamps opened as well (design.md section 19).
- **Dropping the hands to his sides by "hang" swung the right hand over the jig** at
  1.9 m/s: the joint blend from reaching to the hanging pose. Explicit targets where
  the hands hang, 2 cm up; a target 6 cm higher or 4 cm further forward missed by 25
  to 130 mm, and at one seed the fingers-forward task alone put the wrist on a limit
  and missed by 16 cm, so the last resort solves the position alone from hanging.
- **He stood inside the bench to reach the cover tote.** The stance for the far totes
  was (-0.55, -0.20), 25 cm inside the bench's footprint, since gate 2; he does not
  collide with the bench, so nothing said so. From in front of the bench the far totes
  were 16 to 20 cm out of reach, because the rack stood against the bench's left end.
  The rack now stands 30 cm further toward him, both stances are in front of the bench,
  the rack trips got 0.2 to 0.6 s more so his hands stay under 1.6 m/s, and the gate 1
  to 4 numbers that involve the rack were measured with the old position.
- **Waiting with his hands on the bench edge put them 26 cm from the front cover
  holes**, inside the planner's margin: after one cover screw the robot found nothing
  clear for 15 s while he stood there. He waits with his hands at his sides now.
- **Two noise pixels at the tool's edge read as a hand touching the arm.** The camera's
  lateral noise sampled the tool's front face at its edge, 70 mm in front of the
  rendered edge; the grid kept the two cells as "hidden under the arm" and the arm
  backed away from itself, 2.7 s lost before a rail screw. A return within 80 mm in
  front of the robot's own rendered surface is the robot now (notes.md).
- **Turning from the rack to the bench swung his free hand 6 mm from the robot's
  shoulder.** With the rack moved, the turn is 105 degrees, and the joint blend of
  the turn and the walk swept the hanging right hand in an arc over the jig's back
  edge. He turns in place first, part held in front of him and the other hand at his
  side, then walks; the nearest a hand comes to the base axis is 30 cm now.
- **The worktop beside the jig plate read as the person once in a few hundred
  frames**, two pixels at a time, and the arm backed away from a corner of the jig in
  the middle of driving a rail screw, twice. The camera's lateral noise hands a pixel
  a neighbour's depth, so beside the 12 mm plate edge the worktop reads 16 mm nearer,
  and two sigma of axial noise on top crossed the fixed 40 mm gate. The gate is per
  pixel now: three sigma plus how much nearer anything within 3 px of the pixel is,
  computed from the empty station's own image. On the empty station that takes the
  cells passing the grid's rule from 1.17 per frame to 0.00; at that spot, from 91
  pixels in 300 frames to 0; it costs 6 % of his pixels, the ones against depth edges.
  The parts' silhouettes grow 4 px instead of 2 for the same reason.
- **Waiting against the bench, his head kept the arm off the two left cover holes.**
  Not a bug: the 15 cm hidden zone behind anything the camera sees, plus the margin,
  is a keep-out about 33 cm along the line of sight behind his head, and the holes
  were 23 cm from it. He steps back 30 cm to watch; from there every left-hole option
  is clear and he is still fully in view.
- **The worker's "hanging" right arm was raised behind him on two rack trips.** The
  script's stand pose reset only the shoulder and kept the elbow bent from the last
  reach, so with his torso turned to the rack the hand stood 1.42 m up in the robot's
  side of the cell. An arm with nothing to reach now hangs whole (0.84 m); an arm
  with a target keeps its joints as the starting guess, since from a straight arm
  the left hand's reach to its rest spot missed by 54 cm. All 19 reaches solve to
  0.0 mm. Gates 2 to 4 were measured with the old motion.

### Gate 4

- **The depth noise model, extrapolated, turned half the hall into a person.**
  Nguyen et al. fitted their axial noise between 0.5 and 2.75 m; at the 6 m far wall
  it gives 60 mm, which beat the 40 mm background threshold and left 16 793 false
  foreground pixels against 10 451 real ones. Fixed by holding the model flat past
  the range it was fitted in, cutting the depth image off at 4 m, and making the
  background test three standard deviations of the noise at that distance rather
  than one fixed number. False foreground is now 427 pixels.
- **Unprojecting was half a pixel out.** Using `u - width/2` instead of
  `u + 0.5 - width/2` put every point 4 mm out at 2.5 m. With the half pixel it is
  1.6 mm, which is the floor of this depth path.
- **A camera pose read before `mj_forward` is silently the origin.** Camera frames
  come from `mj_camlight`, not `mj_kinematics`, so the whole point cloud collapsed to
  (0, 0, 0). `sensors.camera_pose` now calls it and refuses to return a zero pose.
  Same root cause as the planner's wrist camera in gate 3.
- **The background map included the box.** It was taken with geom group 0, which the
  parts belong to, so the box was background and the box pose fit had only the
  worker's hands to look at. It is now taken with the jig empty, which is how a real
  cell takes it.
- **The box pose fit locked onto hands.** Taking the highest points in the jig volume
  put the fit 50 mm out and the yaw 80 degrees out whenever a hand rested on the
  cover. Fixed by taking the flat face with the most points, checking its height
  against the part drawing, and fitting the known 200 by 150 rectangle to it rather
  than reading a principal axis. A face that is not that size is reported as not
  visible, and the last good fit stands.
- **The hole detector took the centroid of everything dark.** Shadow under the tool
  and the neighbouring hole are dark too, which is why gate 3 measured 7 to 16 mm of
  error. Fixed by taking the connected blob nearest the prediction and checking its
  area against the hole's own diameter at the current distance. Error is now 0.5 to
  3.8 mm where it finds one.
- **The gate 4 box error compared against the wrong part.** The truth read the cover's
  own body, which spends half the cycle in its tote, so the error read 1047 mm. It now
  reads whatever is in the jig, and says nothing when the jig is empty.
- **Two cover screws were built into the part.** The cover mesh always carried two
  driven screw heads, left over from posing the gate 1 stills, so the robot was
  driving screws into holes that already had screws in them. Now a layout option,
  off by default, and the gate 1 stills ask for two.

### Gate 3

- **The screwdriver had no collision geometry below its body.** The bit tip sat
  74 mm outside every robot collision geom, so R1, which must cover the arm, the
  screwdriver, the wrist camera and the screw (docs/design.md section 4), would have
  scored a poke with the screw as clear. Fixed by adding capsules along the nose
  and the bit and a box for the wrist camera. The bit capsule is 8.5 mm fatter than
  the screw, so contacts are flagged slightly early, which is the safe direction.
- **The enclosure's collision shape was a solid block.** The rail screws are inside
  it, so once the tool had collision geometry every pose that reached them counted
  as a crash and the reach table read 0 of 24. Fixed by modelling the base as a
  floor, four walls and the screw bosses, and the DIN rail as a strip plus its
  block cluster, so the two screw slots stay open. The number is 17 of 24.
- **The per-hand visibility sample silently dropped the thumb.** `worker_seen` asked
  for the hand and fingers but not the thumb, so a hand went from 201 sampled points
  to 168 and every per-hand number moved by a few points, including the worst case
  reading 0.0 % where the truth is 6.5 %. Found by re-running gate 1 and failing to
  reproduce its own stored numbers. Fixed, and gate 1 now reproduces exactly.
- **`sightline_sim/gates/gate1_numbers.py` could not reproduce the gate 1 record.** It called
  the visibility measurement with the default ray stride and no field-of-view check,
  while results.md documents every 5th vertex and points counted only if in shot.
  Fixed, and the stride is written into the results file.

- **The planner's wrist camera never moved.** `mj_kinematics` updates bodies and
  sites but not cameras, so the planner's own camera pose stayed at the origin with
  a zero rotation and every hole projected to behind the lens. Hole detection
  returned "nothing here" on every frame and the visual servoing never ran, quietly.
  Fixed by calling `mj_camlight` as well. Found by asking why gate 3 reported zero
  hole detections and zero misses, which is not a number a working detector gives.
- **B0 stalled 17 mm short of every target.** Driving the tool along a straight line
  with a damped least squares step walks joints into their limits and stops there, so
  the first gate 3 run drove 0 of 6 screws and stood in the worker's way instead.
  Fixed by solving the pose and moving joint to joint, the way a taught cell does,
  with straight-line insertion only where it matters.
- **A joint move bowed the bit 37 mm sideways going into one hole.** Two poses 60 mm
  apart can sit either side of an arm configuration change, and interpolating across
  it swings the whole arm. Fixed by scoring every turn angle by how far the joints
  move between the approach and the hole, and taking the quietest, which is what a
  programmer teaching the cell does by eye. Worst insertion wander is now 0.99 mm.
- **R2 called every blocked frame "the person moved in".** The rule rewinds the robot
  10 ms, which is 2.5 mm of arm travel, and by the time a 30 Hz camera catches an
  overlap it is wider than that. Fixed by labelling each blocking event once, when it
  starts, and by reporting the same question one camera frame back alongside it. For
  open: docs/design.md section 4 defines the 10 ms test, and it does not answer this for
  R2. Both numbers are in the results until decided.
- **Segmentation rendering crashed, and anti-aliasing was the reason.**
  `Renderer.render()` raised `IndexError: index 579 is out of bounds for axis 0
  with size 531`. The renderer decodes each pixel's colour back into a geom id, and
  with `offsamples = 8` the multisampling blends neighbouring ids into values that
  belong to no geom. Gate 1 worked around it by ray casting. Fixed properly by
  building the segmentation renderer while `model.vis.quality.offsamples` is 0 and
  restoring it afterwards, so the colour renderers keep their anti-aliasing. The R2
  judge needs pixels (docs/design.md section 4), so the workaround could not stand.

### Gate 2

- **The cycle asked for targets the arm cannot reach.** The first script sent the
  worker's hand to the rack totes from the bench stance: 0.82 to 0.96 m from the
  shoulder against a 0.77 m envelope, so the arm hung in the air and the reach
  error hit 276 mm. Fixed with two rack stances, a side step, the right arm left
  hanging while he is at the rack, and the tray handed to the right hand instead
  of the left. Test: every target in the cycle is inside the measured arm envelope
  (`test_station.py::test_every_cycle_target_is_within_arm_reach`).
- **Hands slid sideways off parts and cut straight through them.** Key poses blend
  joint by joint, so a hand leaving a part travelled through it: 57.8 mm into the
  box. Fixed by adding a lift-off via pose at the start of each transit, which is
  what a real hand does. Worst case is now 36.5 mm, into the bench top, while the
  worker's body hides the moment from the camera. Test: the lift segments exist and
  fit inside their own duration (`test_station.py::test_lift_segments_add_a_via_pose`).
- **A failed hand solve poisoned the retry.** The right arm was solved from a
  reaching seed. Trying the rest pose first, to stop the hand swinging a wide arc,
  left the arm folded up when that solve failed, and the seeded retry then started
  from the folded pose and missed by 757 mm. Fixed by restoring the pose before the
  retry.
- **`Playback` grew via poses, so the segment lookup broke.** With extra keys in the
  timeline, `segment_index` mapped time to the wrong segment, which would have fired
  grabs and releases at the wrong moment. Fixed by keeping segment boundaries in
  their own list (`seg_times`) apart from the interpolation keys.
- **A citation in results.md was wrong.** Hand speeds were attributed to Nguyen et
  al. 2012, which is the Kinect depth noise paper and says nothing about hand speed.
  Removed and replaced with what the speeds actually are: a property of the script.

### Gate 1

- **The camera was placed behind its own housing.** The eyes camera sat at
  `pos - fwd * depth` instead of in front of the lens, so the first thing it saw
  was the back of its own case: a black band across the top of every frame.
  Fixed by placing the housing behind the lens plane. Test: a ray from the camera
  to the jig, with the camera body excluded, must reach the jig
  (`test_station.py::test_eyes_camera_sees_the_jig`).
- **The camera pole's bracket filled the lens.** The first pole mount put a 9 cm
  plate 5 cm under the camera. Fixed by standing the pole behind the camera
  instead of under it. Same test as above.
- **Hand inverse kinematics stalled 0.3 m from the target.** Solving position,
  finger direction and palm orientation as one weighted task got stuck against
  joint limits. Fixed with a prioritised solve: position first, the rest in its
  null space, then a position-only pass because clipping at joint limits can
  undo the primary task. Test: every work pose reaches within 1 mm
  (`test_station.py::test_work_poses_reach`).
- **Ear clipping failed on axis-aligned profiles.** A vertex lying exactly on a
  candidate diagonal was not treated as blocking, so the aluminium profile
  triangulation produced a self-intersecting remnant and raised. Fixed by
  blocking an ear when a vertex is inside it or on its diagonal. Test: the T-slot
  profile triangulates (`test_geometry.py::test_tslot_profile_triangulates`).
- **Three scene builds in one process ran the machine out of memory** (7 GB, 
  shared with other work). Not a code bug: scripts now build one scene per
  process, and `eyes_options.py` takes the variant names as an argument.
