# Model notes

Every number here either carries its source, taken from a document that was
opened, or is marked **scene choice**. Scene choices are picked to look and
behave like a real cell; they are not measurements.

## Worker

- Model: dm_control's CMU humanoid `CMUHumanoidPositionControlledV2020`, the one
  CobotSafe uses, so CobotSafe's body regions and limits apply.
- As shipped it stands **2.05 m** (measured in the standing pose). Scaled by
  **0.8531** to a stature of **1.75 m**. That height is a scene choice inside the
  published range of mean adult male height for the 1996 birth cohort, about
  1.60 m to 1.825 m (NCD Risk Factor Collaboration, "A century of trends in adult
  human height", eLife 2016;5:e13410).
- Scaling multiplies every length by k and divides every geom density by k^3, so
  the total mass stays **70 kg** (checked in tests).
- Standing pose (feet about 0.17 m apart) from CobotSafe `scene.STANDING_POSE`;
  hanging arms from `human_motion.ARMS_AT_REST`. The model's foot is pitched
  toes-down in that pose, so `pose._foot_pitch_flat` solves the ankle angle that
  puts heel and toes level before the feet are set on the floor.
- Clothing, skin and hair are visual meshes on the same bodies (group 1). The
  humanoid's own capsules stay as the collision body (group 3), hidden in renders.
  Colours and shapes are scene choices.
- The right arm's joint ranges are not mirrored from the left (noted in
  CobotSafe `human_motion.py`), so right-hand reaches start from CobotSafe's
  measured `RIGHT_ARM_REACH` pose.

## Bench and station

- Work surface at **1.118 m**: the measured elbow height of the scaled worker is
  **1.193 m**, minus **7.5 cm**. Source for the rule: CCOHS, "Working in a
  Standing Position - Basic Information" (updated 2022-11-30): for light work the
  work surface sits about 5 to 10 cm below elbow height. The 7.5 cm is the middle
  of that range, a scene choice inside a sourced band.
- Worktop 1.60 x 0.80 m, 40 mm aluminium profile frame, top frame 1.10 m above
  the worktop, ESD mat, jig plate 0.32 x 0.25 m, bins, flow rack, out tray, screw
  feeder, stack light, station screen: all **scene choices**.
- UR5e control box **460 x 449 x 254 mm** and teach pendant **300 x 231 x 50 mm**
  (UR5e technical specification, updated May 2025).
- The hall (floor, walls, windows, racking, columns, neighbouring stations, cart,
  floor markings) is a **scene choice**.

## Robot

- UR5e from MuJoCo Menagerie. Reach **850 mm**, every joint **+-360 deg** and
  **+-180 deg/s**, maximum TCP speed **4 m/s** (UR5e technical specification,
  updated May 2025). The planner will use lower, stated limits near the person.
- Mounted on the worktop at the rear centre on a 200 x 200 x 15 mm plate: scene
  choice, confirmed against the reach table in gate 1.
- **Gap:** the tool's 2.5 kg is not in the model yet. The Menagerie wrist keeps
  its own inertial, and the tool geoms have zero density. Fix before any dynamics
  matter.

## Screwdriver

OnRobot Screwdriver, datasheet v1.7 (p. 2 for the specification table, p. 34 for
the drawing). Numbers used:

- Overall **322 mm** from the top to the bit holder tip, body **86 mm** wide and
  **114.1 mm** deep, nose **diameter 49 mm** reaching **60.4 mm** below the body,
  bit holder **13.5 mm** long and **13.5 mm** across.
- The robot flange sits on the **back face**, with its centre **166.6 mm** above
  the bit holder tip.
- Screw axis about **93.5 mm** from the flange face: measured off the drawing on
  p. 34, so about +-2 mm. The quick changer thickness of 12 mm inside that number
  is a **scene choice**.
- Consequence, and it matters for the whole project: the tool mounts from the
  side, so turning the robot around the screw axis moves the whole arm, not just
  the wrist. Reach therefore depends on the turn angle (gate 1 reach table).
- Torque, speed, the 55 mm shank stroke and the screw feeding are **not
  simulated**. Screws are placed geometry.
- No branding is modelled; the shape is the product's, the logo is not.

## Cameras

- Eyes camera: Intel RealSense **D455**, depth field of view **86 x 57 deg**,
  housing **124 x 29 x 26 mm**, working range **0.6 to 6 m** (Intel product
  specification page). MuJoCo camera `fovy` is set to the vertical 57 deg.
- Wrist camera: Intel RealSense **D405**, **87 x 58 deg**, **42 x 42 x 23 mm**,
  ideal range **7 to 50 cm** (Intel product specification page).
- Eyes camera position: on a pole beside the bench at x 1.15 m, y -0.90 m,
  **3.25 m** above the floor, looking down at **56 deg**. A **scene choice**,
  picked by comparing three mounts on the same worker pose (results.md).
- Wrist camera position: **75 mm** from the screw axis, **52 mm** above the bit
  tip, aimed at a point 35 mm below the tip: **scene choice**, set so the nose
  does not hide the hole.
- **Gap:** no sensor noise or delay yet. design.md section 8.2 fixes the model
  (Nguyen, Izadi and Lovell 2012) and it is added in gate 4.

## Parts

All **scene choices**, not yet matched to a product:

- Enclosure 200 x 150 x 80 mm, wall 2.8 mm, four corner bosses for the cover and
  two bosses for the rail, cover 4 mm thick with four 4.6 mm holes.
- DIN rail modelled as the common 35 x 7.5 mm top hat with two mounting slots.
  The size matches the usual rail; EN 60715 has not been opened, so this stays a
  scene choice.
- Terminal blocks 5.2 mm wide, screws about M4 x 12 with a pan head.
- Screw feeder, totes, labels: shapes and sizes invented.

## How things are measured

- Visibility is measured by ray casting, not by counting pixels: a ray runs from
  the camera to each point of the worker's visible surface (clothing mesh
  vertices, every 7th), and the first geom it meets names the blocker (robot, the
  worker's own body, or the station). The R2 judge in design.md section 4 still
  uses rendered segmentation; this is the design-time tool.
- Reach is measured by damped least squares inverse kinematics on the `tool_tip`
  site from 128 seeds, then a MuJoCo collision check of the robot against the
  fixed station.

## Rendering

- Geom groups: 0 station and hall, 1 worker, 2 robot and tool, 3 collision
  shapes. Renders show 0, 1 and 2.
- Lighting: two spot lights on the bench with shadows, a directional hall light
  without shadows, two fill lights, window light. All **scene choices**.
- Shadow artefacts: tightening `map.shadowclip` produced striped shadow acne, so
  it stays at the default and only the two bench lights cast shadows.

## Collision shapes, corrected 2026-09-18

- The screwdriver's nose, bit holder and the screw on the bit are capsules along
  the screw axis, radius 24.5 mm for the nose and 8.5 mm for the bit. The bit
  capsule is about 3 mm fatter than the real bit and its rounded end reaches
  8.5 mm past the tip, so R1 reports a contact slightly **before** the screw
  itself touches. That is the safe direction, and the reach tables count it.
- The wrist camera has a collision box the size of its housing. R1 covers the arm,
  the screwdriver, the wrist camera and the screw (design.md section 4).
- The enclosure's collision shape is a floor, four 2.8 mm walls and the six screw
  bosses, not a solid block, because the rail screws are driven inside it.
- The DIN rail is a 160 x 35 x 7.5 mm strip plus one box over the four terminal
  blocks, so the two screw slots at x = +-70 mm stay open.
- Part masses that follow from these shapes: base 0.758 kg, rail 0.131 kg, cover
  0.072 kg. Densities are **scene choices** (2700, 2000 and 1200 kg/m3).

## B0, the baseline planner

All **scene choices** unless a source is named:

- Joint speed limit 90 deg/s. The UR5e datasheet allows 180 deg/s on every joint
  (UR5e technical specification, May 2025); a cell that works next to a person is
  run slower, and the later variants use the same number so the comparison is fair.
- Tool speed limit 0.25 m/s for fine motion. The datasheet's maximum is 4 m/s.
- Stand-off above a hole 60 mm, screw length 15.6 mm, drive time 1.2 s, feeder
  dwell 0.4 s, "arrived" tolerance 1.5 mm and 0.004 rad per joint.
- Long moves are joint moves to a solved pose. Going into a hole and coming back
  out are straight lines, built as five solved waypoints along the line, because
  interpolating joints between the two ends bows the bit up to 1.8 mm sideways and
  across a configuration change up to 37 mm.
- The turn angle around the bit is chosen once per hole, as the angle whose
  approach pose and hole pose are closest in joint space. Taking the first angle
  that solves instead put one screw next to a configuration change.
- A wrist camera correction larger than 4 mm is refused. The gate 3 detector is a
  centroid of dark pixels and lands 7 to 16 mm off, so in gate 3 every correction
  was refused and the robot drove from taught positions. Gate 4 replaces it.
- The robot is played back kinematically: commanded joint velocities are
  integrated and written, so tracking error is zero and there is no contact force.
  Dynamics and the stopping distance the monitor needs come with gate 5.


## The cameras, as the planner sees them (gate 4)

- **Eyes camera:** 640 x 480 colour and depth at 25 Hz with one frame of delay
  (a design choice in gate 5; gates 2 to 4 ran at 30 Hz). Depth noise from Nguyen, Izadi and Lovell 2012: axial
  sigma_z = 0.0012 + 0.0019 (z - 0.4)^2 m (Eq. 3), lateral
  sigma_L = 0.8 + 0.035 theta / (pi/2 - theta) px (Eq. 1). They fitted it between
  0.5 and 2.75 m, so it is **held flat past 2.75 m**: run out to the 6 m hall wall
  it gives 60 mm, which beats any sensible background test and turns the floor into
  a person. No returns past 4 m or across a depth step of more than 20 mm per pixel,
  both scene choices standing in for what a real camera loses at edges and range.
- **Unprojection** uses the pixel centre convention, u + 0.5 - width/2. The floor of
  this depth path, measured against ray casting, is 1.6 mm at 2.5 m.
- **Calibration error** given to the planner: 2 mm in position and 0.15 degrees in
  rotation (scene choice).
- **Background:** one depth map of the cell with nobody in it and nothing in the jig.
  A point counts as foreground if it beats that map by a per pixel gate: at least
  40 mm, and at least three standard deviations of the noise at that distance plus
  how much nearer anything within 3 px of the pixel is (gate 5: the lateral noise
  hands a pixel a neighbour's depth, and beside the jig plate's edge the worktop
  read as the person). Median gate 63 mm, 141 mm at the 90th percentile, and
  pixels right on a big depth edge, the bench against the floor, are blind. Measured
  cost on the balanced cycle: recall 87.6 % against 88.1 % with the fixed gate,
  precision 99.7 % against 99.0 %.
- **Self filter:** points within 50 mm of the robot's own collision shapes at its
  current joint angles are its own arm. A real cell does this from its URDF. Replaced
  in gate 5 by the known-world render (below): a return is the person only if it is
  clearly in front of the rendered arm and parts, and a return within 80 mm in front of
  the rendered arm alone is still the arm, since at a link's edge the lateral noise
  samples the link's front face, up to a link radius nearer than the edge.
- **Person voxels** 2 cm; a voxel needs two points. **Unseen space** 30 cm behind
  each person voxel along the camera ray, treated as occupied for R1 (Flacco et al.
  2012). Both scene choices.
- **Box fit:** the flat face with the most points inside the jig volume, its height
  checked against the part drawing, and the known 200 by 150 mm rectangle turned to
  fit it. Refused if the face on show is not that size. Yaw is not observable from
  3.25 m and is not used; the jig sets it.
- **Hole detector:** the connected dark blob nearest the predicted hole, accepted on
  area against the hole's own diameter at the current distance, on roundness, and on
  being within 18 px of the prediction.


## The motion layer (gate 5)

All **scene choices** unless a source is named. The QP is the dual active set
method of Goldfarb and Idnani 1983 in numpy, checked on random problems against
the optimality conditions (1e-13) and against scipy's SLSQP (1e-10).

- **Safety distance d_s = 0.10 m** to anything treated as a person. It has to
  cover what can change between one look and the next: the person's travel during
  the camera's one frame of delay plus one motion cycle (43 ms at the 1.31 m/s peak
  hand speed measured in gate 2: 56 mm; at the ISO 13855 2000 mm/s setting: 86 mm),
  the 20 mm voxel, the 10 mm of depth noise at 2.5 m, and no tracking error, since
  the robot is played back. So 0.10 m suits the measured speed setting and the
  2000 mm/s setting will need 0.13 m. Both go into gate 7's two settings.
- **Influence distance d_i = 0.35 m**, damper gain xi = 1.0: the closing speed is
  capped at xi (d - d_s)/(d_i - d_s), which reaches zero at d_s (Faverjon and
  Tournassoud 1987).
- **R2 margin r_vis = 0.06 m** off any sight line from the eyes camera to a person
  voxel, influence 0.20 m. Three voxels of clearance.
- **Static:** the worktop plane and the bench, 0.02 m safety, 0.10 m influence.
- **Joint limits:** 90 deg/s, 400 deg/s^2, and the position limits, as velocity
  bounds every cycle. The datasheet allows 180 deg/s.
- Only the nearest four obstacles per link enter the QP: 25 rows median, 65 worst,
  which solves in 6.8 ms median and 12.6 ms worst on this machine.
- A pair already inside d_s is not ordered to open up by the constraint, which
  cannot be satisfied when several pairs disagree; the constraint forbids closing,
  and the objective asks for the retreat with weight 4.0 per metre of shortfall.
- Fallbacks, counted: if the task and the rules cannot both hold, the rules alone are
  solved with three times the acceleration a protective stop is allowed; if that has
  no answer either, the arm brakes.
- Held time: the task wanted to move and the rules let through less than 10 % of it.
  After 2 s of that the cell asks the worker to move his hand. Both are counted per
  episode.

- **Corrected during gate 5:** the layer's capsules were all balls. The test
  `geom_type[g] in (mjGEOM_CAPSULE, mjGEOM_CYLINDER)` is always False in these
  bindings (`==` works, `in` does not), so every link was a ball of its own radius at
  its centre and the ends of the 400 mm upper arm and 380 mm forearm were not in the
  model at all. The same test sat in B0's clear check, B4's wrist view check and the
  first version of the guard. All four now take their shapes from one helper,
  `kin.shape_line`. Every gate 5 number from before this is superseded.


## The camera grid and the guard (gate 5, B4)

The grid algorithm: the camera's view as a grid of directions, the person's
directions off limits from the camera out to just behind him, rebuilt with every
picture, and the arm checked against it every 10 ms. The same idea as the depth space
approach of Flacco et al. 2012 (section 19, method reference; no numbers taken).
All **scene choices** unless a source is named.

- **Cells** 4 x 4 pixels, about 2.3 cm at 3.2 m. A cell holds him if two of its
  pixels are foreground and a neighbouring cell holds him too. The neighbour rule
  took stray cells from 18.9 to 1.75 a picture over the eight work poses and lost
  none of his (40.42 against 40.46 cells missed of 474, the ones the 40 mm
  background gate already hides). Three pixels a cell lost 15 more of his.
- **Off limits:** a point of the arm is off limits if, for some cell of him, its line
  of sight passes that cell closer than the margin, measured at the point's own
  depth, and it is nearer the camera than the cell's far end plus the margin. The far
  end is 0.15 m behind the seen surface, a limb's thickness.
- **Margin** = 0.10 m safety distance + the size of that part of the arm + how far
  that part of him could have moved since the picture. Each point carries its own
  size: 8.5 mm for the bit, 32 mm for the wrist camera, a link's radius plus half the
  spacing between its points. One margin for all points, the thickest link's, kept
  the tool 6 cm further from him than it needs to be.
- **Points on the arm:** 63. Each joint, the tool tip, the wrist camera, and a point
  at least every 5 cm along every collision capsule. The 5 on the shoulder's own axis
  are left out of the guard's decisions, since no joint moves them.
- **Speed of each part of him:** each cell's distance to the nearest cell of him
  three pictures back (120 ms), where each old cell counts both its nearest and its
  farthest pixel of him; a reading needs a neighbouring cell that agrees (the second
  largest of its 3 by 3); spread over 0.15 m round it (a hand and wrist share their
  leading edge's speed), held for 0.2 s, capped at the ISO 2000 mm/s. Read only from
  the parts of him above the worktop minus 0.1 m and in blobs of six cells or more.
  Against his true hands over one parked cycle: still hands read 0.26 m/s median,
  0.48 at p90; moving hands 1.27 times their true speed; below the hand's true peak in
  10 % of pictures, by up to 0.40 m/s. The acceleration term of design.md section 8.6 (b)
  is zero until a published bound is found.
- **What the arm hides:** a cell of him that the arm covers in the next picture is
  kept at its own depth and speed, for 0.5 s at most or until it is seen into again,
  and counts for touching only.
- **The fixed blind area** (the shadow of the bench's top frame, 480 cells of 5 cm;
  the 8 inside the jig and box are left out): a blind cell counts as occupied when a
  cell of him in the grid is within 20 cm, a hand's length; a point of the arm is off
  limits within 0.10 m + its size + 43 mm of one.
- **The worktop:** no point of the arm below the worktop by more than its size.
- **Every 10 ms** the guard takes the task's command (after the joint speed,
  acceleration and position limits and the worktop) and checks the arm where it
  would be if it kept that command one cycle and then stopped at 1200 deg/s^2, three
  times the working ramp, one check point per 40 ms. The rule: no point of the arm
  deeper inside the off-limits space than holding still would leave it at the same
  moment, so a clear point stays clear and a point he has come too close to gets no
  closer. If the command fails, half and a quarter of it, then braking. If the arm is
  already inside the margin, it scores ways out 0.1 and 0.2 s ahead by the depth left:
  braking, back to the latest pose it passed through that is clear now, straight up
  toward the camera, the screw's stand-off (at the hole) and park; one that takes
  1 cm off goes first. The 5 points on the shoulder's axis are left out; the 8 that
  stay within 0.2 m of it may not close on him but do not send the arm fleeing.
- **Planning** asks the grid as it will be 0.15 s ahead, the guard's own stopping
  horizon: the chosen hole's approach and screwing poses and the joint line there,
  sampled every 0.06 rad of the fastest joint; a point that moves less than 3 cm on
  the way may stay as deep as it is. Every new picture it asks again about the rest of
  the way and swaps the hole if the block is less than 0.5 s of travel ahead in two
  pictures running. A hole given up waits 1 s before it can be chosen again.
- **Options:** every turn angle round the bit, 15 degrees apart, the arm
  configuration nearest the feeder pose that goes down into the hole smoothly and
  keeps the arm above the worktop: 20 to 24 per screw, 7 to 12 of them with both
  wrist joints on the robot's side of the hole.
- **The feeder pose** is B4's own: of every turn angle and configuration, the one
  furthest behind the jig's back edge (6 cm). B4 fetches the next screw while he works
  and waits there with it.
- **Checked against the truth** on the eight work poses, every hole option: the grid
  never called a pose clear that was within 10 cm of him or blocked the camera's view
  of him (0 of 104), and called 2 of 104 off limits that were fine (101 and 139 mm),
  while the pose changes inflated the measured speed.


## The worker's cycle with the clamps (gate 5)

All **scene choices**. The jig's two toggle clamps, on its right side, hold the cover
(a design decision). His tasks are ordered so that the robot's two screwing windows
fall while he has work beside it (a design decision, design.md section 18):

- base and rail in the jig as before (17 s); pressing the rail down is the switch for
  the rail screws;
- while the robot drives the rail screws: step over to the prep area (2.4 s), clip
  blocks on the next rail (3.4 + 3.4 s), drop the hands to his sides (1.6 s), step
  across for the cover (2.8 s), take it (1.4 s), bring it (2.6 s) and press it on
  (1.8 s). The cover placed is the switch that ends the rail screws: they are under it;
- close the front and rear clamps (1.3 + 0.9 s); the rear one is the switch for the
  cover screws;
- while the robot drives the cover screws: step to the prep area (1.8 s), fit the end
  stops, the jumper bars and the labels (3.0 + 3.0 + 3.4 s), drop the hands (1.6 s),
  step back 30 cm from the bench (1.8 s) and wait there upright with the hands at his
  sides, watching (11.0 s, `WAIT_FOR_COVER_SCREWS_S`);
- step in and open the clamps (2.6 s), which is the switch that ends the cover screws,
  lift the box out and put it in the tray.

The wait is set to the time the robot's four cover screws take; a real cell would end it
on the stack light, which a fixed script cannot. A slower robot loses the screws it has
not driven when the clamps open. With the seed's jitter the cycle is 78 to 81 s, his
hands peak at 1.2 to 1.6 m/s, every reach solves to within 1 mm on seeds 0 to 5, no
hand passes behind the jig, and the nearest a hand comes to the robot's base axis is
30 cm.

Coming back from the rack with a part he turns in place first (2.2 s, 2.0 s with the
cover), part held in front of him and the free hand at his side, then walks the last
step: turning and walking as one blended move swung the free hand over the jig's back
edge to 6 mm from the robot's shoulder. The prep area sits at x = -0.55 m, 15 cm
further from the jig than first placed: at
-0.40 m his upper arm, clipping blocks, was 17 to 25 cm from the robot's wrist at a
rail screw. Dropping the hands goes by targets at his sides (where they hang, 2 cm
up); the blend from reaching to the hanging pose swung the right hand 45 cm sideways
over the jig at 1.9 m/s. He waits 30 cm back from the bench with his hands at his
sides: on the bench edge the hands sat 26 cm from the front cover holes, and against
the bench his head sat 23 cm from them, inside the line-of-sight keep-out behind him. The parts rack stands 30 cm further
toward him than in gates 1 to 4 (its near end at y = -0.82 m): against the bench's
left end, its far totes were reachable only from inside the bench's footprint, which
is where the script had him standing. From where he works the rack is still 1.07 m
away, out of reach, so he side-steps to it as before.

**The bound for any planner on this cycle**, true worker every 0.2 s, some screw of the
stage with every part of the arm at least 10 cm from him and off the camera's lines of
sight to him: the rail screws in 74 % of their 19.0 s window, longest stretch 8.8 s,
the whole of it while he clips blocks; the cover screws in 97 % of their 28.0 s window.

## Gazebo as a second renderer (2026-09-18 to 19)

The next step was the work to run and be recorded on ROS 2 tools. What exists:
Gazebo Sim 10 and rviz2 show the run the judge scored in MuJoCo, from a recording of
every body pose at 25 Hz. Gazebo simulates nothing here: its physics runs with no
collision shapes and no gravity, and every frame's poses are set from the recording.
Anyone reading a Gazebo frame reads MuJoCo's state.

**Export.** `station.export_mjcf` writes the cell as one MJCF file with an assets
folder (PNG textures instead of buffers, every mesh as an OBJ with area weighted
normals, the Menagerie UR5e meshes copied beside them). Three things the converter
needs: no "/" in names (the attached parts are `worker__lhand`, `ur5e__base`), every
mesh in a file (the worker's 39 clothing meshes were vertex lists in the model), and
the visual geoms in group 0 (the converter takes group 0 as visual, 3 as collision
and drops every other group; the worker's skin was group 1 and the robot's group 2,
and both came out invisible the first time). The exported model reloads with the same
bodies, geoms and meshes, sizes within 5 um (`sightline_ros/test/test_ros2.py`).

**Conversion.** Gazebo's `sdformat-mjcf` 0.1.2 against the sdformat 16 binding
vendored with ROS Lyrical. `sightline_sim/gazebo/mjcf_to_sdf.py` holds the shims it needs there: the
module names it imports (`sdformat13`, `gz.math7`) mapped to the ones installed,
`gz.math` imported before `sdformat`, and every setter given plain floats where
dm_control hands it one element arrays. Free joints are skipped with a warning; the
replay does not need joints at all.

**The replay world.** `sightline_sim/gazebo/make_replay_world.py` makes one model per MJCF body that
has visuals (41 of 51; the rest are joint frames), placed at the body's world pose,
with the converter's visuals verbatim and no collisions; the hall's lights, prefixed
`light_` because a light and a model may not share a name; the two cameras as sensors
at MuJoCo's camera poses (MuJoCo cameras look down their -z with y up, SDF cameras
down +x with z up: the matrix in the script), the horizontal field of view from
MuJoCo's vertical one and the aspect ratio; a 1 ms physics step; and a `<gui>` block
that opens the desktop view where the hero camera is. Gazebo's sensors render only
when something subscribes.

**Playing it.** `sightline_sim/gazebo/replay_gazebo.py` starts the world paused, and for every frame
sends the poses of all 40 body models in one `set_pose_vector` request, then a
`run_to_sim_time` of the frame's time, and waits for each camera's picture with that
stamp, which `sightline_sim/gazebo/frame_grabber.py` saves in a process of its own: a Python callback
receiving images and a blocking request in the same process wait on each other for
the interpreter lock. The stamp match is exact; a picture a whole period past the
one wanted is reported, not used. Two 720p cameras with 4x anti-aliasing run at 3.8
frames a second on the GTX 1650 (2023 frames in 531 s), none missed.
`sightline_sim/gazebo/assemble_video.py` puts the judge's words on the frames, the same words as the
MuJoCo video, plus one line saying where the picture came from. The eyes video has
no grid tint: the grid is the planner's and is not in the recording.

**The desktop.** `sightline_bringup/scripts/record_windows.sh` runs a server with headless rendering, the
Gazebo GUI on it, `robot_state_publisher` with ur_description's UR5e (its root link
renamed to our `robot_mount` frame; its `base_link` sits exactly on MuJoCo's base
body, checked with forward kinematics against random poses), `ros_gz_bridge` for the
eyes camera picture, rviz2 with `sightline_bringup/config/sightline_replay.rviz`, and `sightline_ros/replay_live.py`
in real time: poses into Gazebo as above without waiting for pictures, and to ROS 2
`/joint_states` (the six angles recovered exactly from the recorded body orientations,
`sightline_ros/test/test_ros2.py`), `/tf`, the worker as the judge's 47 capsules and spheres, the
station's boxes and cylinders (sent again every 2 s for a late rviz), and the judge's
words as a text marker; 3 ms a frame on the ROS side, the messages built once and
re-posed. The windows are captured one by one with x11grab through XWayland, the pattern from
locrec, an earlier screen-recording tool of mine, and the Gazebo GUI opens on
`results/ros2/gazebo/gazebo_gui.config`,
written by the world builder with the hero camera's pose.

**The front camera and the shadows (2026-09-19, a request).** The hero camera
looks over the worker's shoulder and his back hides the arm at times. The replay
world adds cameras of its own, placed in world coordinates (`EXTRA_CAMERAS` in
`sightline_sim/gazebo/make_replay_world.py`): `front` stands on the far side of the bench behind the
robot, at (-0.40, 2.50, 2.25) looking at (-0.15, -0.40, 1.20) with a 42 degree
vertical field, so his face and hands, the arm and the box are in one frame and the
feeder sits at the left. A camera further left put the pendant on the upright across
his face; one further right and lower let the robot's base hide the jig. With
`--shadows` the two key lights cast shadows in the sensors' render (Gazebo's
converter writes every light with `cast_shadows` off); the desktop GUI cannot have
them (its scene plugin crashes on `<shadows>` while a server is up) and opens on the
front view with `--gui-camera front`.

**The desktop recording, and why the Gazebo half is a camera (2026-09-19).** The
user saw the Gazebo window lag in the first desktop recording. Measured: it repainted
at 7.6 fps while rviz beside it did 24.8. The machine is a hybrid laptop: the display
runs on the Intel UHD 630, the GTX 1650 is a render offload device. The headless
server was on the NVIDIA GPU (the EGL vendor file); the GUI window went through X and
rendered on the Intel side, too slow for this scene. Sent to the NVIDIA GPU with PRIME
offload, the GUI renders fast, but a window rendered there reads back through X at 13
fps for 1280x720 and about 1 fps after a resize, with stale frames; that readback is
what x11grab does. Gazebo's own VideoRecorder plugin has no remote trigger in this
version (only its on-screen button), and GNOME's screencast service answers "session
creation inhibited" to scripts. So `sightline_bringup/scripts/record_windows.sh` takes Gazebo's picture
where it is made: the `front` camera of the replay world, rendered by the server on
the NVIDIA GPU as the run plays, encoded straight from the frames by
`sightline_sim/gazebo/frame_grabber.py --video`, one frame per 40 ms of sim time (2073 frames for the
2 s lead and the 81 s run, none missing, none repeated), then captioned with the
judge's record by `sightline_sim/gazebo/caption_video.py`. rviz stays on the Intel GPU and is grabbed
at 25 fps; the grab starts before the driver and is trimmed to the driver's first
frame. Both halves of the result change on every frame at 25 fps. The GUI is not
part of the recording any more; it was only ever a viewer of the same scene.

**Limits.** The Gazebo GUI and rviz show the recorded truth, not the planner's grid.
The system's Python packages must come last on the path (rclpy's message modules
import `em` from there; ahead of the venv they shadow its protobuf, which the Gazebo
messages need). Processes are stopped by PID; a pattern kill matched a shell's own
command line twice in this project.

## The planner live on ROS 2, stage 1 (2026-09-19)

Two nodes, `sightline_ros/cell_node.py` (the cell) and `sightline_ros/planner_node.py` (B4), on topics,
in lockstep on sim time. The point of the lockstep: the run does not depend on how
fast the machine is. The cell publishes `/joint_states` and `/sightline/tick` (JSON:
the tick number, the sim time, the joint state, the jig signal and the index of the
last picture) every 10 ms of sim time and waits for the `/sightline/cmd` that
answers the tick before it steps on; every 40 ms it publishes `/eyes/depth` and
`/sightline/frame` (the picture's index, time, the arm's joints now and a frame ago)
first, and the tick after names that picture. The planner keeps a tick that names a
picture it has not processed until the picture arrives. Messages of different
topics may arrive in any order; this makes the order irrelevant. Nothing waits on
wall time except a 30 s guard against a dead planner (the cell holds, and counts it).

What the cell tells the planner once, kept for a late subscriber: the calibration
it may believe (the true camera pose with gate 5's errors, drawn from the same
random generator in the same order as the gate 5 runner, so seed 0 is seed 0), the
empty station's depth taken at commissioning, the blind cells the furniture hides,
and the taught points (the jig, the holes, the feeder, the park pose). The planner
builds its world from those and says `/sightline/planner/ready`; the cell's first
tick waits for it (B4's build takes 75 s here). The planner imports nothing from the
simulation but the robot-only model of its own arm, as the gate 5 runner does.

Per tick the planner runs the task, the motion layer and the guard exactly as the
runner does (`b4._task`, `solve_joint` with the worktop plane, `safe_command`); per
picture the perception, the grid and the live blind cells. Parity: over the same
6 s the two runs give bit-identical robot and worker poses, states and verdicts
(`results/ros2/stage1/parity`). Cost on this machine: a picture takes the planner
about 95 ms (the perception), a tick 1.3 ms in the guard and 0.4 ms in the QP, the
cell waits 3.5 ms median for a command; the run goes at a fifth of real time. That
is the same compute as before, now with the topics between.

The judge stays in the cell with the truth: R1 and R2 as in gate 5, the held time,
the asks. The cell records every body's pose as `--record-poses` does, so the Gazebo
replay renders stage 1 runs too. A bag of the small topics (everything but the
pictures) replays into rviz at real time (`sightline_bringup/scripts/record_bag_rviz.sh`): the UR5e on the
joint states, the worker's capsules, the grid as three point clouds (where the
camera sees him, what the arm hides and it remembers, the far end 15 cm behind
him), the 63 points the guard checks, the hole it is going for, and two picture
strips with the judge's record and the planner's own numbers. The eyes picture is
not in the bag (2023 pictures are 1.9 GB); the Gazebo replay shows it.

The hardware path, the planner node takes a depth picture and a
joint state in and puts a joint velocity out. On a real cell the picture comes from
the RealSense driver at 30 Hz and the joint state and command go through the UR
ROS 2 driver at 500 Hz; the lockstep goes, wall time rules, and the 95 ms picture
must come down to 40 ms or run one picture behind. That is the next engineering
job, not this one.

## The planner in C++

The Python planner is where the algorithms were worked out and it is what the gate
scripts run. `sightline_planner_cpp` is the same code in C++17 on Eigen and the
MuJoCo C API: the QP solver, the kinematics, the view grid, the motion layer, the
guard, the perception and the B0 and B4 task layers, with the same constants, the
same order of operations and the same comments about why each number is what it is.
It links no ROS and no simulator, so it is what would go on a real controller.
`sightline_ros_cpp` wraps it in the lockstep contract the Python node already
speaks, so either planner drives the cell node over the same topics.

Three things differ from the Python, and none of them is a design change:

- **The robot model comes from a file.** The Python planner calls
  `station.robot_only_model()`, which builds the model in process. The C++ one takes a
  `robot_model` parameter and loads an MJCF. `station.export_robot_mjcf` writes that
  MJCF from the same builder, with the textures as PNG files beside it and the slashes
  left in the names. The exported model has the same bodies, sites, cameras, mocap
  parts, collision shapes, joint limits and camera field of view as the one built in
  process.
- **The eight sample poses the guard uses come from a different draw.** The guard
  decides which points on the arm can move at all, and which never leave the base's
  axis by more than 20 cm, by sampling eight random joint vectors. The Python draws
  them from numpy's PCG64 with seed 0; the C++ draws them from `mt19937_64` with seed
  0. Both are fixed, so both are the same on every run, and the two tests they feed
  (a point moves more than 1 mm; a point stays within 20 cm of the axis) are coarse
  enough that which eight poses they are does not matter. Checked on the exported
  model: both give 63 points on the arm, 58 of them moving, 8 of those rooted on the
  base's axis, and the same 4.009591 m of radii over the 63.
- **Arithmetic is double throughout.** The view grid's speed measurement casts its
  points to float32 in the Python, for speed on large arrays. The C++ keeps them in
  double. The port is therefore close to the Python but not bit-identical, and it has
  not yet been run against it over a full cycle.

Two pieces of MuJoCo's rendering had to be copied exactly, and both are recorded in
the changelog for 2026-09-20: the offscreen context is made on an EGL device rather
than `EGL_DEFAULT_DISPLAY` and left surfaceless, and `readDepthMap` is set to
`mjDEPTH_ZEROFAR` so the reversed depth coefficients undo what the buffer holds. With
both right, the C++ known world render matches the Python renderer on the same model
and camera to four decimal places.
