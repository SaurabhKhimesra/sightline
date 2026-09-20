# Results log

Every result with the conditions it was measured under. Old numbers stay and are
marked, never deleted.

## 2026-09-18, gate 1: the station

Conditions: `station.build()` with the default layout, worker 1.75 m, robot parked
at the chosen park pose unless stated. Visibility measured by ray casting from the
eyes camera to points on the worker's clothing meshes (every 5th vertex), counting
a point as seen only if it is inside the camera's field of view and nothing blocks
the line. Raw numbers in `results/gate1/gate1.yaml`.

### Heights

| what | value | where it comes from |
|---|---|---|
| worker stature | 1.75 m | scaled from the model's 2.05 m, scene choice inside the NCD-RisC 2016 range |
| elbow height | 1.193 m | measured on the scaled model, standing with flat feet |
| work surface | 1.118 m | elbow minus 7.5 cm (CCOHS band 5 to 10 cm for light work) |
| eyes camera | 3.25 m, 55.7 deg down | scene choice, picked by the comparison below |

### Overhead camera position

Three mounts, same worker pose ("left hand on the cover"), share of the worker's
hand surface the camera can see:

| mount | angle | height | hands seen | note |
|---|---|---|---|---|
| on the cell frame, looking down | 73 deg | 2.14 m | 24 % | sits 40 cm above the worker's head |
| boom off the frame corner | 47 deg | 2.66 m | 29 % | the frame's own post cuts into the image |
| pole beside the bench (**chosen**) | 56 deg | 3.25 m | 35 % | looks over the frame, not through it |

A camera in front of the cell was tried first and dropped: at 45 to 55 deg the
worker's own head hides the box, and the view is of his back.

### What the camera sees, per work pose (robot parked)

Share of each hand's surface, and of the head and arms:

| worker pose | left hand | right hand | head | arms |
|---|---|---|---|---|
| both hands on the cover | 22.4 % | 27.9 % | 20.5 % | 45.9 % |
| left hand on the cover | 29.9 % | 29.9 % | 22.6 % | 47.1 % |
| hand beside the next hole | 16.9 % | 29.9 % | 22.6 % | 46.5 % |
| reaching across the jig | 10.9 % | 29.9 % | 24.0 % | 43.5 % |
| leaning in to look | **6.5 %** | **8.0 %** | 21.2 % | 50.6 % |
| taking a part from the rack | 46.8 % | 22.4 % | 23.3 % | 32.9 % |
| putting a box in the tray | 10.9 % | 10.4 % | 24.0 % | 28.2 % |
| clipping blocks in the prep area | 25.4 % | 27.4 % | 17.8 % | 37.1 % |

- Every hand is visible in every pose. The worst case is the worker leaning over
  his own hands: only 6.5 % of the left hand, and what hides it is his own body
  (180 of 201 sampled points), not the robot and not the station.
- No sampled point of the worker fell outside the camera's field of view in any
  pose, so the framing is not the limit; occlusion is.
- The old top-down mount was measured in the same three hard poses and is not
  better: leaning in to look gives 6.0 % and 4.5 %. A single overhead camera
  cannot see hands that the worker leans over. The planner has to treat that as
  unseen space, which design.md section 8.3 already requires.

**Correction:** the first run reported 0.0 % for the left hand while leaning. That
was a sampling artefact of a coarser ray grid (every 7th vertex). On the finer
grid it is 6.5 %. The coarse number is wrong and is not used.

### What the robot hides while it works

Robot holding a screw over the rear right cover hole, share of the worker's whole
visible surface hidden by the robot:

| worker pose | hidden by the robot |
|---|---|
| hand beside the next hole | 12.6 % |
| both hands on the cover | 4.9 % |
| reaching across the jig | 1.6 % |
| the other five poses | 0.0 % |

So rule R2 is not decorative at this camera position: the robot blocks the view
exactly in the poses where a hand is near the screw it is driving.

**Superseded:** the first run of this table read 11.8 %, 5.4 % and 1.7 %. It was
measured before the screwdriver had collision geometry, so the arm pose picked at
that hole was one where the nose passed through the cover. With the tool modelled
the solver picks a different pose at the same hole, and it hides slightly more.

### Park pose

Ten collision-free park poses were tested against all eight work poses. **All ten
hide 0.00 % of the worker.** The one kept puts the bit 0.30 m above the worktop at
x -0.45 m, y +0.33 m, behind the work, turn angle 0 deg.

### Reach, per hole and turn angle

Turn angle stepped 15 deg, so 24 angles per hole. "Clear" also means the robot and
its tool touch nothing in the station.

| target | reachable | clear of the station |
|---|---|---|
| cover hole 1 (front left) | 24/24 | 24/24 |
| cover hole 2 (rear left) | 20/24 | 20/24 |
| cover hole 3 (front right) | 24/24 | 24/24 |
| cover hole 4 (rear right) | 20/24 | 20/24 |
| rail hole 1 (cover off) | 23/24 | **17/24** |
| rail hole 2 (cover off) | 23/24 | **17/24** |
| screw feeder pick point | 24/24 | 24/24 |

The four angles the rear holes lose are the ones that put the tool's mount toward
the robot base. That is the side mount of the real screwdriver showing up in the
numbers, and it is why the look-around layer has to check reach per turn angle.

**The rail screws are the tight ones.** They sit inside the open enclosure, and at
6 of 24 turn angles the screwdriver's nose, 49 mm across, touches the enclosure
wall on the way in. So the look-around layer does not only choose a turn angle for
the camera's sake: for these two screws most angles simply do not fit.

**Superseded:** this table first read 23/24 clear for both rail holes. That was
measured with no collision geometry on the screwdriver at all, so nothing could
touch the box. The first attempt to fix it gave 0/24, because the enclosure's own
collision shape was a solid block and every pose that reached inside it counted as
a crash. Both are wrong. With the tool modelled and the enclosure modelled as a
floor with four walls, it is 17/24.

### The camera moved, 2026-09-18

Chosen by `sightline_sim/gates/eyes_optimise.py` from 76 pole positions, for two things at once:
how much of the space the arm and a person can share above the worktop the camera
sees past the furniture, and how much of the space behind the robot it keeps in the
picture in case a second person walks up.

| pole | pitch | shared space seen | behind the robot seen |
|---|---|---|---|
| beside the bench, 3.25 m (the gate 1 choice) | 55.8 deg | 84.5 % | 79.4 % |
| **further out, 3.60 m (chosen)** | 47.6 deg | **86.6 %** | **82.0 %** |

No position gets the shared space above about 90 %: the bench top shadows the strip
along its own front edge from anywhere a pole can stand. What the camera cannot see
is handled by the planner (gate 5).

Coverage at the eight work poses with the moved camera, same method as above
(every 5th vertex, in shot, robot parked):

| worker pose | left hand | right hand | head | arms |
|---|---|---|---|---|
| both hands on the cover | 18.9 % | 21.4 % | 19.2 % | 45.9 % |
| left hand on the cover | 25.9 % | 23.4 % | 23.3 % | 43.5 % |
| hand beside the next hole | 32.8 % | 23.4 % | 23.3 % | 41.2 % |
| reaching across the jig | 32.3 % | 22.9 % | 24.7 % | 35.3 % |
| leaning in to look | 7.0 % | **5.0 %** | 20.5 % | 42.9 % |
| taking a part from the rack | 42.8 % | 18.9 % | 20.5 % | 32.9 % |
| putting a box in the tray | 11.4 % | 11.4 % | 24.0 % | 32.4 % |
| clipping blocks in the prep area | 25.9 % | 19.4 % | 16.4 % | 35.3 % |

Against the first table: the hands beside the screw the robot works on are seen
about twice as well (16.9 to 32.8 %, 10.9 to 32.3 %), the mean over all sixteen
hand numbers slips from 22.2 % to 21.4 %, and the worst single hand from 6.5 % to
5.0 %, because the shallower angle lets his own body hide a little more when he
leans. The robot at the rear right hole now hides 15.2 % of him in the "hand beside
the next hole" pose (was 12.6 %) and 7.7 % reaching across (was 1.6 %): R2 has more
work with this camera, not less. The park pose still hides 0.00 % in every pose.
Reach is unchanged. The tables above this note are superseded by these.

### Gate 1 verdict against its own stop rules

- Reach: every hole and the feeder reachable from at least 20 of 24 turn angles,
  and collision free from at least 17 of 24. **Pass**, with the rail screws noted
  above as the tight ones.
- Camera coverage: every hand visible in every work pose, nothing out of frame.
  **Pass**, with the worst case recorded above.
- Park pose exists that hides nothing. **Pass.**
- R2 has work to do (the robot hides up to 11.8 % while screwing next to a hand),
  so the rule is not decorative. **Pass.**

Still open in gate 1: the enclosure, DIN rail, terminal blocks and screw feeder are
scene choices, not products.

## 2026-09-18, gate 2: the worker's cycle

Conditions: `station.build()` with the default layout, the robot parked at the
gate 1 park pose and not moving, the collision model as corrected on 2026-09-18
(hollow enclosure, screwdriver nose and bit, thumb counted as part of the hand), the worker played back from the script (joint
positions written every 2 ms, no dynamics, so he can neither fall nor be pushed).
One cycle, 19 segments, 44.4 s. Raw numbers in `results/gate2/gate2.yaml`, video
in `results/gate2/cycle_hero.mp4` and `cycle_eyes_cam.mp4`.

### The cycle

Empty jig to a finished box in the tray: fetch the base from the rack, set it in
the jig, fetch a rail, fit and press it, stand clear while the robot drives the
two rail screws, clip blocks on the next rail, fetch a cover, press it on, hold it
while the robot drives the four cover screws, lift the box out and put it in the
tray. Three stances: at the bench, and two at the rack, because the rack is out of
arm's reach from the bench. The two screwing steps are waits of about the right
length; the robot does not move in gate 2.

### Numbers gate 2 asks for

| what | value |
|---|---|
| cycle time | 44.4 s |
| peak hand speed | 1.31 m/s left, 1.20 m/s right |
| peak hand acceleration | 4.4 m/s² left, 3.9 m/s² right |
| worst hand penetration | 36.5 mm into the bench, while carrying the base back |
| penetration, everything else | jig 26.9, cover 24.5, rail 20.6, base 20.5 mm |
| foot travel | 818 mm, which is the two scripted side steps to the rack |
| worst reach error | 0.0 mm, all 19 segments |

The hand speeds are a property of the script, not a measurement of anyone: the
segment durations are scene choices and the minimum jerk profile sets the peaks.
They are worth recording because the planner has to cope with them, and because
the speed and separation formula uses a person speed v_H, for which ISO 13855
gives 2000 mm/s and 1600 mm/s beyond 500 mm and allows v_H to be measured instead
(Marvel and Norcross 2017, Sec. 3, pp. 146 and 148). We take the measured route,
decided in design.md section 17.2, and these are the speeds it will measure.

**The penetrations are a playback artefact and are not a safety result.** The
worker is written into place frame by frame with no contact response, so a hand
can pass through the bench or a part. They matter only because they would look
wrong on camera. Adding a lift-off pose at the start of each transit, the way a
real hand comes off a part before moving, cut the worst from 57.8 mm (into the
box) to 36.5 mm (into the bench top, while his body hides the moment from the
camera anyway). What is left is the arm swinging a wide arc as it comes back from
the rack: the key poses blend joint by joint, so the hand does not travel in a
straight line. Nothing in the safety layer depends on it.

### What the overhead camera sees during the cycle

Share of the worker's hand surface the eyes camera can see, sampled twice a
second, every 9th vertex:

| moment | hands seen |
|---|---|
| mean over the cycle | 23.5 % |
| best (holding the cover while the robot screws) | 31.7 % |
| at the bench, worst moment (putting the box in the tray) | 15.0 % |
| picking from the tote at the rack | 3 % |
| **carrying the part back from the rack** | **0.0 %** |

**There is a blind moment, and it is the bench, not the rack.** At the worst
moment none of 662 sampled hand points is visible. Checked at three ray grids
(every 9th, 5th and 3rd vertex) and it is 0.0 % in all three, so unlike the gate 1
case this is not a sampling artefact. The tote pick reads 1.1 % on the coarse grid
and 2.5 to 3.3 % on the finer ones; call it 3 %.

**Correction, 2026-09-18.** The first version of this paragraph said the rack's
shelves hide the hands. Naming the blocker of each ray says otherwise: the bench top
hides 492 of the 662 points, the worker's own head and neck 135, the jig 25, the
rack none. His hands are low and on the far side of the bench from the camera while
he carries the part, so the bench is in the way. Moving the pick points to the front
of the totes changes nothing (still 3 %), which is how the wrong attribution was
caught. What follows from the right one is in gate 5: the camera moved a little, the
arm stays above the worktop, and the bench's shadow on the worker's side counts as
occupied while he is in the cell.

### Gate 2 verdict against its own stop rules

- Every hand target reachable: 0.0 mm error on all 19 segments. **Pass.**
- Hand speeds and accelerations in a human range. **Pass.**
- Penetration: worst 36.5 mm, playback artefact, cause understood. **Pass with
  the note above.**
- Foot travel is the scripted side steps only, no sliding at a stance. **Pass.**

New in gate 2, carried forward: the rack blind spot above, and the fact that the
worker's own body is the main occluder at the bench (the robot hides nothing while
parked).

## 2026-09-18, gate 3: B0, the baseline that ignores the worker

**What would have killed this gate:** no contacts and no blocked frames. That would
mean there is nothing to avoid and the whole project needs rethinking. It did not
happen, by a wide margin.

Conditions: `station.build()` with the default layout. The worker plays his cycle
with per-seed variation (segment durations ±12 %, stance ±15 mm, lean ±0.02 rad).
The robot runs B0: taught positions, no awareness of the person at all. Both are
played back kinematically, so the robot tracks its own commands exactly and passes
through whatever is in the way; there is no contact force. The episode is one worker
cycle. The planner sees only what design.md section 8.1 allows, and the judge measures
R1 and R2 from the simulator's state (design.md section 4). Raw numbers in
`results/gate3/b0_seed{0,1,2}.yaml`, video for seed 0.

### Three development seeds

| seed | cycle | screws driven | robot idle | contacts | robot moved in | person moved in | deepest | blocked frames | blocking events | longest block | worst pixels |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 44.1 s | 2 of 6 | 19.1 s | 100 | 46 | 54 | 137 mm | 509 of 1299 (39 %) | 6 | 6.5 s | 1272 |
| 1 | 45.3 s | 3 of 6 | 19.3 s | 83 | 32 | 51 | 133 mm | 546 of 1332 (41 %) | 4 | 6.3 s | 1144 |
| 2 | 44.6 s | 3 of 6 | 18.9 s | 94 | 34 | 60 | 125 mm | 546 of 1312 (42 %) | 6 | 6.3 s | 1128 |

An episode costs about 9 minutes of wall time, most of it the two segmentation
renders per camera frame that the R2 judge needs.

### R1: it hits him, and it hits his head

Every seed ends with 83 to 100 separate contacts. A third to a half of them are
labelled "the robot moved in", which is the label that matters: those are the ones
no rule about human behaviour can excuse. What it touches, over the logged contacts
of all three seeds:

| part | contacts |
|---|---|
| head and neck | 37 |
| right upper arm, shoulder, forearm | 39 |
| right hand, fingers, thumb, wrist | 26 |
| chest | 10 |
| left shoulder and upper arm | 8 |

The first contact of seed 0 is the clearest picture of the problem: at 21.1 s the arm
is carrying a screw from the feeder to the first rail hole, and it goes through the
worker's **head**, 137 mm deep, while he leans over the box. He was not moving fast
(0.11 m/s). The robot simply drove its taught path through the space he was in.

### R2: the view is blocked about 40 % of the time

Roughly 4 frames in 10 have at least one pixel of the worker hidden by the robot,
and the worst single frame hides 1272 pixels of him. The blockages come in a few
long events rather than flickers: the longest is 6.5 s, which is the robot standing
over the box driving a screw while the worker works beside it.

**A measurement problem, open.** design.md section 4 says to label a
violation by putting the robot back one motion cycle, 10 ms, and looking again. For
contacts that works. For blocked frames it does not: 10 ms is 2.5 mm of arm travel,
and by the time a 30 Hz camera catches an overlap it is already wider than that, so
the answer is "the person moved in" almost every time (1 of 16 events across the
three seeds came out as "robot"). Asking the same question one camera frame back,
33 ms, gives 8 of 16 events as "the robot moved in", which matches what the video
shows. Both numbers are recorded. The rule text needs one line changed, and that is
a design call, not a measurement.

### What B0 got done

Two or three of the six screws per cycle. At about 11 s per screw and 44 s of worker
cycle, one arm with a one-at-a-time feeder cannot keep pace with this worker, and it
spends 19 s of every cycle waiting for the jig signal. That is a cell design finding,
not a planner one, and it is worth saying to a manufacturer up front: the robot is
not the bottleneck because it is careful, it is the bottleneck because there is one
of it.

### The wrist camera

The hole detector in B0 is a centroid of dark pixels in a window around the predicted
hole. It found a hole in 40 of 120 sampled approach frames, and where it found one it
was 7 to 16 mm away from the true hole, which is worse than the taught position it was
supposed to correct. So B0 refuses any correction over 4 mm and in this gate drove
every screw from its taught position. Gate 4 is the perception gate and builds this
properly; the numbers above are what it has to beat.

### Gate 3 verdict

- Contacts on every seed, a third to a half of them the robot's doing. **The kill
  test does not fire.**
- Blocked frames on every seed, about 40 %, in long events. **R2 has real work.**
- B0 is a fair baseline: it uses the same tool, the same reach and the same taught
  points the later variants will use, and it fails only because it never looks.

## 2026-09-18, gate 4: perception

**What would have killed this gate:** person recall so low that the planner would be
planning around someone it cannot see, or a box and hole error too large to put a
screw in. Neither happened.

Conditions: the same episode as gate 3, seed 0, B0 driving and the worker working.
The eyes camera is a RealSense D455 stand-in at 640 x 480, 30 Hz, one frame late,
with the depth noise of Nguyen, Izadi and Lovell 2012 (their Eqs. 1 and 3) held flat
past the 2.75 m they fitted it over, no returns past 4 m, and no return across a
depth step. The cell's own calibration of the camera is 2 mm and 0.15 degrees out
(scene choice). The planner sees only those images plus its own joint angles, its own
robot model and the static cell numbers. Ground truth is a perfect depth image
labelled by segmentation, so the yardstick is what the camera could have seen. Raw
numbers in `results/gate4/gate4_seed0.yaml` and `hole_views.yaml`, overlay video in
`perception_seed0.mp4`.

### Finding the person

| what | value |
|---|---|
| recall, mean over 1298 frames | 93.0 % |
| recall, worst frame | 83.2 % |
| precision, mean | 97.6 % |
| person voxels per frame (2 cm) | 1507 |
| unseen voxels behind him per frame | 11286 |
| compute per frame | 50.6 ms mean, 115 ms worst |
| delay | 1 camera frame, 33 ms |

A voxel counts as found if it is within one 2 cm cell of a true one, because a grid
built from noisy depth does not land on the same cells as a grid built from a perfect
one. The delay is measured by shifting the truth back frame by frame and seeing which
shift fits best, over the 450 frames where the worker was actually moving: standing
still, every shift ties. One frame is exactly the delay the camera model was given,
so nothing else in the pipeline adds lag.

The 2.4 % of voxels that are not the person are the parts he is carrying and the
totes he has disturbed. That is the conservative way round and it is deliberate:
anything that is not the station, not the robot and not the box in the jig counts as
person, so a part left on the bench makes the robot cautious rather than confident.

**Unseen space is seven times the person.** Behind every person voxel, up to 30 cm of
space the camera cannot see is treated as occupied, which is the depth space idea of
Flacco et al. 2012. That is 11286 voxels against 1507. It is the consequence of
one camera, and gate 5 will show what it costs in room to move.

### The box

| what | value |
|---|---|
| frames with an accepted fit | 15.6 % |
| centre error, median | 6.5 mm |
| centre error, worst | 25.3 mm |
| top face height error, median | 12.1 mm |
| yaw | not observable, see below |

The fit is refused unless the flat face on show is the size the part drawing says,
200 by 150 mm. That is why it only fires on one frame in six: the worker's hands are
on the cover most of the cycle. When it is refused the planner keeps the last good
fit, which is what a cell with a bolted jig does anyway.

**Yaw cannot be measured from this camera.** From 3.25 m the cover is about 34 by 26
pixels. A free search over 90 degrees lands on whatever angle fits the noise, up to
45 degrees out. The search is therefore limited to the jig's own tolerance, plus or
minus 10 degrees, and it still runs to that limit, so the number it returns carries no
information. The jig sets the yaw. What this fit contributes is the centre and the
height.

### The holes, and the reason B0 could not see them

Two different numbers, and the difference between them is the point of the gate.

**What the wrist camera can do.** With the tool at its approach pose, 60 mm above the
hole, for every turn angle around the bit:

| screw | reachable angles | hole in view | hole detected |
|---|---|---|---|
| rail 1 | 23 of 24 | 23 | 18 |
| rail 2 | 23 of 24 | 21 | 17 |
| cover 1 | 24 of 24 | 23 | 23 |
| cover 2 | 20 of 24 | 20 | 20 |
| cover 3 | 24 of 24 | 21 | 21 |
| cover 4 | 20 of 24 | 20 | 20 |

So every screw has between 17 and 23 turn angles from which the camera sees the hole
and the detector finds it. Where the view is lost it is the robot's own upper arm or
forearm in the way, never the enclosure.

**What B0 actually got.** B0 picks its turn angle for a smooth insertion and nothing
else, and it landed on an angle that shows the hole to the camera for **2 of its 6
screws**. Over the episode the detector fired on 3 of 170 approach frames. Where it
did fire it was good: 1.69 mm median error, 3.76 mm worst, against 7 to 16 mm for the
gate 3 detector.

That is the clearest argument yet for the look-around layer. The sixth axis is free,
20 or more angles per screw are good, and a planner that does not look throws that
away. design.md section 8.4 already lists "a clear line from the wrist camera to the
hole" as one of the checks; gate 6 is where it gets used.

### Predicting a blocked view before it happens

The planner asks whether anything it can see sits within 3 cm of the line from the
wrist camera to the hole. Over 170 approach frames:

| what the planner counts as in the way | true positive | false positive | true negative | accuracy |
|---|---|---|---|---|
| person voxels only | 0 | 0 | 170 | 100 % |
| person voxels and unseen space | 0 | 73 | 97 | 57 % |

In this episode the worker never actually got between the wrist camera and a hole, so
there is nothing to catch. What the table does show is the cost of being conservative:
treating unseen space as occupied on the wrist sight line raises a false alarm on 43 %
of frames. Unseen space is the right call for R1, where a hand could be hiding there
and the price of being wrong is a contact. For "can I see this hole", the price of
being wrong is a pause, and the person voxels alone are the better test. The two uses
should not share one rule, and gate 6 will keep them apart.

### Gate 4 verdict

- Person found on every frame, 93 % of what the camera could see, 97.6 % of what is
  reported is really him, one frame of delay and 51 ms of compute. **Pass.**
- Box centre to 6.5 mm and height to 12 mm when the face is on show, and a
  refusal when it is not. Yaw is not observable and is not used. **Pass, with the
  yaw recorded as a limit of this camera.**
- Holes detected to 1.7 mm from 17 to 23 of 24 turn angles per screw. **Pass.**
- Blocking prediction has nothing to catch in this episode. **Not proven.** It needs
  an episode where the worker reaches across a hole the robot is working on, which
  is a stress script for gate 7.


## 2026-09-19, stage 1: the planner live on ROS 2, the same run to the millimetre

The cell (MuJoCo, the worker, the eyes camera, the jig, the judge) and the planner
(B4) ran as two ROS 2 nodes in lockstep on sim time (`sightline_ros/cell_node.py`,
`sightline_ros/planner_node.py`, notes.md "The planner live on ROS 2"). Seed 0, the
balanced cycle, 80.9 s.

- The judge's result is gate 5's: 6 of 6 screws, 0 contacts, 0 blocked frames,
  closest approach 96.6 mm, held by the rules 3.43 s, no asks, and the guard's
  verdict counts are the same to the tick (brake 536, escape to park 327, slow 114,
  escape up 3, to standoff 7, back 4, escape brake 1, go the rest of 8090).
- Parity, checked directly over 6 s against the in-process runner
  (`results/ros2/stage1/parity`): robot and worker poses equal to 0.0000 mm, the
  planner's state and the guard's verdict equal in every frame. Over the whole run,
  the recorded poses of all 51 bodies equal gate 5's recording
  (`results/ros2/run_B4_seed0_poses.npz`) to 0.0 m in all 2023 frames.
- Bag: `results/ros2/stage1/bag_seed0` (mcap, zstd, 38 MB): 53,804 messages on 20
  topics, everything but the pictures. Replayed into rviz at 5.81x (the run's wall
  time spacing) with the cell's sim time on `/clock`: `B4_rviz_seed0.mp4`. The Gazebo
  render of the stage 1 poses (`B4_hero_gazebo_seed0.mp4`, `B4_eyes_gazebo_seed0.mp4`
  in `results/ros2/stage1/`) is frame for frame the gate 5 one, as the poses are.
- Cost: 471.6 s wall for 80.9 s of sim (real time factor 0.172); 8090 ticks,
  0 timeouts, the cell waited 2.77 ms median / 121.9 ms p99
  for a command; the planner's picture 160.6 ms median / 496.9 max
  (perception 137.0, grid 15.2, blind cells 7.5,
  rviz output 6.7), the guard 1.2 ms median / 7.9 p99, the QP 0.38 ms.
- Files: `results/ros2/stage1/cell_B4_seed0.yaml`, `planner_B4_seed0.yaml`,
  `run_B4_seed0_poses.npz` (the Gazebo replay renders it).
- the lockstep makes the run independent of the machine's speed; a
  real cell has no lockstep, and the 137 ms picture must come down to the 40 ms
  camera period or run one picture behind before this planner drives hardware.

## 2026-09-19, the judged run rendered by Gazebo Sim

The next step was the work run and recorded on ROS 2 tools. The B4 run on the
balanced cycle (seed 0, 80.9 s) was scored again by the judge in MuJoCo with every
body pose recorded at 25 Hz, then played into Gazebo Sim frame for frame
(README "Gazebo and ROS 2", notes.md "Gazebo as a second renderer").

- The re-run reproduced the gate 5 result: 6 of 6 screws, 0 contacts, 0 blocked
  frames, closest approach 96.6 mm, no asks. `results/ros2/run_B4_seed0_poses.npz`.
- Replay: 2023 frames, all 40 posed bodies per frame, both cameras rendered by Gazebo
  on the GPU at 1280x720 and 960x720, 0 frames missed, 531 s (3.8 frames a second).
- Videos: `results/ros2/gazebo/B4_hero_gazebo_seed0.mp4` and
  `B4_eyes_gazebo_seed0.mp4`, captioned with the judge's record and a line saying the
  picture is Gazebo's replay of the run judged in MuJoCo. Checked by eye at 26.5 s
  (rail screw 2 coming out while he clips blocks at the prep area), 50.2 s (three
  screws done, he fits jumper bars) and 55.2 s.
- Front view, a request (2026-09-19): `B4_front_gazebo_seed0.mp4`, a camera on the
  far side of the bench behind the robot, so his face and hands, the arm and the box
  are in one frame. All three Gazebo videos were rendered again with the two key
  lights casting shadows (the first renders had none): 2023 frames from three cameras
  in 725 s, none missed.
- Desktop recording, after I saw the Gazebo window lag: measured 7.6 fps for
  the Gazebo window against 24.8 for rviz (the GUI rendered on the Intel GPU that
  drives the display; the NVIDIA one only had the headless server). Sending the GUI
  to the NVIDIA GPU made it render fast but read back through X at 1 to 13 fps, so
  `B4_windows_front_seed0.mp4` now has Gazebo's front camera rendered live on the GPU
  during the run (2073 frames, none missing or repeated, 25 fps in every frame of the
  result) beside rviz grabbed at 25 fps, the two aligned on the driver's first frame.
  The earlier window grabs are superseded; notes.md "The desktop recording" has the
  measurements and the three paths that did not work.
- What the picture is not: the eyes video carries no grid tint (the grid is the
  planner's and is not recorded).
- Desktop recording, `results/ros2/gazebo/B4_windows_seed0.mp4` (2560x760, 92 s):
  Gazebo's GUI on the hero camera's view and rviz2 side by side, the run played in
  real time (2023 frames in 80.9 s wall, 1 of them more than a frame late). rviz shows the UR5e model on the
  recorded joint angles, the worker as the judge's 47 capsules and spheres, 351
  station shapes, the eyes camera's picture bridged from Gazebo and the judge's
  record as a picture strip. Checked by eye at 31 s and at the end.


**Supersedes every gate 5 table below for the comparison.** The design decisions after
the first clamp-cycle results: change the script, not the task, and at least two
screws must go in while he is visibly working beside the robot, really driven under
the rules (design.md section 19). His tasks were re-ordered the way a line is balanced: the
rail screws while he clips blocks on the next rail 40 cm from the jig and then fetches
the cover; the cover screws while he finishes that rail and then watches from 30 cm
back. That order was chosen for this and says so here. The jig gives four signals
(rail pressed, cover placed, clamps closed, clamps opened), the rack stands 30 cm
further toward him, and three faults in his script were fixed on the way (CHANGELOG).
Seed 0, one cycle of 80.9 s, 25 Hz, the same limits for every variant. Raw numbers in
`results/gate5/B{0,2,3,4}_seed0.yaml`, the B4 video in `results/gate5/B4_*_seed0.mp4`.

| | B0, rules off | B2, R1 dampers | B3, R1 and R2 dampers | B4, the camera grid |
|---|---|---|---|---|
| screws driven | 6 of 6 | 4 of 6 | 4 of 6 | **6 of 6** |
| contacts, **robot** moved in | 5 | 5 | 8 | **0** |
| contacts, he moved in | 35 | 1 | 29 | **0** |
| deepest contact | 76 mm | 60 mm | 76 mm | none, closest 97 mm |
| camera frames with him blocked | 10.1 % | 9.8 % | 14.4 % | **0.0 %** |
| blocking events the robot caused (one frame back) | 1 | 2 | 0 | **0** |
| asked him to move | 0 | 3 | 3 | 0 |
| compute per 10 ms cycle, median / p99 / worst | 0.15 / 0.8 ms | 7.4 / 18 ms | 9.3 / 31 ms | 1.0 / 6.4 / 12.9 ms |

**What B4 did, screw by screw, with what he was doing at that moment** (the runner
records both): rail 2 at 26.5 s while he clipped the rest of the blocks at the prep
area; rail 1 at 36.0 s while he turned at the rack with the cover; cover 4 at 50.2 s
while he fitted the jumper bars; cover 3 at 55.2 s while he dropped his hands at the
prep area; cover 2 at 61.2 s and cover 1 at 67.1 s while he watched from 30 cm back.
Three screws with him working 40 cm from the jig, in the same frame. The robot was
done at 67 s; he opened the clamps at 73 s. It held for 3.4 s in the whole cycle,
stepped back twice for 1.2 s, and never asked him to move.

**The bound for any planner** on this cycle, true worker every 0.2 s, 10 cm clear
and off the camera's lines of sight to him: a rail screw in 80 % of its 20.4 s window
(longest stretch 8.4 s, all of it while he clips blocks), a cover screw in 96 % of
its 29.8 s window. B4 used it.

**Caveats:**
- One seed. The held-out seeds and the constant 2000 mm/s speed setting are gate 7.
- The 97 mm closest approach is at 43.7 s, while he fitted the end stops at the prep
  area: the planner backed the arm to park because of a one-frame phantom cell near
  the robot's base, and the arm passed him on the way. Not a contact, and not a real
  hand. The phantom's source is still open.
- The measured person speed reads 0.19 m/s above the truth at the median and is below
  the hand's true peak in 11 % of pictures, by up to 0.68 m/s.
- The per pixel background gate that removed the jig-corner phantoms costs almost no
  recall. Gate 4's check rerun on this cycle: recall 87.6 % mean and 69.1 % worst,
  precision 99.7 % with the new gate; 88.1 %, 69.4 % and 99.0 % with gate 4's fixed
  40 mm gate on the same cycle. The drop from gate 4's 93 % is the cycle, not the
  gate: he now spends more of it at the rack, further from the camera, and a step
  back from the bench. Raw numbers in `results/gate4_recheck/`.
- The wait before he opens the clamps is a scene choice, 11 s, standing in for the
  stack light a real cell would use.
- B3 is worse than B2 on this cycle: the R2 dampers make the arm dodge the camera's
  sight lines into him (8 contacts the robot caused, 29 he did). Dampers on lines
  cannot both stay off the lines and away from him at once; the grid does, because it
  treats the line of sight and the space behind him as one off-limits volume.

## 2026-09-18, gate 5 on the clamp cycle: B4 is the only variant that never causes a violation

**Supersedes the next two sections for the comparison.** The jig's clamps now hold the
cover and he works elsewhere while the robot screws (a design decision); his script
was corrected with it (design.md section 19, CHANGELOG). Seed 0, one cycle of 69.8 s, 25 Hz,
the same limits for every variant. Raw numbers in `results/gate5/B{0,2,3,4}_seed0.yaml`.

| | B0, rules off | B2, R1 dampers | B3, R1 and R2 dampers | B4, the camera grid |
|---|---|---|---|---|
| screws driven | 6 of 6 | 1 of 6 | 0 of 6, 2 missed | 1 of 6, 2 missed |
| contacts, **robot** moved in | 12 | 5 | 0 | **0** |
| contacts, he moved in | 37 | 41 | 13 | 3 |
| deepest contact | 126 mm | 136 mm | 80 mm | 23 mm |
| camera frames with him blocked | 20.5 % | 7.4 % | 6.1 % | **0.6 %** |
| blocking events the robot caused (one frame back) | 1 | 1 | 3 | **0** |
| compute per 10 ms cycle, median / worst | 0.16 / 0.8 ms | 8.0 / 17 ms | 11.8 / 22 ms | 1.2 / 13.7 ms (p99 8.9) |

**What would kill gate 5**, stated before: the rule variants hitting him when the
robot moved in. B2 did, 5 times; B3 did not touch him but blocked the camera's view of
him 3 times; B4 did neither. Its 3 contacts and its one blocking event are all his
doing, one moment at 27 s: fetching the cover from the rack, he turns 142 degrees and
his right hand swings behind the jig into the space round the robot's base, 23 mm into
the root of the upper arm, the one part of the arm no motion can take away (the arm
lifted clear of it in the meantime).

**Throughput is the open problem.** The bound for any planner with perfect knowledge of
him, on this cycle: a rail screw can be driven 10 cm clear of him and off the camera's
view of him in 21 % of its 19.2 s window, longest stretch 2.2 s; a cover screw in 69 %
of its 33.6 s window, longest 12.0 s. B4 drove one cover screw. Its margins are what
cost it: 10 cm, plus the size of the part of the arm, plus how far each part of him
could move while the arm stops, at his measured speed, which reads 0.2 m/s above the
truth at the median. Together that asks for 15 to 25 cm of true clearance.

## 2026-09-18, gate 5 at 25 Hz: the camera grid keeps both rules, and the layout stops the work

**Supersedes the numbers in the next section.** Every arm link in the motion layer,
B0's clear check and B4 had been a ball at its centre (CHANGELOG, gate 5), and the
cameras now run at 25 Hz, a design choice. Seed 0, the gate 3 episode, one worker
cycle of 44.1 s. Raw numbers in `results/gate5/B{0,2,3,4}_seed0.yaml`.

| | B0, rules off | B2, R1 dampers | B3, R1 and R2 dampers | B4, the camera grid |
|---|---|---|---|---|
| screws driven | 4 of 6 | 0 of 6, 2 missed | 0 of 6, 2 missed | 0 of 6, 2 missed |
| contacts, robot moved in | 26 | 0 | 0 | 0 |
| contacts, he moved in | 50 | 0 | 0 | 0 |
| deepest contact | 137 mm | none | none | none |
| closest robot to him (judge) | 136 mm inside him | 9.7 mm | 14.0 mm | 81.1 mm (49.4 on the corrected worker) |
| camera frames with him blocked | 27.0 % | 0.0 % | 0.0 % | 0.0 % |
| compute per 10 ms cycle, median / worst | 0.15 / 1.0 ms | 8.5 / 33 ms | 11.0 / 38 ms | 1.7 / 14.9 ms (p99 10.3) |

B2 and B3 ran before the blind-area rule changed (below) and will be run again.

**B4** is the grid algorithm: the camera's view as a grid of directions, every
direction in which it sees him off limits from the camera out to 15 cm behind him,
with a margin of 10 cm plus the size of that part of the arm plus how far that part
of him could have moved since the picture. The grid is rebuilt with every picture,
every 40 ms, and every 10 ms the arm's next moves, 63 points on it, are checked
against it (notes.md, the camera grid and the guard). It keeps both rules with
more room than the dampers did: never closer than 81 mm, never in the way of the
camera, and its checks cost a fifth of B3's.

**It drove no screws, and the reasons are in the cell, not the code:**

- The rail screws have one window, from 16.1 s, when the rail is in, to 30.5 s, when
  the cover goes on. Through all of it he is either working at the jig or reaching
  past it. At 25.5 s, for example, his right arm is beyond the robot's left side, and
  the camera's lines of sight to it cross the jig 1.5 m up, exactly where the arm has
  to be to drive a rail screw: all 46 ways of driving the two rail screws would hide
  him from the camera. The second rail screw came closest, 3 points of the arm 25 to
  50 mm inside the margin.
- The robot's base is 83 cm from where he stands and 24 cm behind the jig's back
  edge. With the screwdriver mounted on the side, the wrist sits 15 to 19 cm past
  each hole toward him at half the turn angles; at rail 1, turn angles 0 to 75 and
  270 to 330 degrees keep it on the robot's side, and B4 now keeps all of them.
- Each part he carries counts as him, and he places them 20 cm from the root of the
  upper arm, which no motion of the arm can take away.

**Measured on the way**, one cycle with the arm parked, against his true hands
(design.md 17.2 required this before the measured speed may lead): the grid's speed at
his hands reads 0.19 m/s above the truth at the median and 1.24 at the 95th
percentile; still hands read 0.26 m/s. It is below the hand's true peak in 10 % of
pictures, by up to 0.40 m/s, the lag of any speed read from past pictures while a
hand speeds up; at the oldest picture that uses 6.8 cm of the 10 cm safety distance.
The first estimator read still hands at 0.92 m/s and sent the parked arm fleeing.

**Correction to the worker, found while doing this.** On his second and third trips
to the rack his "hanging" right hand stood 1.42 m up behind him, in the robot's side
of the cell: the script reset only the shoulder and left the elbow bent from the
last reach. It now hangs at 0.84 m, as on the first trip; all 19 reaches still solve
to 0.0 mm. Gates 2 to 4 were measured with the old motion; the difference is those two
trips. B4 on the corrected cycle: still 0 of 6, 0 contacts, 0 frames blocked, closest
49.4 mm.

**The bound for any planner.** With the worker's true position every 0.2 s, at which
moments could some screw of the current stage be driven, every part of the arm at
least a given distance from him and off the camera's lines of sight to him? The
approach and the screwing pose of every one of the 20 to 24 ways per screw:

| stage | window | 10 cm clear | 7 cm | 5 cm | only not touching |
|---|---|---|---|---|---|
| rail screws | 14.4 s | 17 %, longest 2.4 s | 17 %, 2.4 s | 17 %, 2.4 s | 26 %, 2.6 s |
| cover screws | 13.6 s | **0 %** | **0 %** | **0 %** | 31 %, 3.4 s |

A screw needs about 2.5 s at the hole. So no planner can drive the cover screws in
this cell as scripted: he holds the cover and then rests his hand beside the next
hole, and every way of reaching a cover hole brings some part of the arm within 5 cm
of him or into the camera's view of him. The rail screws have one stretch long
enough for about one screw. B4's 0 of 6 is close to the bound, not far from it.

**What would change the result** is the cell or the task, and that is my
decision: whether the jig holds the cover so that he works elsewhere while the robot
screws, where the robot and the feeder stand against the worker and the camera, and
whether the screwdriver mounts in line with the flange.

## 2026-09-18, gate 5, first numbers: the rules work, a fixed order does not

**What would kill this gate:** the rule variants still hitting him when the robot
moved in, or a cycle so slow the cell is useless. The first did not happen. The
second did, for B2 and B3 as design.md defines them, and that is the finding.

Conditions: seed 0, the gate 3 episode. All three variants are the same controller:
the same taught poses, the same synchronised joint profile, the same motion layer
with the UR5e's joint limits at 90 deg/s and 400 deg/s^2. B0 has the rules off, B2
adds R1, B3 adds R2. The planner sees only the eyes camera's noisy, one frame late
depth, its own joints and its own model. The camera is the moved pole (gate 1 note).
Raw numbers in `results/gate5/B{0,2,3}_seed0.yaml`, video for each.

| | B0, rules off | B2, R1 | B3, R1 and R2 |
|---|---|---|---|
| screws driven in the cycle | 4 of 6 | 0 of 6, 2 missed | 0 of 6, 2 missed |
| contacts where the robot moved in | 26 | **0** | **0** |
| contacts where he walked into it | 50 | 0 | 2, 7 mm deep |
| deepest contact | 137 mm | none | 7 mm |
| camera frames with him blocked | 27.2 % | 20.2 % | **5.6 %** |
| stepped back to wait | never | once | once |

The rules do what they are for. Contacts the robot caused go from 26 to none, and
with R2 the camera loses sight of him a fifth as often.

Then they stop the work. What happens in B2 and B3 is the same:

- The rail screws are ready at 16.1 s. The arm fetches a screw and reaches the first
  rail hole at about 20 s, when he is working at the prep area 35 cm away, and is
  held back. It steps back, comes in again at 25.6 s, and at 26.7 s he brings the
  cover. The cover goes on at 30.5 s and both rail screws are recorded as missed.
- It moves on to the cover screws in their fixed order. The first is the front left
  hole, **3.2 cm from where his left hand holds the cover**. With a 10 cm safety
  distance it cannot go there while he holds it, so it waits beside his hand until the
  cycle ends. The three holes 12 to 19 cm from his hand, which it could have driven,
  are never tried.

So the limit is not safety and not speed: it is that B2 and B3 do the screws in the
order they were written down. That is exactly what the look-around layer, B4, is for:
drive the screw that is clear, choose a turn angle the wrist camera can see through,
and wait at a stand-off only when nothing is clear. Gate 6.

Also measured: the per cycle QP takes 7 to 9 ms median and up to
29 ms worst in B3, against a 10 ms cycle. The worst case needs trimming before the
gate 7 matrix.

**Superseded:** gate 3's B0 took about 11 s per screw and concluded one arm could not
keep pace with this worker. That B0 moved each joint on its own exponential approach.
With a synchronised profile and taught poses it does six screws in 33.7 s alone, and
4 of 6 inside the worker's cycle. The gate 3 contact and blocking numbers were measured
with the old B0 and stand as a record of that controller.
