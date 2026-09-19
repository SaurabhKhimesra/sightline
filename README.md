# SightLine

A ROS 2 workspace for a screwdriving cell where a person and a UR5e build a control
box together. The robot plans in real time from an overhead depth camera, keeps two
hard rules, and is scored by an independent judge:

- **R1** never touch the person.
- **R2** never block the overhead camera's view of the person.

The planner treats the camera's view as a grid of directions. Wherever the camera
sees the worker, that direction is off limits to the arm, from the camera out to
15 cm behind him, with a margin for the arm's size and for how far he could move
before the arm stops. The grid is rebuilt with every picture (25 Hz) and the arm's
next moves are checked against it at 63 points every 10 ms.

**Result, one 81 s cycle (seed 0):** 6 of 6 screws driven, 0 contacts, 0 frames with
the worker's view blocked, closest approach 97 mm, three of the screws driven while
he worked 40 cm from the robot. The same planner run live as a ROS 2 node reproduces
that run to the millimetre. Numbers and how they were measured: `docs/RESULTS_LOG.md`.

Everything here is measured in simulation. Nothing on this page certifies a cell or
replaces a physical measurement.

## Packages

| Package | What it is |
|---|---|
| `sightline_planner` | The robot's own code: the view grid, the 10 ms guard, the task that picks the screw. Imports nothing from the simulation (a test enforces it). |
| `sightline_sim` | The cell in MuJoCo: station, worker and his cycle, cameras with measured noise, the judge that scores R1 and R2 from the simulator's state, the validation gates, the export to Gazebo Sim. |
| `sightline_ros` | The ROS 2 layer: `cell_node` and `planner_node` in lockstep on sim time, the topic contract, `replay_live` into Gazebo and rviz. |
| `sightline_bringup` | Launch files, rviz and Gazebo configurations, the UR5e description on its mount, the recording scripts. |

## Setup

ROS 2 Lyrical with Gazebo Sim, `ros_gz_bridge`, `rviz2`, `robot_state_publisher`,
`ur_description` and `xacro`. The MuJoCo Menagerie for the UR5e model, and the Python
packages of `requirements.txt` in the interpreter that runs the nodes:

```
git clone https://github.com/google-deepmind/mujoco_menagerie ~/mujoco_menagerie
export MUJOCO_MENAGERIE=~/mujoco_menagerie          # the default is this path
pip install -r requirements.txt                    # or into a venv whose site-packages you put on PYTHONPATH
pip install --no-deps sdformat-mjcf                 # only for the Gazebo export

colcon build --symlink-install
source install/setup.bash
```

Offscreen rendering uses EGL (`MUJOCO_GL=egl`, set by the launch files). On a laptop
with two GPUs, point EGL at the discrete one:
`export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json`.

## Run

The planner live, as two nodes in lockstep on sim time (the run does not depend on
how fast the machine is), with rviz and a bag of the small topics:

```
ros2 launch sightline_bringup stage1.launch.py seed:=0 seconds:=full rviz:=true bag:=true
```

Results land in `results/ros2/stage1/` (the judge's yaml, the planner's yaml, every
body's pose at 25 Hz, the bag). In rviz: the UR5e on the joint states, the worker as
the judge's capsules, the planner's grid as point clouds (red where the camera sees
him, orange for cells its own arm hides, dark red the far end 15 cm behind him), the
63 points the guard checks, the hole it is going for.

The judged run rendered by Gazebo Sim, from the recorded poses:

```
python -c "from sightline_sim.station import export_mjcf; export_mjcf('results/ros2/scene')"
ros2 run sightline_sim mjcf_to_sdf results/ros2/scene/sightline_cell.xml results/ros2/gazebo/sightline_cell.sdf
ros2 run sightline_sim make_replay_world results/ros2/scene/sightline_cell.xml \
    results/ros2/gazebo/sightline_cell.sdf results/ros2/gazebo/sightline_replay.sdf --shadows --gui-camera front

GZ_PARTITION=sightline gz sim -s --headless-rendering results/ros2/gazebo/sightline_replay.sdf &
GZ_PARTITION=sightline ros2 run sightline_sim replay_gazebo results/ros2/stage1/run_B4_seed0_poses.npz \
    results/ros2/gazebo/frames --cameras front cycle_hero eyes_cam
ros2 run sightline_sim assemble_video results/ros2/stage1/run_B4_seed0_poses.npz results/ros2/gazebo/frames results/ros2/gazebo
```

Live, in real time, with the Gazebo GUI and rviz open:

```
ros2 launch sightline_bringup replay.launch.py poses:=results/ros2/stage1/run_B4_seed0_poses.npz gui:=true rviz:=true
```

Recording the desktop (`ros2 run sightline_bringup record_windows.sh ...`) and a bag
replayed into rviz (`record_bag_rviz.sh`) are in `sightline_bringup/scripts`;
`docs/MODEL_NOTES.md` says how each is made and why.

The validation gates that produced the numbers, one command each, run from the
workspace root (`ros2 run sightline_sim gate5_planner results/gate5 0 B4 --video`,
and `gate1_stills`, `gate1_numbers`, `gate2_cycle`, `gate3_b0`, `gate4_perception`,
`gate4_hole_views`, `eyes_options`, `eyes_optimise`, `blind_zone`).

Tests (unittest, no other dependency):

```
python -m unittest discover -s src/sightline_planner/test
python -m unittest discover -s src/sightline_sim/test
python -m unittest discover -s src/sightline_ros/test      # needs the ROS 2 environment sourced
```

## Videos

In the release assets of this repository:

- `B4_front_gazebo_seed0.mp4`: the judged run rendered by Gazebo Sim from a camera
  facing the worker, captioned with the judge's record.
- `B4_windows_front_seed0.mp4`: Gazebo's camera live on the GPU beside rviz2, in real time.
- `B4_rviz_seed0.mp4`: the planner's own view during the live run, replayed from the bag.
- `B4_hero_seed0.mp4`: the same run rendered by MuJoCo over the worker's shoulder.

## Documents

- `docs/SPEC.md`: the design, the rules, the gates, and every change since (section 19).
- `docs/MODEL_NOTES.md`: every model choice with its source, or marked as a scene choice.
- `docs/RESULTS_LOG.md`: the numbers of every gate, newest first, with what would change them.
- `docs/CHANGELOG.md`: every bug found and the test that now catches it.

## Stated plainly

- The simulation is MuJoCo; Gazebo Sim renders the recorded runs and does not
  simulate here.
- The lockstep between the cell and the planner makes the run independent of the
  machine's speed. A real cell has no lockstep: the planner's picture costs 137 ms on
  this laptop against the camera's 40 ms period, and must come down or run one
  picture behind before it drives hardware.
- One seed so far. Held-out seeds, the look-around layer and the camera search with
  the frame's mounts are next.
- The worker model, the enclosure, the DIN rail, the terminal blocks and the feeder
  are scene choices, not products. The cameras, the screwdriver, the screws and the
  worker's height are from datasheets and published ranges, cited in the notes.

MIT licence.
