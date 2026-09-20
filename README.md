# SightLine

A UR5e that drives screws next to a person without touching them and without
getting between them and the camera that watches them. ROS 2 Lyrical, MuJoCo for
the physics, Gazebo Sim and rviz2 for looking at it.

![one cycle from the front](docs/media/front.gif)

*One assembly cycle, rendered by Gazebo from a camera facing the worker. The band at
the top is what the judge recorded. Full videos are under
[releases](../../releases).*

## What it does

An overhead depth camera looks down at the bench. I treat the camera's view as a
grid of directions. Every direction in which it currently sees the person is off
limits for the arm, from the camera down to 15 cm behind them, plus a margin for the
arm's own thickness and for how far the person could move before the arm can stop.
The grid is rebuilt for every frame (25 Hz), and the arm's next moves are checked
against it at 63 points along the links every 10 ms. If a move would enter the grid
the arm brakes, backs out along its own path, or retreats to a standoff. It also
picks which screw to drive next, and from which side, based on what is clear.

The two rules are scored by a judge that reads the simulator state directly. The
planner never sees that state; it gets a noisy, one-frame-late depth image and its
own joint angles, like it would on a real cell.

1. never touch the person
2. never block the overhead camera's view of the person

## Results

One 81 s cycle, seed 0. B0 is the same controller with the rules switched off, B2 and
B3 use the usual distance dampers (B3 also dampens the sight lines), B4 is the grid.

|                                        | B0     | B2     | B3     | B4 (grid) |
|----------------------------------------|--------|--------|--------|-----------|
| screws driven                          | 6 / 6  | 4 / 6  | 4 / 6  | 6 / 6     |
| contacts the robot caused              | 5      | 5      | 8      | 0         |
| contacts the person caused             | 35     | 1      | 29     | 0         |
| deepest contact                        | 76 mm  | 60 mm  | 76 mm  | none, closest 97 mm |
| frames with the person hidden          | 10.1 % | 9.8 %  | 14.4 % | 0.0 %     |
| times it asked the person to move      | 0      | 3      | 3      | 0         |
| compute per 10 ms cycle, median        | 0.2 ms | 7.4 ms | 9.3 ms | 1.0 ms    |

Three of B4's six screws went in while the worker was busy 40 cm away from the
robot, which was the point of ordering the tasks that way. Running the planner as a
ROS 2 node instead of in-process gives the same run to the millimetre (checked, see
`docs/results.md`).

This is one seed, in simulation. It does not certify anything.

## Layout

```
src/sightline_planner   the robot's own code: view grid, guard, task. Imports nothing from the sim.
src/sightline_sim       the cell in MuJoCo: station, worker, cameras, the judge, the gate scripts, Gazebo export
src/sightline_ros       cell_node and planner_node (lockstep on sim time), replay_live, the topic contract
src/sightline_bringup   launch files, rviz/Gazebo configs, UR5e xacro, recording scripts
docs/                   design.md (the spec and every change since), notes.md (model choices with sources), results.md
results/                the numbers of every gate; videos, frames and bags are not committed
```

## Install

Tested on Ubuntu with ROS 2 Lyrical, Gazebo Sim 10, Python 3.14, a GTX 1650.

```bash
sudo apt install ros-lyrical-ros-gz-bridge ros-lyrical-ur-description ros-lyrical-xacro \
                 ros-lyrical-robot-state-publisher ros-lyrical-rviz2
git clone https://github.com/google-deepmind/mujoco_menagerie ~/mujoco_menagerie   # UR5e model
pip install -r requirements.txt            # into the python that runs your ROS 2 nodes
pip install --no-deps sdformat-mjcf        # only needed for the Gazebo export

colcon build --symlink-install
source install/setup.bash
```

If the Menagerie checkout is somewhere else, set `MUJOCO_MENAGERIE`. Rendering is
offscreen through EGL; on a laptop with two GPUs point it at the discrete one with
`export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json`.

## Run

The planner live, as two nodes in lockstep on sim time (so the run is the same on a
slow machine and a fast one), with rviz and a bag:

```bash
ros2 launch sightline_bringup stage1.launch.py seed:=0 seconds:=full rviz:=true bag:=true
```

Results go to `results/ros2/stage1/`: the judge's numbers, the planner's numbers,
every body's pose at 25 Hz, the bag. rviz shows the UR5e on the joint states, the
worker as the judge's capsules, and the planner's grid as point clouds (red where
the camera sees the person, orange for cells the arm hides and the planner keeps for
half a second, dark red the far end 15 cm behind them).

Gazebo rendering of a recorded run:

```bash
python -c "from sightline_sim.station import export_mjcf; export_mjcf('results/ros2/scene')"
ros2 run sightline_sim mjcf_to_sdf results/ros2/scene/sightline_cell.xml results/ros2/gazebo/sightline_cell.sdf
ros2 run sightline_sim make_replay_world results/ros2/scene/sightline_cell.xml \
    results/ros2/gazebo/sightline_cell.sdf results/ros2/gazebo/sightline_replay.sdf --shadows --gui-camera front

GZ_PARTITION=sightline gz sim -s --headless-rendering results/ros2/gazebo/sightline_replay.sdf &
GZ_PARTITION=sightline ros2 run sightline_sim replay_gazebo results/ros2/stage1/run_B4_seed0_poses.npz \
    results/ros2/gazebo/frames --cameras front cycle_hero eyes_cam
ros2 run sightline_sim assemble_video results/ros2/stage1/run_B4_seed0_poses.npz results/ros2/gazebo/frames results/ros2/gazebo
```

Or live, with the Gazebo GUI and rviz open:

```bash
ros2 launch sightline_bringup replay.launch.py poses:=results/ros2/stage1/run_B4_seed0_poses.npz gui:=true rviz:=true
```

The gate scripts that produced the numbers in `results/` run one at a time from the
workspace root, e.g. `ros2 run sightline_sim gate5_planner results/gate5 0 B4 --video`.
Tests are plain unittest: `python -m unittest discover -s src/sightline_sim/test`
(and the same for `sightline_planner` and `sightline_ros`, the last one with ROS
sourced).

## Known issues and what's next

- Perception costs about 137 ms per frame on this laptop against the camera's 40 ms.
  The lockstep hides that in simulation; a real cell has no lockstep. Needs to come
  down, or run one frame behind, before this touches hardware.
- One seed. More seeds and a proper search for the camera mount (the cell has an
  aluminium frame that would carry it better than the pole) are next.
- The worker model, enclosure, DIN rail, terminal blocks and feeder are made up. The
  cameras, screwdriver, screws and the worker's height come from datasheets and
  published ranges; `docs/notes.md` has the sources.
- Gazebo's lights only cast shadows in the headless render, the GUI crashes on them.

MIT licence.
