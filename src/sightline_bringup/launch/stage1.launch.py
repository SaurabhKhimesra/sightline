"""Stage 1: the cell and the planner as two nodes in lockstep on sim time.

    ros2 launch sightline_bringup stage1.launch.py seed:=0 seconds:=full rviz:=true bag:=true

Run from the workspace root: the nodes write their results under results/ros2/stage1.
The cell waits for the planner's ready message before its first tick, and every tick
waits for the planner's command, so the run is the same run at any speed the machine
manages (docs/notes.md, "The planner live on ROS 2").
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

BAG_TOPICS = ["/joint_states", "/sightline/tick", "/sightline/cmd", "/sightline/frame", "/sightline/judge",
              "/sightline/grid/seen", "/sightline/grid/remembered", "/sightline/grid/behind", "/sightline/arm_points",
              "/sightline/target", "/sightline/worker", "/sightline/station", "/sightline/caption_image",
              "/sightline/planner_caption", "/tf", "/clock", "/sightline/cell/calibration", "/sightline/cell/blind_cells",
              "/sightline/cell/taught", "/sightline/planner/ready"]


def nodes(context):
    seed = LaunchConfiguration("seed").perform(context)
    seconds = LaunchConfiguration("seconds").perform(context)
    out = LaunchConfiguration("out").perform(context)
    cell_args = ["--seed", seed, "--out", out] + ([] if seconds in ("", "full") else ["--seconds", seconds])
    return [
        Node(package="sightline_ros", executable="planner_node", name="sightline_planner", output="screen",
             arguments=["--seed", seed, "--out", out]),
        Node(package="sightline_ros", executable="cell_node", name="sightline_cell", output="screen", arguments=cell_args),
        ExecuteProcess(cmd=["ros2", "bag", "record", "-o", out + "/bag_seed" + seed, "--compression-mode", "file",
                            "--compression-format", "zstd", "--topics"] + BAG_TOPICS,
                       output="log", condition=IfCondition(LaunchConfiguration("bag"))),
    ]


def generate_launch_description():
    share = FindPackageShare("sightline_bringup")
    description = ParameterValue(Command([FindExecutable(name="xacro"), " ",
                                          PathJoinSubstitution([share, "urdf", "ur5e_cell.urdf.xacro"])]), value_type=str)
    return LaunchDescription([
        DeclareLaunchArgument("seed", default_value="0", description="the episode's seed"),
        DeclareLaunchArgument("seconds", default_value="full", description="seconds of the cycle to run, or full"),
        DeclareLaunchArgument("out", default_value="results/ros2/stage1", description="where the results go"),
        DeclareLaunchArgument("rviz", default_value="false", description="open rviz on the run"),
        DeclareLaunchArgument("bag", default_value="false", description="record the small topics to a bag"),
        SetEnvironmentVariable("MUJOCO_GL", "egl"),
        OpaqueFunction(function=nodes),
        Node(package="robot_state_publisher", executable="robot_state_publisher", output="log",
             parameters=[{"robot_description": description, "use_sim_time": True}],
             condition=IfCondition(LaunchConfiguration("rviz"))),
        Node(package="rviz2", executable="rviz2", output="log",
             arguments=["-d", PathJoinSubstitution([share, "config", "sightline_stage1.rviz"])],
             parameters=[{"use_sim_time": True}], condition=IfCondition(LaunchConfiguration("rviz"))),
    ])
