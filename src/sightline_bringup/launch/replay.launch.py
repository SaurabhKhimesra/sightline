"""Play a recorded run into the Gazebo replay world and ROS 2 in real time, with the
Gazebo GUI and rviz if asked.

    ros2 launch sightline_bringup replay.launch.py poses:=results/ros2/run_B4_seed0_poses.npz gui:=true rviz:=true

The world comes from `ros2 run sightline_sim make_replay_world` (README, "Gazebo and
ROS 2"). Every Gazebo process here shares one GZ_PARTITION so other Gazebo sessions on
the machine stay apart; the server renders on the GPU named by
__EGL_VENDOR_LIBRARY_FILENAMES when that is set in the environment.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare("sightline_bringup")
    world = LaunchConfiguration("world")
    description = ParameterValue(Command([FindExecutable(name="xacro"), " ",
                                          PathJoinSubstitution([share, "urdf", "ur5e_cell.urdf.xacro"])]), value_type=str)
    return LaunchDescription([
        DeclareLaunchArgument("poses", default_value="results/ros2/run_B4_seed0_poses.npz", description="the recorded run"),
        DeclareLaunchArgument("world", default_value="results/ros2/gazebo/sightline_replay.sdf", description="the replay world"),
        DeclareLaunchArgument("gui", default_value="false", description="open the Gazebo GUI on the server"),
        DeclareLaunchArgument("rviz", default_value="true", description="open rviz"),
        DeclareLaunchArgument("end", default_value="1e9", description="stop after this many seconds of the run"),
        SetEnvironmentVariable("GZ_PARTITION", "sightline_replay_" + str(os.getpid())),
        SetEnvironmentVariable("QT_QPA_PLATFORM", "xcb"),
        ExecuteProcess(cmd=["gz", "sim", "-s", "--headless-rendering", "-v", "1", world], output="log"),
        ExecuteProcess(cmd=["gz", "sim", "-g", "-v", "1", "--gui-config",
                            PathJoinSubstitution([share, "config", "gazebo_gui.config"])],
                       output="log", condition=IfCondition(LaunchConfiguration("gui"))),
        Node(package="robot_state_publisher", executable="robot_state_publisher", output="log",
             parameters=[{"robot_description": description}]),
        Node(package="ros_gz_bridge", executable="parameter_bridge", output="log",
             arguments=["/replay/eyes_cam@sensor_msgs/msg/Image[gz.msgs.Image"]),
        Node(package="rviz2", executable="rviz2", output="log",
             arguments=["-d", PathJoinSubstitution([share, "config", "sightline_replay.rviz"])],
             condition=IfCondition(LaunchConfiguration("rviz"))),
        # the replay waits for the world's model list itself; the delay only spares the log
        TimerAction(period=10.0, actions=[
            Node(package="sightline_ros", executable="replay_live", name="sightline_replay", output="screen",
                 arguments=[LaunchConfiguration("poses"), "--end", LaunchConfiguration("end")])]),
    ])
