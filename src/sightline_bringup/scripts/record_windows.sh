#!/usr/bin/env bash
# Screen-record the live desktop replay of a judged run: the Gazebo Sim window and the rviz
# window side by side, as an engineer would have them open. Nothing is drawn over them.
#
#   ros2 run sightline_bringup record_windows.sh results/ros2/run_B4_seed0_poses.npz results/ros2/gazebo/B4_windows_seed0.mp4 [end_s]
#
# What runs: a Gazebo server (headless rendering for the eyes camera) and the Gazebo GUI on it,
# robot_state_publisher with the UR5e description from ur_description, ros_gz_bridge for the
# eyes camera picture, rviz2 with the replay layout, and the replay_live node, which
# sets the recorded poses in Gazebo and publishes the recorded joint angles, the worker's
# capsules and the judge's words to ROS 2, in real time.
#
# The desktop is GNOME on Wayland, where one app cannot read another's pixels; these windows run
# through XWayland (QT_QPA_PLATFORM=xcb) and ffmpeg's x11grab reads each window on its own. The
# pattern, and the reasons, are locrec's ros2/record_windows.sh. Processes are stopped by PID.
# ROS's setup script reads variables it never set, so it is sourced before the strict mode
source /opt/ros/lyrical/setup.bash
set -euo pipefail
poses="${1:?poses npz}"
out="${2:?output mp4}"
end_s="${3:-9999}"
fps=25
here="$(cd "$(dirname "$0")" && pwd)"
py="${SIGHTLINE_PYTHON:-python3}"
world="${SIGHTLINE_WORLD:-$PWD/results/ros2/gazebo/sightline_replay.sdf}"   # from make_replay_world
share="$(ros2 pkg prefix sightline_bringup)/share/sightline_bringup"
work=$(mktemp -d /tmp/sightline_windows_XXXX)
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
# The display runs on the Intel GPU. The Gazebo GUI needs the NVIDIA GPU (on the Intel it
# repainted at 7.6 fps), but a window rendered on the NVIDIA GPU reads back through X at one
# to thirteen frames a second, Gazebo's own VideoRecorder plugin has no remote trigger in
# this version, and GNOME's screencast service refuses scripts. So Gazebo's picture is taken
# where it is made: the front camera of the replay world, rendered by the server on the
# NVIDIA GPU as the run plays, encoded straight from the frames by the frame_grabber tool,
# one frame per 40 ms of sim time. rviz, on the Intel GPU, is screen-grabbed at 25 fps.
export GZ_PARTITION="sightline_live_$$"
export ROS_DOMAIN_ID=$(( 60 + RANDOM % 30 ))
export QT_QPA_PLATFORM=xcb
export DISPLAY="${DISPLAY:-:0}"
FF=$($py -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
camera="${SIGHTLINE_CAMERA:-front}"
lead=2.0

descendants() {  # every process under a PID, deepest first
    local kids; kids=$(ps -eo pid=,ppid= | awk -v p="$1" '$2 == p {print $1}')
    for k in $kids; do descendants "$k"; echo "$k"; done
}
stop_tree() {  # by PID, never by pattern: a pattern would match this script's own command line
    local tree; tree="$(descendants "$1") $1"
    kill -TERM $tree 2>/dev/null || true
    sleep 3
    kill -KILL $tree 2>/dev/null || true
}
pids=()
cleanup() {
    for p in ${grab_pid:-}; do kill -TERM "$p" 2>/dev/null || true; done
    touch "$work/.grabber_stop" 2>/dev/null || true
    for p in "${pids[@]}"; do stop_tree "$p"; done
}
trap cleanup EXIT

# the UR5e description: ur_description's model with its root link named after our mount frame
xacro "$(ros2 pkg prefix ur_description)/share/ur_description/urdf/ur.urdf.xacro" ur_type:=ur5e name:=ur5e \
    | sed 's/<link name="world"/<link name="robot_mount"/; s/<parent link="world"/<parent link="robot_mount"/' > "$work/ur5e.urdf"

gz sim -s --headless-rendering -v 1 "$world" > "$work/server.log" 2>&1 &
pids+=($!)
sleep 8
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:="$(cat "$work/ur5e.urdf")" > "$work/rsp.log" 2>&1 &
pids+=($!)
ros2 run ros_gz_bridge parameter_bridge "/replay/eyes_cam@sensor_msgs/msg/Image[gz.msgs.Image" > "$work/bridge.log" 2>&1 &
pids+=($!)
rviz2 -d "$share/config/sightline_replay.rviz" -qwindowgeometry 1280x720+0+0 > "$work/rviz.log" 2>&1 &
rviz_pid=$!; pids+=($rviz_pid)

# the rviz window, found by the process that owns it (the desktop has other sessions'
# windows), once the same window has been there for 8 s in a row
window_of() {  # pid -> the largest window of that process tree
    local tree; tree="$(descendants "$1") $1"
    ros2 run sightline_ros xwin find $tree 2>/dev/null | head -1 | awk '{print $1}'
}
settle() {
    local last="" seen=0 id
    for _ in $(seq 180); do
        id=$(window_of "$1")
        if [ -n "$id" ] && [ "$id" = "$last" ]; then seen=$((seen + 1)); else seen=0; fi
        last="$id"
        [ "$seen" -ge 8 ] && { echo "$id"; return 0; }
        sleep 1
    done
    return 0
}
echo "waiting for the rviz window to settle..."
rviz_win=$(settle "$rviz_pid")
[ -n "$rviz_win" ] || { echo "the rviz window never settled; what X had:"; xwininfo -root -tree 2>/dev/null | grep -i rviz | head; exit 1; }
sleep 5   # rviz loads the robot meshes

# Gazebo's camera into a video as the run plays (the server renders it once something subscribes)
rm -f "$work/.grabber_stop" "$work/.grabber_ready"
ros2 run sightline_sim frame_grabber "$work" "$camera" --video "$work/gazebo_raw.mp4" --fps $fps > "$work/grabber.log" 2>&1 &
pids+=($!)
for _ in $(seq 50); do [ -f "$work/.grabber_ready" ] && break; sleep 0.2; done
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader > "$work/gpu.txt" 2>&1 || true

# the grab starts first and the driver logs the wall time of its first frame; the
# difference is trimmed from the grab when the two are put side by side
even="crop=trunc(iw/2)*2:trunc(ih/2)*2"
grab_start=$(date +%s.%N)
"$FF" -loglevel error -y -f x11grab -framerate $fps -window_id "$rviz_win" -i "$DISPLAY" -vf "$even" \
    -c:v libx264 -preset ultrafast -crf 18 -pix_fmt yuv420p "$work/rviz.mkv" > "$work/rviz_grab.log" 2>&1 &
grab_pid=$!
seconds_arg=(--end "$end_s")
ros2 run sightline_ros replay_live "$poses" "${seconds_arg[@]}" --lead $lead > "$work/replay.log" 2>&1 &
replay_pid=$!
echo "recording: Gazebo's $camera camera by the frame grabber, rviz window $rviz_win by x11grab"
nvidia-smi 2>/dev/null | grep -E "gz|rviz|python" >> "$work/gpu.txt" || true
wait $replay_pid || true
grep -v "^\[INFO\]" "$work/replay.log" || true
grep "done:" "$work/replay.log" | tail -1
sleep 1
kill -TERM "$grab_pid" 2>/dev/null || true
touch "$work/.grabber_stop"
for _ in $(seq 60); do kill -0 "$grab_pid" 2>/dev/null || break; sleep 0.5; done
grab_pid=""
for _ in $(seq 60); do grep -q "frames encoded" "$work/grabber.log" 2>/dev/null && break; sleep 0.5; done
cat "$work/grabber.log" | tail -1
[ -s "$work/gazebo_raw.mp4" ] || { echo "no Gazebo video; logs in $work"; exit 1; }

# the judge's words on Gazebo's picture, then side by side with rviz at 1280x720 each, real time
MUJOCO_GL=egl ros2 run sightline_sim caption_video "$poses" "$work/gazebo_raw.mp4" "$work/gazebo.mp4" --lead $lead 2>&1 | grep -v "UserWarning\|spec.compile\|njmax\|nconmax\|integrator\|Attach conflict\|host.compile" | tail -1
first=$(grep -o "first frame at wall [0-9.]*" "$work/replay.log" | awk '{print $5}')
offset=$(awk -v a="${first:-$grab_start}" -v b="$grab_start" 'BEGIN { d = a - b - 0.15; if (d < 0) d = 0; printf "%.3f", d }')
echo "rviz grab started $offset s before the first frame; trimmed"
"$FF" -loglevel error -y -i "$work/gazebo.mp4" -ss "$offset" -i "$work/rviz.mkv" -filter_complex "\
[0:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[a];\
[1:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[b];\
[a][b]hstack=inputs=2:shortest=1,fps=$fps[v]" -map "[v]" -c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p -movflags +faststart "$out"
"$FF" -loglevel error -y -sseof -3 -i "$out" -frames:v 1 "${out%.mp4}_last.png" || true
echo "wrote $out ($(du -h "$out" | cut -f1)); raw videos and logs kept in $work"
