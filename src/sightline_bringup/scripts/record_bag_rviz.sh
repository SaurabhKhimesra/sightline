#!/usr/bin/env bash
# Replay a stage 1 bag into rviz2 at real time and record the rviz window.
#
#   ros2 run sightline_bringup record_bag_rviz.sh results/ros2/stage1/bag_seed0 results/ros2/stage1/B4_rviz_seed0.mp4 5.81
#
# The live run went slower than real time (the planner's picture costs more than 40 ms on
# this machine), and a bag keeps the wall time spacing of the recording, so it is played
# back at 1 / the run's real time factor (the third argument, from cell_*.yaml) and rviz
# follows the sim time the cell put on /clock. The rate is one number for the whole run,
# so the playback runs a little fast where the run was slow and the other way round.
# The window is captured with x11grab through XWayland.
source /opt/ros/lyrical/setup.bash
set -euo pipefail
bag="${1:?bag folder}"
out="${2:?output mp4}"
rate="${3:-1.0}"
here="$(cd "$(dirname "$0")" && pwd)"
py="${SIGHTLINE_PYTHON:-python3}"
work=$(mktemp -d /tmp/sightline_bagrviz_XXXX)
export ROS_DOMAIN_ID=$(( 60 + RANDOM % 30 ))
export QT_QPA_PLATFORM=xcb
export DISPLAY="${DISPLAY:-:0}"
FF=$($py -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
descendants() { local kids; kids=$(ps -eo pid=,ppid= | awk -v p="$1" '$2 == p {print $1}'); for k in $kids; do descendants "$k"; echo "$k"; done; }
stop_tree() { local tree; tree="$(descendants "$1") $1"; kill -TERM $tree 2>/dev/null || true; sleep 2; kill -KILL $tree 2>/dev/null || true; }
pids=()
cleanup() { for p in ${grab_pid:-}; do kill -TERM "$p" 2>/dev/null || true; done; for p in "${pids[@]}"; do stop_tree "$p"; done; }
trap cleanup EXIT

before=$(xwininfo -root -tree 2>/dev/null | grep -i "RViz" | awk '{print $1}' | tr '\n' ' ' || true)
xacro "$(ros2 pkg prefix ur_description)/share/ur_description/urdf/ur.urdf.xacro" ur_type:=ur5e name:=ur5e \
    | sed 's/<link name="world"/<link name="robot_mount"/; s/<parent link="world"/<parent link="robot_mount"/' > "$work/ur5e.urdf"
ros2 run robot_state_publisher robot_state_publisher --ros-args -p use_sim_time:=true \
    -p robot_description:="$(cat "$work/ur5e.urdf")" > "$work/rsp.log" 2>&1 &
pids+=($!)
rviz2 -d "$share/config/sightline_stage1_bag.rviz" --ros-args -p use_sim_time:=true > "$work/rviz.log" 2>&1 &
pids+=($!)
win=""
for _ in $(seq 90); do
    win=$(xwininfo -root -tree 2>/dev/null | grep -i "RViz" | grep -v "has no name" | awk -v skip=" $before " '
        { if (index(skip, " " $1 " ")) next
          for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+x[0-9]+[+-]/) { split($i, g, /[x+-]/); a = g[1] * g[2]
          if (g[1] >= 300 && g[2] >= 300 && a > best) { best = a; id = $1 } } } END { if (id) print id }' || true)
    [ -n "$win" ] && break
    sleep 1
done
[ -n "$win" ] || { echo "no rviz window"; exit 1; }
sleep 8
"$FF" -loglevel error -y -f x11grab -framerate 25 -window_id "$win" -i "$DISPLAY" -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" \
    -c:v libx264 -preset ultrafast -crf 18 -pix_fmt yuv420p "$work/rviz.mkv" > "$work/grab.log" 2>&1 &
grab_pid=$!
echo "recording rviz window $win"
sleep 2
ros2 bag play "$bag" --rate "$rate" > "$work/play.log" 2>&1
sleep 2
kill -TERM "$grab_pid" 2>/dev/null || true
for _ in $(seq 40); do kill -0 "$grab_pid" 2>/dev/null || break; sleep 0.5; done
grab_pid=""
"$FF" -loglevel error -y -i "$work/rviz.mkv" -c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p -movflags +faststart "$out"
"$FF" -loglevel error -y -sseof -3 -i "$out" -frames:v 1 "${out%.mp4}_last.png" || true
echo "wrote $out ($(du -h "$out" | cut -f1)); logs in $work"
