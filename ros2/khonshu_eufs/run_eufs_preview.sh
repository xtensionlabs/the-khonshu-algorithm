#!/usr/bin/env bash
set -Ee -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SIM_PID=""
PREVIEW_PID=""
RVIZ_PID=""

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  for pid in "${SIM_PID}" "${PREVIEW_PID}" "${RVIZ_PID}"; do
    if [[ -n "${pid}" ]]; then
      kill -INT "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${SIM_PID}" "${PREVIEW_PID}" "${RVIZ_PID}"; do
    if [[ -n "${pid}" ]]; then
      wait "${pid}" 2>/dev/null || true
    fi
  done
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

source ~/fsds_ws/scripts/ros_env.sh
export EUFS_MASTER=/home/brian_terer/eufs_ws

echo "Starting EUFS small_track..."
ros2 launch /home/brian_terer/eufs_ws/src/eufs_sim/eufs_launcher/launch/simulation.launch.py \
  track:=small_track gazebo_gui:=true rviz:=false \
  publish_gt_tf:=true pub_ground_truth:=true &
SIM_PID=$!

echo "Starting Khonshu preview (500 ms visualization deadline) and configured RViz..."
bash "${SCRIPT_DIR}/run_preview.sh" --ros-args \
  -p use_sim_time:=true \
  -p cones_topic:=/ground_truth/cones \
  -p state_topic:=/ground_truth/odom \
  -p state_type:=nav_msgs/Odometry \
  -p map_frame:=map \
  -p base_frame:=base_footprint \
  -p maximum_cycle_time_ms:=500.0 &
PREVIEW_PID=$!

rviz2 -d "${SCRIPT_DIR}/khonshu_preview.rviz" --ros-args -p use_sim_time:=true &
RVIZ_PID=$!
wait "${RVIZ_PID}"
