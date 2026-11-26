#!/usr/bin/env bash
set -Ee -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SIM_PID=""
PREVIEW_PID=""
FOLLOWER_PID=""
RVIZ_PID=""
SHOW_RVIZ="${KHONSHU_SHOW_RVIZ:-1}"
PUBLISH_CONSTRUCTION=false
ROS_ENV_SCRIPT="${KHONSHU_ROS_ENV_SCRIPT:-${HOME}/fsds_ws/scripts/ros_env.sh}"
EUFS_WORKSPACE="${KHONSHU_EUFS_WS:-${HOME}/eufs_ws}"
KHONSHU_PYTHON_ENV="${KHONSHU_PYTHON_ENV:-${HOME}/.venvs/khonshu}"
EUFS_LAUNCH_FILE="${EUFS_WORKSPACE}/src/eufs_sim/eufs_launcher/launch/simulation.launch.py"
if [[ "${SHOW_RVIZ}" == "1" ]]; then
  PUBLISH_CONSTRUCTION=true
fi

for required_path in \
  "${ROS_ENV_SCRIPT}" \
  "${KHONSHU_PYTHON_ENV}/bin/activate" \
  "${EUFS_LAUNCH_FILE}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Required EUFS setup path does not exist: ${required_path}" >&2
    exit 1
  fi
done

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  for pid in "${FOLLOWER_PID}" "${PREVIEW_PID}" "${RVIZ_PID}" "${SIM_PID}"; do
    if [[ -n "${pid}" ]]; then
      kill -INT -- "-${pid}" 2>/dev/null || kill -INT "${pid}" 2>/dev/null || true
    fi
  done
  sleep 1
  for pid in "${FOLLOWER_PID}" "${PREVIEW_PID}" "${RVIZ_PID}" "${SIM_PID}"; do
    if [[ -n "${pid}" ]]; then
      kill -TERM -- "-${pid}" 2>/dev/null || true
    fi
  done
  sleep 1
  for pid in "${FOLLOWER_PID}" "${PREVIEW_PID}" "${RVIZ_PID}" "${SIM_PID}"; do
    if [[ -n "${pid}" ]]; then
      kill -KILL -- "-${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

source "${ROS_ENV_SCRIPT}"
source "${KHONSHU_PYTHON_ENV}/bin/activate"
export EUFS_MASTER="${EUFS_WORKSPACE}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "Starting visible EUFS rectangle course in velocity command mode..."
setsid ros2 launch "${EUFS_LAUNCH_FILE}" \
  track:=rectangle gazebo_gui:=true rviz:=false publish_gt_tf:=true \
  pub_ground_truth:=true commandMode:=velocity launch_group:=no_perception \
  show_rqt_gui:=false &
SIM_PID=$!
sleep 1
if ! kill -0 "${SIM_PID}" 2>/dev/null; then
  set +e
  wait "${SIM_PID}"
  launch_status=$?
  set -e
  echo "EUFS launch exited during startup (status ${launch_status})."
  exit 1
fi

echo "Starting the Delaunay/Voronoi Stage A planner..."
setsid bash "${SCRIPT_DIR}/run_preview.sh" --ros-args \
  -p use_sim_time:=true \
  -p cones_topic:=/ground_truth/cones \
  -p state_topic:=/ground_truth/odom \
  -p state_type:=nav_msgs/Odometry \
  -p map_frame:=map \
  -p base_frame:=base_footprint \
  -p planner_backend:=stage_a \
  -p warm_up:=false \
  -p publish_construction_markers:=${PUBLISH_CONSTRUCTION} \
  -p maximum_cycle_time_ms:=200.0 &
PREVIEW_PID=$!

if [[ "${SHOW_RVIZ}" == "1" ]]; then
  echo "Starting the Khonshu RViz path and cone view..."
  setsid rviz2 -d "${SCRIPT_DIR}/khonshu_preview.rviz" --ros-args \
    -p use_sim_time:=true &
  RVIZ_PID=$!
else
  echo "RViz disabled for this run; the visible Gazebo course remains open."
fi

echo "Starting low-speed pure pursuit on /cmd; it exits after one lap or timeout..."
setsid python3 "${SCRIPT_DIR}/eufs_pure_pursuit.py" --ros-args \
  -p use_sim_time:=true \
  -p path_topic:=/khonshu/centerline \
  -p state_topic:=/ground_truth/odom \
  -p command_topic:=/cmd \
  -p speed_mps:=0.8 \
  -p lookahead_m:=1.0 \
  -p lap_detection_enabled:=true \
  -p minimum_lap_distance_m:=20.0 \
  -p lap_timeout_s:=160.0 &
FOLLOWER_PID=$!

wait "${FOLLOWER_PID}"
echo "Lap complete; keeping the Gazebo view open for ${KHONSHU_POST_LAP_HOLD_S:-10} seconds."
sleep "${KHONSHU_POST_LAP_HOLD_S:-10}"
