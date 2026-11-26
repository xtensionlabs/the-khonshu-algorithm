#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

source ~/fsds_ws/scripts/ros_env.sh
source ~/.venvs/khonshu/bin/activate
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

exec python3 "${SCRIPT_DIR}/centerline_preview.py" "$@"
