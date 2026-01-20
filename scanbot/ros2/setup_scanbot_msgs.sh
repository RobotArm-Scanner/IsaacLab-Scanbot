#!/usr/bin/env bash
set -euo pipefail

# One-shot setup/build for scanbot_msgs inside the container.
# Assumes this repo is mounted at /workspace/isaaclab.

if [ ! -f /opt/ros/humble/setup.bash ]; then
  echo "[ERROR] ROS 2 Humble not found at /opt/ros/humble." >&2
  exit 1
fi

WS_ROOT="/workspace/isaaclab/scanbot/ros2"
PKG_DIR="${WS_ROOT}/scanbot_msgs"

if [ ! -d "${PKG_DIR}" ]; then
  echo "[ERROR] scanbot_msgs not found at ${PKG_DIR}" >&2
  exit 1
fi

# Ensure system pip is available
apt-get update
apt-get install -y --no-install-recommends python3-pip python3-wheel
apt-get clean
rm -rf /var/lib/apt/lists/*

# ROS 2 interface generators need these in the system python (3.10)
/usr/bin/python3 -m pip install --no-cache-dir empy==3.3.4 catkin_pkg lark numpy

# Build with system python to match ROS 2 toolchain
# ROS setup scripts expect some vars to be unset; avoid nounset failures.
set +u
source /opt/ros/humble/setup.bash
set -u
# Force CMake to use system Python 3.10 (ROS 2 runtime)
export Python3_EXECUTABLE=/usr/bin/python3
export PYTHON_EXECUTABLE=/usr/bin/python3
cd "${WS_ROOT}"
rm -rf build/scanbot_msgs install/scanbot_msgs log
colcon build --packages-select scanbot_msgs --cmake-args \
  -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DPYTHON_EXECUTABLE=/usr/bin/python3

echo "[INFO] Build complete. Source these before running Isaac Sim:"
echo "  source /opt/ros/humble/setup.bash"
echo "  source /workspace/isaaclab/scanbot/ros2/install/setup.bash"
