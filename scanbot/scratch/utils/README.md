# Scanbot scratch utils (ROS2 host + TargetTcp)

## Why the "rosidl_typesupport_c" error happens
- `scanbot/bin/scanbot.sh` runs Isaac Lab via `script -c ...`.
- `script` uses `/bin/sh` by default, so `source /opt/ros/humble/setup.bash` was **not** being executed.
- Without ROS2 env, `LD_LIBRARY_PATH` and `PYTHONPATH` miss `scanbot_msgs` libs →
  `Could not import 'rosidl_typesupport_c' for package 'scanbot_msgs'`.

### Current fix in this repo
- `scanbot/bin/scanbot.sh` now runs the command under `bash -lc` and sources:
  - `/opt/ros/humble/setup.bash`
  - `/workspace/isaaclab/scanbot/ros2/install/setup.bash`

If you change the launcher, keep the **bash login shell + ROS2 setup** or the typesupport error will come back.

## Host conda ROS2 setup (robostack)
The goal is a ROS2 Humble env on the **host** so you can send `TargetTcp` actions.

### 1) Create conda env
```bash
# scripted helper (optional)
./scanbot/scratch/utils/setup_host_ros2_conda.sh scanbot-ros2

# or manual (pick one: conda / mamba / micromamba)
conda create -y -n scanbot-ros2 -c conda-forge -c robostack-staging \
  python=3.10 \
  ros-humble-ros-base \
  ros-humble-ros2cli \
  colcon-common-extensions
```
If a package is missing, try replacing `robostack-staging` with `robostack`.

### 2) Build `scanbot_msgs` on the host
```bash
conda activate scanbot-ros2
cd /mnt/ext_sda1/dev/IsaacLab-ScanBot/scanbot/ros2
colcon build --packages-select scanbot_msgs
source install/setup.bash
```
Quick check:
```bash
python - <<'PY'
from scanbot_msgs.action import TargetTcp
print('ok', TargetTcp)
PY
```

## Sending TargetTcp from the host
Use `send_target_tcp.py` after activating the conda env and sourcing the workspace.

```bash
conda activate scanbot-ros2
cd /mnt/ext_sda1/dev/IsaacLab-ScanBot
source scanbot/ros2/install/setup.bash
python scanbot/scratch/utils/send_target_tcp.py \
  --pose "0.40,0.00,0.20,0,0,0,1" \
  --pos-tol 0.005 --rot-tol 0.02 --timeout-sec 10
```

## Capture current TCP pose (demo slots)
Inside the container (or any ROS2 environment that can see `/scanbot/tcp_pose`):
```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 /workspace/isaaclab/scanbot/scratch/utils/save_demo_pose.py --slot 1
```
Saved to: `scanbot/scratch/utils/demo_poses.json`

### Notes
- If host ↔ container ROS2 discovery fails, set a matching domain ID:
  ```bash
  export ROS_DOMAIN_ID=0
  export ROS_LOCALHOST_ONLY=0
  ```
- Default action name is `/scanbot/target_tcp` (matches the extension).
