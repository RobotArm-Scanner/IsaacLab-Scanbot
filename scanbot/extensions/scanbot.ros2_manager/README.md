# Scanbot ROS2 Manager

Provides ROS 2 interfaces for driving and observing Scanbot in Isaac Lab:
- TCP target + teleport actions
- Joint/TCP state topics
- Camera RGB/depth/info/pose topics (and optional PointCloud2)
- Marker draw/clear topics
- Reset + joint limit services

## Action
`scanbot_msgs/action/TargetTcp`

Action name:
- `/scanbot/target_tcp`

Goal:
- `geometry_msgs/PoseStamped target` (base frame)
- `float32 pos_tolerance` (meters, optional)
- `float32 rot_tolerance` (radians, optional)
- `float32 timeout_sec` (seconds, optional; <= 0 uses default)

Feedback:
- `float32 pos_error`
- `float32 rot_error`

Result:
- `bool success`
- `string message`

## Topics
TCP pose:
- `geometry_msgs/msg/PoseStamped` on `/scanbot/tcp_pose`
- `header.frame_id`: always `"base"`

Joint state:
- `sensor_msgs/msg/JointState` on `/scanbot/joint_states`
- Published in simulator joint order.

## Notes
- The target pose is interpreted in **robot base frame**.
- This extension expects `scanbot_msgs` to be built and sourced in the ROS 2
  environment used to launch Isaac Sim.
- TargetTcp is implemented as a differential-IK loop; completion requires meeting
  tolerances for multiple consecutive updates to avoid premature success.

## Marker topic
`scanbot_msgs/msg/MarkerPoseArray` on `/scanbot/markers`

- `header.frame_id`: `"world"` or `"base"` (default `"base"` if empty; only `"world"` skips base->world conversion)
- `markers[]`: each entry contains
  - `pose7d`: `[x, y, z, qx, qy, qz, qw]`
  - `color`: `std_msgs/ColorRGBA`

Clear all markers:
`std_msgs/Empty` on `/scanbot/markers/clear`

Spawn debug cubes:
`std_msgs/Empty` on `/scanbot/markers/debug_cubes`

## Camera topics
Per camera (`<camera_name>` is the sensor name in the Isaac Lab scene):
- RGB: `sensor_msgs/msg/Image` on `/scanbot/cameras/<camera_name>/image_raw` (`rgb8`)
- Depth: `sensor_msgs/msg/Image` on `/scanbot/cameras/<camera_name>/depth_raw` (`32FC1`)
- Intrinsics: `sensor_msgs/msg/CameraInfo` on `/scanbot/cameras/<camera_name>/camera_info`
- Pose (world): `geometry_msgs/msg/PoseStamped` on `/scanbot/cameras/<camera_name>/pose_world` (`frame_id="world"`)
- Point cloud (world): `sensor_msgs/msg/PointCloud2` on `/scanbot/cameras/<camera_name>/points` (`frame_id="world"`)

Default camera alias (first detected "default"):
- `/scanbot/cameras/default/<same suffixes as above>`

Notes:
- PointCloud2 is generated from RGB+depth using intrinsics and camera pose, and is downsampled using `POINTCLOUD_STRIDE`.
- If `CAMERA_USE_COMPRESSED_TRANSPORT` is enabled, `/image_raw/compressed` is provided via `image_transport republish`.

## Teleport actions
Instantly move the robot (bypasses controller loop) using ROS 2 actions.

`scanbot_msgs/action/TeleportTcp` on `/scanbot/teleport_tcp`
- `target`: `geometry_msgs/PoseStamped`
- `pos_tolerance`, `rot_tolerance`, `timeout_sec`: optional; `header.frame_id` defaults to **base**; set to `"world"` to interpret in world frame

`scanbot_msgs/action/TeleportJoints` on `/scanbot/teleport_joint`
- `name[]` + `position[]`: set by joint names (lengths must match)
- If `name[]` is empty, `position[]` is interpreted in simulator joint order
- `tolerance`, `timeout_sec`: optional

## Reset service
Resets the environment (same behavior as keyboard teleop `R`).

`std_srvs/srv/Trigger` on `/scanbot/reset_env`

## Joint limits service
Query joint position limits from the simulator.

`scanbot_msgs/srv/GetJointLimits` on `/scanbot/get_joint_limits`

## Example (ROS 2)
```bash
# Build and source the interface package
cd scanbot/ros2
colcon build --packages-select scanbot_msgs
source install/setup.bash

# Send an action goal
ros2 action send_goal /scanbot/target_tcp scanbot_msgs/action/TargetTcp "{target: {header: {frame_id: 'base'}, pose: {position: {x: 0.4, y: 0.0, z: 0.2}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}, pos_tolerance: 0.005, rot_tolerance: 0.02, timeout_sec: 10.0}"

# Or use the helper (7D pose vector)
python scanbot/scratch/utils/send_target_tcp.py --pose "0.4,0,0.2,0,0,0,1"

# Teleport TCP (base frame)
ros2 action send_goal /scanbot/teleport_tcp scanbot_msgs/action/TeleportTcp "{target: {header: {frame_id: 'base'}, pose: {position: {x: 0.4, y: 0.0, z: 0.2}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}, pos_tolerance: 0.005, rot_tolerance: 0.02, timeout_sec: 10.0}"

# Teleport joints (by name)
ros2 action send_goal /scanbot/teleport_joint scanbot_msgs/action/TeleportJoints "{name: ['joint1','joint2'], position: [0.1, -0.2], tolerance: 0.001, timeout_sec: 10.0}"

# Reset environment
ros2 service call /scanbot/reset_env std_srvs/srv/Trigger "{}"

# Query joint limits
ros2 service call /scanbot/get_joint_limits scanbot_msgs/srv/GetJointLimits "{}"

# Echo joint state once
ros2 topic echo --once /scanbot/joint_states sensor_msgs/msg/JointState

```
