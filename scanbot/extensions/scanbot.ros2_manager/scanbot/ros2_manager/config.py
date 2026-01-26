"""Shared configuration constants for scanbot.ros2_manager."""

from __future__ import annotations

import os

TOPIC_PREFIX = "/scanbot"

ACTION_NAME = f"{TOPIC_PREFIX}/target_tcp"
TCP_POSE_TOPIC = f"{TOPIC_PREFIX}/tcp_pose"
SCANPOINT_POSE_TOPIC = f"{TOPIC_PREFIX}/sp_pose"
JOINT_STATE_TOPIC = f"{TOPIC_PREFIX}/joint_states"
TELEPORT_TCP_ACTION = f"{TOPIC_PREFIX}/teleport_tcp"
TELEPORT_JOINT_ACTION = f"{TOPIC_PREFIX}/teleport_joint"
TELEPORT_TCP_SERVICE = f"{TOPIC_PREFIX}/teleport_tcp"
TELEPORT_JOINT_SERVICE = f"{TOPIC_PREFIX}/teleport_joint"
RESET_ENV_SERVICE = f"{TOPIC_PREFIX}/reset_env"
GET_JOINT_LIMITS_SERVICE = f"{TOPIC_PREFIX}/get_joint_limits"

CAMERA_TOPIC_PREFIX = f"{TOPIC_PREFIX}/cameras"
DEFAULT_CAMERA_ALIASES = ("default_camera", "global_camera", "camera", "main_camera")
CAMERA_POINTCLOUD_SUFFIX = "points"

MARKER_POSE_TOPIC = f"{TOPIC_PREFIX}/markers"
MARKER_CLEAR_TOPIC = f"{TOPIC_PREFIX}/markers/clear"
MARKER_DEBUG_CUBES_TOPIC = f"{TOPIC_PREFIX}/markers/debug_cubes"
MARKER_FORWARD_AXIS = (0.0, 0.0, 1.0)
MARKER_INVERT_DIRECTION = False

DEFAULT_POS_TOL = 0.005
DEFAULT_ROT_TOL = 0.02
DEFAULT_TIMEOUT_SEC = 60.0
MAX_POS_STEP = 0.15
MAX_ROT_STEP = 0.5
GAIN_POS = 5.0
GAIN_ROT = 2.0
POS_ONLY_GATE = 0.08
TARGET_TCP_STABLE_STEPS = 5

TCP_PUB_HZ = 10.0
SCANPOINT_PUB_HZ = TCP_PUB_HZ
JOINT_PUB_HZ = 10.0
# Camera topics are high-bandwidth (RGB + depth). Keeping this modest avoids starving
# other ROS2 traffic (e.g. teleport actions) when multiple cameras are enabled.
CAMERA_PUB_HZ = 2.0
CAMERA_STRIDE = 1
POINTCLOUD_STRIDE = 8

CAMERA_USE_COMPRESSED_TRANSPORT = False
CAMERA_COMPRESSED_FORMAT = "png"

# Pinocchio URDF path for teleport IK (defaults to Piper no-gripper model).
PIPER_URDF_PATH = os.path.join(
    os.environ.get("ISAACLAB_PATH", "/workspace/isaaclab"),
    "scanbot",
    "resources",
    "piper_isaac_sim",
    "piper_description",
    "urdf",
    "piper_no_gripper_description.urdf",
)
