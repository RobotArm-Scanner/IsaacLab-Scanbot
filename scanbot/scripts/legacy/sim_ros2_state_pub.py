# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drive sim with random actions and publish joints + camera images to /sim/* topics."""

import argparse
import os
from typing import Dict

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Publish sim joint states and camera images to /sim/* topics.")
parser.add_argument("--task", type=str, default="Scanbot-Piper-Scene-v0", help="Gym task id to load.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs.")
parser.add_argument("--max_steps", type=int, default=0, help="0 = until window closed.")
parser.add_argument(
    "--action_scale",
    type=float,
    default=0.05,
    help="Stddev for random actions sent to env.",
)
parser.add_argument(
    "--joint_topic",
    type=str,
    default="/sim/joint_states",
    help="JointState publish topic.",
)
parser.add_argument(
    "--camera_prefix",
    type=str,
    default="/sim/cameras",
    help="Prefix for camera topics (topic will be <prefix>/<name>/image_raw).",
)
parser.add_argument(
    "--camera_decimation",
    type=int,
    default=1,
    help="Publish cameras every N steps (1 = every step).",
)
parser.add_argument(
    "--camera_stride",
    type=int,
    default=1,
    help="Pixel stride for downsampling camera images (1 = no downsample).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ROS defaults if not provided
os.environ.setdefault("ROS_DOMAIN_ID", "0")
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "0")

# Launch app
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

# Task registration

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState, Image

try:
    from isaaclab.sensors import Camera
except Exception:
    Camera = None


def to_image_msg(clock, frame_id: str, rgb_tensor: torch.Tensor, stride: int = 1) -> Image:
    """Convert a torch RGB(A) tensor to sensor_msgs/Image."""
    img = Image()
    img.header.stamp = clock.now().to_msg()
    img.header.frame_id = frame_id

    # Expect shape (H, W, C) on CPU.
    if rgb_tensor.is_cuda:
        rgb_tensor = rgb_tensor.cpu()
    arr = rgb_tensor
    if arr.dim() == 4:
        # pick first env if batched
        arr = arr[0]
    if stride > 1:
        arr = arr[::stride, ::stride, ...]
    if arr.dtype != torch.uint8:
        arr = (arr.clamp(0, 1) * 255).to(dtype=torch.uint8)
    h, w, c = arr.shape
    img.height = h
    img.width = w
    img.step = w * min(c, 3)
    if c == 4:
        arr = arr[..., :3]
    img.encoding = "rgb8"
    img.data = arr.contiguous().numpy().tobytes()
    return img


def main() -> None:
    rclpy.init()
    qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
    node = rclpy.create_node("sim_ros2_state_pub")
    joint_pub = node.create_publisher(JointState, args_cli.joint_topic, qos)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    env_cfg.terminations.time_out = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    joint_names = list(robot.joint_names)

    # Collect cameras (if any)
    cameras: Dict[str, object] = {}
    if hasattr(env.scene, "sensors"):
        for name, sensor in env.scene.sensors.items():
            if Camera is not None and isinstance(sensor, Camera):
                cameras[name] = sensor
    cam_pubs: Dict[str, any] = {
        name: node.create_publisher(Image, f"{args_cli.camera_prefix}/{name}/image_raw", qos) for name in cameras
    }
    if cameras:
        node.get_logger().info(f"Publishing cameras: {', '.join(cameras.keys())}")
    else:
        node.get_logger().warn("No cameras found in scene; only joints will be published.")

    scale = float(args_cli.action_scale)
    step_count = 0

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                actions = torch.randn((env.num_envs, env.action_manager.total_action_dim), device=env.device) * scale
                env.step(actions)

            # Publish joints from env 0
            jp = robot.data.joint_pos[0].detach().cpu().numpy()
            try:
                jv = robot.data.joint_vel[0].detach().cpu().numpy()
            except Exception:
                jv = None
            msg = JointState()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.header.frame_id = "sim"
            msg.name = joint_names
            msg.position = jp.tolist()
            if jv is not None and len(jv) == len(joint_names):
                msg.velocity = jv.tolist()
            msg.effort = [0.0] * len(joint_names)
            joint_pub.publish(msg)

            # Publish cameras with decimation
            if cameras and (step_count % max(1, args_cli.camera_decimation) == 0):
                for name, cam in cameras.items():
                    try:
                        cam_data = cam.data.output.get("rgb") if hasattr(cam, "data") else None
                        if cam_data is None:
                            continue
                        img_msg = to_image_msg(
                            node.get_clock(), frame_id=name, rgb_tensor=cam_data, stride=max(1, args_cli.camera_stride)
                        )
                        cam_pubs[name].publish(img_msg)
                    except Exception as exc:
                        node.get_logger().warn(f"Camera '{name}' publish failed: {exc}")

            rclpy.spin_once(node, timeout_sec=0.0)

            step_count += 1
            if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                break
    finally:
        env.close()
        node.destroy_node()
        rclpy.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()
