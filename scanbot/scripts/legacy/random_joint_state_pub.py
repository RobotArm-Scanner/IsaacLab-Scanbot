# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Publish random JointState messages for testing ROS 2 connectivity."""

import argparse
from typing import List

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from rclpy.qos import qos_profile_sensor_data


class RandomJointStatePublisher(Node):
    """Simple ROS 2 node that publishes random joint states."""

    def __init__(self, joint_names: List[str], hz: float, amplitude: float):
        super().__init__("random_joint_state_publisher")
        self.publisher = self.create_publisher(JointState, "/joint_states", qos_profile_sensor_data)
        self.joint_names = joint_names
        self.amplitude = float(amplitude)
        self.timer = self.create_timer(1.0 / hz, self._on_timer)
        self.get_logger().info(
            f"Publishing /joint_states at {hz} Hz for joints: {', '.join(self.joint_names)} (amp={self.amplitude})"
        )

    def _on_timer(self) -> None:
        """Publish one random JointState message."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        # random positions in [-amplitude, +amplitude]
        msg.position = (np.random.rand(len(self.joint_names)) * 2.0 - 1.0) * self.amplitude
        self.publisher.publish(msg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish random JointState messages.")
    parser.add_argument(
        "--joints",
        nargs="+",
        default=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        help="Joint names to publish.",
    )
    parser.add_argument("--hz", type=float, default=20.0, help="Publish rate in Hz.")
    parser.add_argument(
        "--amplitude",
        type=float,
        default=1.0,
        help="Maximum absolute joint position (radians) for random values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = RandomJointStatePublisher(args.joints, args.hz, args.amplitude)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
