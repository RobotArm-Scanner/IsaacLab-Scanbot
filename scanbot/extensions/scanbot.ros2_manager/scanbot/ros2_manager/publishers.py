"""ROS2 topic publishers for scanbot.ros2_manager."""

from __future__ import annotations

import time
import importlib

import torch

from . import config as _cfg
from scanbot.scripts.utilities import pos_util as _pos_util

# Force reload so hot-reloads of scanbot.common are reflected in this extension.
pos_util = importlib.reload(_pos_util)


TCP_POSE_TOPIC = _cfg.TCP_POSE_TOPIC
TCP_PUB_HZ = _cfg.TCP_PUB_HZ
SCANPOINT_POSE_TOPIC = _cfg.SCANPOINT_POSE_TOPIC
SCANPOINT_PUB_HZ = _cfg.SCANPOINT_PUB_HZ
JOINT_STATE_TOPIC = _cfg.JOINT_STATE_TOPIC
JOINT_PUB_HZ = _cfg.JOINT_PUB_HZ


class TcpPosePublisher:
    def __init__(self, node, pose_stamped_type) -> None:
        self._node = node
        self._pose_stamped_type = pose_stamped_type
        self._pub = None
        self._next_pub = 0.0

        self._pub = self._node.create_publisher(self._pose_stamped_type, TCP_POSE_TOPIC, 10)

    def shutdown(self) -> None:
        self._node.destroy_publisher(self._pub)
        self._pub = None

    def maybe_publish(self, curr_pos: torch.Tensor, curr_quat: torch.Tensor, frame_id: str = "base") -> None:
        if self._node is None or self._pub is None:
            return
        if not self._pub_has_subscribers(self._pub):
            return

        now = time.monotonic()
        if now < self._next_pub:
            return
        self._next_pub = now + (1.0 / max(TCP_PUB_HZ, 1.0))

        msg = self._pose_stamped_type()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        pos = curr_pos[0].detach().cpu().numpy().tolist()
        quat = curr_quat[0].detach().cpu().numpy().tolist()
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.w = float(quat[0])
        msg.pose.orientation.x = float(quat[1])
        msg.pose.orientation.y = float(quat[2])
        msg.pose.orientation.z = float(quat[3])
        self._pub.publish(msg)

    @staticmethod
    def _pub_has_subscribers(pub) -> bool:
        return pub.get_subscription_count() > 0


class ScanpointPosePublisher:
    def __init__(self, node, pose_stamped_type) -> None:
        self._node = node
        self._pose_stamped_type = pose_stamped_type
        self._pub = None
        self._next_pub = 0.0

        self._pub = self._node.create_publisher(self._pose_stamped_type, SCANPOINT_POSE_TOPIC, 10)

    def shutdown(self) -> None:
        self._node.destroy_publisher(self._pub)
        self._pub = None

    def maybe_publish(self, tcp_pos: torch.Tensor, tcp_quat: torch.Tensor) -> None:
        if self._node is None or self._pub is None:
            return

        now = time.monotonic()
        if now < self._next_pub:
            return
        self._next_pub = now + (1.0 / max(SCANPOINT_PUB_HZ, 1.0))

        msg = self._pose_stamped_type()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        pos = tcp_pos[0].detach().cpu().numpy().tolist()
        quat = tcp_quat[0].detach().cpu().numpy().tolist()  # wxyz
        scan_pos, scan_quat = pos_util.tcp_to_scanpoint(pos, quat)
        msg.header.frame_id = "base"
        msg.pose.position.x = float(scan_pos[0])
        msg.pose.position.y = float(scan_pos[1])
        msg.pose.position.z = float(scan_pos[2])
        msg.pose.orientation.w = float(scan_quat[0])
        msg.pose.orientation.x = float(scan_quat[1])
        msg.pose.orientation.y = float(scan_quat[2])
        msg.pose.orientation.z = float(scan_quat[3])
        self._pub.publish(msg)




class JointStatePublisher:
    def __init__(self, node, joint_state_type) -> None:
        self._node = node
        self._joint_state_type = joint_state_type
        self._pub = None
        self._next_pub = 0.0

        self._pub = self._node.create_publisher(self._joint_state_type, JOINT_STATE_TOPIC, 10)

    def shutdown(self) -> None:
        self._node.destroy_publisher(self._pub)
        self._pub = None

    def maybe_publish(self, robot) -> None:
        if self._node is None or self._pub is None:
            return
        if not self._pub_has_subscribers(self._pub):
            return

        now = time.monotonic()
        if now < self._next_pub:
            return
        self._next_pub = now + (1.0 / max(JOINT_PUB_HZ, 1.0))

        joint_names = list(robot.joint_names)
        pos = robot.data.joint_pos[0].detach().cpu().numpy().tolist()
        vel = robot.data.joint_vel[0].detach().cpu().numpy().tolist()

        msg = self._joint_state_type()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.name = [str(n) for n in joint_names]
        msg.position = [float(v) for v in pos]
        msg.velocity = [float(v) for v in vel]
        msg.effort = []
        self._pub.publish(msg)

    @staticmethod
    def _pub_has_subscribers(pub) -> bool:
        return pub.get_subscription_count() > 0
