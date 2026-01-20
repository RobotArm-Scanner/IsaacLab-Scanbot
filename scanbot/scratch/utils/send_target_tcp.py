#!/usr/bin/env python3
"""Send a scanbot_msgs/TargetTcp action goal to the simulator."""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Optional


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send TargetTcp action to Scanbot simulator")
    parser.add_argument("--x", type=float, default=0.4, help="Target X (meters)")
    parser.add_argument("--y", type=float, default=0.0, help="Target Y (meters)")
    parser.add_argument("--z", type=float, default=0.2, help="Target Z (meters)")
    parser.add_argument("--qx", type=float, default=0.0, help="Quaternion X")
    parser.add_argument("--qy", type=float, default=0.0, help="Quaternion Y")
    parser.add_argument("--qz", type=float, default=0.0, help="Quaternion Z")
    parser.add_argument("--qw", type=float, default=1.0, help="Quaternion W")
    parser.add_argument(
        "--pose",
        type=str,
        default=None,
        help="7D pose vector 'x,y,z,qx,qy,qz,qw' (overrides x/y/z/qx/qy/qz/qw)",
    )
    parser.add_argument("--frame-id", type=str, default="base", help="Frame ID for PoseStamped")
    parser.add_argument("--pos-tol", type=float, default=0.005, help="Position tolerance")
    parser.add_argument("--rot-tol", type=float, default=0.02, help="Rotation tolerance")
    parser.add_argument("--timeout-sec", type=float, default=10.0, help="Goal timeout (seconds)")
    parser.add_argument("--action-name", type=str, default="/scanbot/target_tcp", help="Action name")
    parser.add_argument("--node-name", type=str, default="scanbot_target_tcp_client", help="ROS node name")
    parser.add_argument("--wait-for-server", type=float, default=5.0, help="Seconds to wait for action server")
    parser.add_argument(
        "--result-timeout",
        type=float,
        default=0.0,
        help="Seconds to wait for result (0 = wait forever)",
    )
    parser.add_argument("--feedback", action="store_true", help="Print feedback while executing")
    parser.add_argument("--domain-id", type=int, default=None, help="Override ROS_DOMAIN_ID")
    return parser.parse_args()


def _parse_pose_vec(pose_text: str) -> list[float]:
    cleaned = pose_text.strip().strip("[]()")
    parts = [p for p in re.split(r"[\\s,]+", cleaned) if p]
    if len(parts) != 7:
        raise ValueError(f"Expected 7 values, got {len(parts)}: {parts}")
    return [float(p) for p in parts]


def _spin_until(node, future, timeout_sec: Optional[float]) -> bool:
    import rclpy

    if timeout_sec is None or timeout_sec <= 0:
        rclpy.spin_until_future_complete(node, future)
        return future.done()
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
    return future.done()


def main() -> int:
    args = _parse_args()
    if args.domain_id is not None:
        os.environ["ROS_DOMAIN_ID"] = str(args.domain_id)

    try:
        import rclpy
        from rclpy.action import ActionClient
        from geometry_msgs.msg import PoseStamped
        from scanbot_msgs.action import TargetTcp
    except Exception as exc:
        print(f"[ERROR] ROS2 imports failed: {exc}", file=sys.stderr)
        return 2

    rclpy.init()
    node = rclpy.create_node(args.node_name)

    def feedback_cb(msg) -> None:
        fb = msg.feedback
        node.get_logger().info(
            f"feedback pos_error={fb.pos_error:.4f} rot_error={fb.rot_error:.4f}"
        )

    client = ActionClient(node, TargetTcp, args.action_name)
    if not client.wait_for_server(timeout_sec=args.wait_for_server):
        node.get_logger().error("Action server not available")
        rclpy.shutdown()
        return 3

    goal = TargetTcp.Goal()
    goal.target = PoseStamped()
    goal.target.header.frame_id = args.frame_id
    if args.pose:
        try:
            vec = _parse_pose_vec(args.pose)
        except Exception as exc:
            node.get_logger().error(f"Invalid --pose: {exc}")
            rclpy.shutdown()
            return 8
        x, y, z, qx, qy, qz, qw = vec
    else:
        x, y, z, qx, qy, qz, qw = args.x, args.y, args.z, args.qx, args.qy, args.qz, args.qw

    goal.target.pose.position.x = x
    goal.target.pose.position.y = y
    goal.target.pose.position.z = z
    goal.target.pose.orientation.x = qx
    goal.target.pose.orientation.y = qy
    goal.target.pose.orientation.z = qz
    goal.target.pose.orientation.w = qw
    goal.pos_tolerance = float(args.pos_tol)
    goal.rot_tolerance = float(args.rot_tol)
    goal.timeout_sec = float(args.timeout_sec)

    send_future = client.send_goal_async(goal, feedback_callback=feedback_cb if args.feedback else None)
    if not _spin_until(node, send_future, args.wait_for_server):
        node.get_logger().error("Timed out waiting for goal response")
        rclpy.shutdown()
        return 4

    goal_handle = send_future.result()
    if goal_handle is None or not goal_handle.accepted:
        node.get_logger().error("Goal rejected")
        rclpy.shutdown()
        return 5

    result_future = goal_handle.get_result_async()
    if not _spin_until(node, result_future, args.result_timeout):
        node.get_logger().error("Timed out waiting for result")
        rclpy.shutdown()
        return 6

    result = result_future.result().result
    if result.success:
        node.get_logger().info(f"Success: {result.message}")
    else:
        node.get_logger().error(f"Failed: {result.message}")

    rclpy.shutdown()
    return 0 if result.success else 7


if __name__ == "__main__":
    raise SystemExit(main())
