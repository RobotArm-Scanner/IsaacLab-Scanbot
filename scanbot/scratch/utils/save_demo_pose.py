#!/usr/bin/env python3
"""Capture current /scanbot/tcp_pose and save into demo slot JSON."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save current /scanbot/tcp_pose into a demo slot")
    parser.add_argument("--slot", type=int, default=1, help="Demo slot number")
    parser.add_argument("--topic", type=str, default="/scanbot/tcp_pose", help="PoseStamped topic name")
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "demo_poses.json"),
        help="Output JSON file",
    )
    parser.add_argument("--timeout", type=float, default=5.0, help="Seconds to wait for pose")
    return parser.parse_args()


def _load_json(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> int:
    args = _parse_args()

    try:
        import rclpy
        from rclpy.qos import QoSProfile
        from geometry_msgs.msg import PoseStamped
    except Exception as exc:
        print(f"[ERROR] ROS2 imports failed: {exc}")
        return 2

    rclpy.init()
    node = rclpy.create_node(f"scanbot_demo_pose_save_{args.slot}")

    msg_holder = {}

    def _cb(msg: PoseStamped) -> None:
        msg_holder["msg"] = msg

    qos = QoSProfile(depth=1)
    sub = node.create_subscription(PoseStamped, args.topic, _cb, qos)

    start = time.time()
    while rclpy.ok() and "msg" not in msg_holder:
        rclpy.spin_once(node, timeout_sec=0.1)
        if args.timeout > 0 and (time.time() - start) > args.timeout:
            break

    sub.destroy()
    node.destroy_node()
    rclpy.shutdown()

    if "msg" not in msg_holder:
        print(f"[ERROR] No pose received on {args.topic} within {args.timeout}s")
        return 3

    msg = msg_holder["msg"]
    pose = {
        "frame_id": msg.header.frame_id,
        "stamp": {"sec": int(msg.header.stamp.sec), "nanosec": int(msg.header.stamp.nanosec)},
        "position": {
            "x": float(msg.pose.position.x),
            "y": float(msg.pose.position.y),
            "z": float(msg.pose.position.z),
        },
        "orientation": {
            "x": float(msg.pose.orientation.x),
            "y": float(msg.pose.orientation.y),
            "z": float(msg.pose.orientation.z),
            "w": float(msg.pose.orientation.w),
        },
        "pose_vec": [
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
            float(msg.pose.orientation.x),
            float(msg.pose.orientation.y),
            float(msg.pose.orientation.z),
            float(msg.pose.orientation.w),
        ],
        "saved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": args.topic,
    }

    data = _load_json(args.out)
    data[str(args.slot)] = pose
    _save_json(args.out, data)

    print(f"[OK] Saved slot {args.slot} -> {args.out}")
    print(json.dumps(pose, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
