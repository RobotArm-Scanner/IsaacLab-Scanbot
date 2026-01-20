#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path


def _timestamp_from_filename(path: Path) -> str:
    match = re.search(r"data_(\d{8}_\d{6}_\d{6})\.json$", path.name)
    if not match:
        raise ValueError(f"Unexpected legacy json name: {path.name}")
    return match.group(1)


def _camera_prefix(camera_name: str) -> str:
    if camera_name == "wrist_camera":
        return ""
    base = camera_name[:-7] if camera_name.endswith("_camera") else camera_name
    return f"{base}_"


@dataclass
class CameraFrame:
    rgb: object | None = None
    depth: object | None = None
    info: object | None = None
    pose_world: object | None = None
    points: object | None = None


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect a legacy-like dataset using scanbot ROS2 APIs.")
    parser.add_argument("--legacy_dir", type=str, required=True, help="Legacy dataset directory with data_*.json")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory to write the new dataset")
    parser.add_argument("--num_frames", type=int, default=10, help="How many frames to sample from legacy")
    parser.add_argument("--skip", type=int, default=0, help="How many initial legacy frames to skip")
    parser.add_argument(
        "--mode",
        choices=("teleport_joint", "teleport_tcp", "target_tcp"),
        default="teleport_joint",
        help="How to move the robot for each frame",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["wrist_camera", "global_camera"],
        help="Camera names to capture (must exist in /scanbot/cameras/<name>/...)",
    )
    parser.add_argument("--timeout_sec", type=float, default=10.0, help="Timeout per wait step")
    parser.add_argument(
        "--discard_topic_sets",
        type=int,
        default=1,
        help="How many complete topic sets to discard after each teleport (helps RTX sensors settle).",
    )
    parser.add_argument(
        "--pcd_source",
        choices=("rgbd", "pointcloud2"),
        default="pointcloud2",
        help="How to generate per-frame point clouds when --no_pcd is not set.",
    )
    parser.add_argument("--pcd_voxel", type=float, default=0.0005, help="Voxel size for per-frame PCD downsampling")
    parser.add_argument("--no_pcd", action="store_true", help="Skip PCD generation")
    args = parser.parse_args()

    legacy_dir = Path(args.legacy_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_pcd_dir = out_dir / "pcd"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pcd_dir.mkdir(parents=True, exist_ok=True)

    legacy_jsons = sorted(legacy_dir.glob("data_*.json"))
    if not legacy_jsons:
        raise SystemExit(f"No legacy frames found in: {legacy_dir}")
    legacy_jsons = legacy_jsons[args.skip : args.skip + args.num_frames]
    if not legacy_jsons:
        raise SystemExit("No frames selected (skip too large?)")

    # ROS imports must happen after sourcing ROS env + scanbot_msgs install.
    import numpy as np
    from PIL import Image as PilImage
    import rclpy
    from rclpy.action import ActionClient
    from geometry_msgs.msg import PoseStamped
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from scanbot_msgs.action import TargetTcp, TeleportJoints, TeleportTcp
    from sensor_msgs.msg import CameraInfo, Image as RosImage, JointState, PointCloud2
    from std_srvs.srv import Trigger

    class Collector(Node):
        def __init__(self) -> None:
            super().__init__("scanbot_legacy_like_collector")

            self._executor: MultiThreadedExecutor | None = None
            self._spin_thread: threading.Thread | None = None

            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)

            self.joint_state: JointState | None = None
            self.tcp_pose: PoseStamped | None = None

            self._cams: dict[str, CameraFrame] = {name: CameraFrame() for name in args.cameras}
            self._pcd_source = str(args.pcd_source)
            self._pcd_camera = "wrist_camera" if "wrist_camera" in args.cameras else (args.cameras[0] if args.cameras else "")

            self.create_subscription(JointState, "/scanbot/joint_states", self._on_joint, qos)
            self.create_subscription(PoseStamped, "/scanbot/tcp_pose", self._on_tcp, qos)

            for cam in args.cameras:
                self.create_subscription(
                    RosImage, f"/scanbot/cameras/{cam}/image_raw", self._make_cam_cb(cam, "rgb"), qos
                )
                self.create_subscription(
                    RosImage, f"/scanbot/cameras/{cam}/depth_raw", self._make_cam_cb(cam, "depth"), qos
                )
                self.create_subscription(
                    CameraInfo,
                    f"/scanbot/cameras/{cam}/camera_info",
                    self._make_cam_cb(cam, "info"),
                    qos,
                )
                self.create_subscription(
                    PoseStamped,
                    f"/scanbot/cameras/{cam}/pose_world",
                    self._make_cam_cb(cam, "pose_world"),
                    qos,
                )
                if not args.no_pcd and self._pcd_source == "pointcloud2" and cam == self._pcd_camera:
                    self.create_subscription(
                        PointCloud2,
                        f"/scanbot/cameras/{cam}/points",
                        self._make_cam_cb(cam, "points"),
                        qos,
                    )

            self._reset = self.create_client(Trigger, "/scanbot/reset_env")
            self._target_tcp = ActionClient(self, TargetTcp, "/scanbot/target_tcp")
            self._teleport_joint = ActionClient(self, TeleportJoints, "/scanbot/teleport_joint")
            self._teleport_tcp = ActionClient(self, TeleportTcp, "/scanbot/teleport_tcp")

            # Spin in a dedicated thread to avoid starving action/service responses while
            # subscribing to high-bandwidth camera topics.
            self._executor = MultiThreadedExecutor()
            self._executor.add_node(self)
            self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
            self._spin_thread.start()

        def shutdown(self) -> None:
            if self._executor is not None:
                try:
                    self._executor.shutdown()
                except Exception:
                    pass
                self._executor = None
            if self._spin_thread is not None:
                self._spin_thread.join(timeout=2.0)
                self._spin_thread = None

        def _on_joint(self, msg: JointState) -> None:
            self.joint_state = msg

        def _on_tcp(self, msg: PoseStamped) -> None:
            self.tcp_pose = msg

        def _make_cam_cb(self, cam: str, field: str):
            def _cb(msg) -> None:
                setattr(self._cams[cam], field, msg)

            return _cb

        def clear_latest(self) -> None:
            self.joint_state = None
            self.tcp_pose = None
            for cam in self._cams.values():
                cam.rgb = None
                cam.depth = None
                cam.info = None
                cam.pose_world = None
                cam.points = None

        def wait_for_services(self, timeout_sec: float) -> None:
            start = time.monotonic()
            while not self._reset.wait_for_service(timeout_sec=0.2):
                if time.monotonic() - start > timeout_sec:
                    raise TimeoutError("Service not available: /scanbot/reset_env")
            while not self._target_tcp.wait_for_server(timeout_sec=0.2):
                if time.monotonic() - start > timeout_sec:
                    raise TimeoutError("Action server not available: /scanbot/target_tcp")
            while not self._teleport_joint.wait_for_server(timeout_sec=0.2):
                if time.monotonic() - start > timeout_sec:
                    raise TimeoutError("Action server not available: /scanbot/teleport_joint")
            while not self._teleport_tcp.wait_for_server(timeout_sec=0.2):
                if time.monotonic() - start > timeout_sec:
                    raise TimeoutError("Action server not available: /scanbot/teleport_tcp")

        def call_trigger(self, client, timeout_sec: float) -> None:
            req = Trigger.Request()
            fut = client.call_async(req)
            if not self._spin_until(lambda: fut.done(), timeout_sec):
                raise TimeoutError("Trigger service timed out")
            resp = fut.result()
            if resp is None or not resp.success:
                raise RuntimeError(f"Trigger failed: {resp.message if resp else 'no response'}")

        def call_teleport_joint(self, names: list[str], positions: list[float], tol: float, timeout_sec: float) -> None:
            goal = TeleportJoints.Goal()
            goal.name = list(names)
            goal.position = [float(x) for x in positions]
            goal.tolerance = float(tol)
            goal.timeout_sec = float(timeout_sec)

            send_fut = self._teleport_joint.send_goal_async(goal)
            if not self._spin_until(lambda: send_fut.done(), timeout_sec):
                raise TimeoutError("TeleportJoints send_goal timed out")
            goal_handle = send_fut.result()
            if goal_handle is None or not goal_handle.accepted:
                raise RuntimeError("TeleportJoints goal rejected")

            result_fut = goal_handle.get_result_async()
            if not self._spin_until(lambda: result_fut.done(), timeout_sec):
                raise TimeoutError("TeleportJoints result timed out")
            wrapped = result_fut.result()
            result = wrapped.result if wrapped is not None else None
            if result is None or not result.success:
                raise RuntimeError(f"TeleportJoints failed: {result.message if result else 'no response'}")

        def call_teleport_tcp(self, target: PoseStamped, pos_tol: float, rot_tol: float, timeout_sec: float) -> None:
            goal = TeleportTcp.Goal()
            goal.target = target
            goal.pos_tolerance = float(pos_tol)
            goal.rot_tolerance = float(rot_tol)
            goal.timeout_sec = float(timeout_sec)

            send_fut = self._teleport_tcp.send_goal_async(goal)
            if not self._spin_until(lambda: send_fut.done(), timeout_sec):
                raise TimeoutError("TeleportTcp send_goal timed out")
            goal_handle = send_fut.result()
            if goal_handle is None or not goal_handle.accepted:
                raise RuntimeError("TeleportTcp goal rejected")

            result_fut = goal_handle.get_result_async()
            if not self._spin_until(lambda: result_fut.done(), timeout_sec):
                raise TimeoutError("TeleportTcp result timed out")
            wrapped = result_fut.result()
            result = wrapped.result if wrapped is not None else None
            if result is None or not result.success:
                raise RuntimeError(f"TeleportTcp failed: {result.message if result else 'no response'}")

        def call_target_tcp(self, target: PoseStamped, pos_tol: float, rot_tol: float, timeout_sec: float) -> None:
            goal = TargetTcp.Goal()
            goal.target = target
            goal.pos_tolerance = float(pos_tol)
            goal.rot_tolerance = float(rot_tol)
            goal.timeout_sec = float(timeout_sec)

            send_fut = self._target_tcp.send_goal_async(goal)
            if not self._spin_until(lambda: send_fut.done(), timeout_sec):
                raise TimeoutError("TargetTcp send_goal timed out")
            goal_handle = send_fut.result()
            if goal_handle is None or not goal_handle.accepted:
                raise RuntimeError("TargetTcp goal rejected")

            result_fut = goal_handle.get_result_async()
            if not self._spin_until(lambda: result_fut.done(), timeout_sec):
                raise TimeoutError("TargetTcp result timed out")
            wrapped = result_fut.result()
            result = wrapped.result if wrapped is not None else None
            if result is None or not result.success:
                raise RuntimeError(f"TargetTcp failed: {result.message if result else 'no response'}")

        @staticmethod
        def _stamp_tuple(stamp_msg) -> tuple[int, int]:
            try:
                return int(getattr(stamp_msg, "sec", 0)), int(getattr(stamp_msg, "nanosec", 0))
            except Exception:
                return 0, 0

        def wait_for_all_topics(self, timeout_sec: float, min_stamp=None) -> None:
            def _ready() -> bool:
                if self.joint_state is None or self.tcp_pose is None:
                    return False
                for cam in self._cams.values():
                    if cam.rgb is None or cam.depth is None or cam.info is None or cam.pose_world is None:
                        return False
                if not args.no_pcd and self._pcd_source == "pointcloud2" and self._pcd_camera:
                    pcd_cam = self._cams.get(self._pcd_camera)
                    if pcd_cam is None or pcd_cam.points is None:
                        return False
                if min_stamp is not None:
                    min_t = self._stamp_tuple(min_stamp)
                    if self._stamp_tuple(getattr(self.joint_state.header, "stamp", None)) < min_t:
                        return False
                    if self._stamp_tuple(getattr(self.tcp_pose.header, "stamp", None)) < min_t:
                        return False
                    for cam in self._cams.values():
                        if self._stamp_tuple(getattr(cam.rgb.header, "stamp", None)) < min_t:
                            return False
                        if self._stamp_tuple(getattr(cam.depth.header, "stamp", None)) < min_t:
                            return False
                        if self._stamp_tuple(getattr(cam.info.header, "stamp", None)) < min_t:
                            return False
                        if self._stamp_tuple(getattr(cam.pose_world.header, "stamp", None)) < min_t:
                            return False
                    if not args.no_pcd and self._pcd_source == "pointcloud2" and self._pcd_camera:
                        pcd_cam = self._cams.get(self._pcd_camera)
                        if pcd_cam is None:
                            return False
                        if self._stamp_tuple(getattr(pcd_cam.points.header, "stamp", None)) < min_t:
                            return False
                return True

            if not self._spin_until(_ready, timeout_sec):
                missing = []
                if self.joint_state is None:
                    missing.append("joint_state")
                if self.tcp_pose is None:
                    missing.append("tcp_pose")
                for name, cam in self._cams.items():
                    for field in ("rgb", "depth", "info", "pose_world"):
                        if getattr(cam, field) is None:
                            missing.append(f"{name}.{field}")
                if not args.no_pcd and self._pcd_source == "pointcloud2" and self._pcd_camera:
                    if self._cams.get(self._pcd_camera) is None or self._cams[self._pcd_camera].points is None:
                        missing.append(f"{self._pcd_camera}.points")
                raise TimeoutError(f"Timed out waiting for: {', '.join(missing)}")

        def wait_for_joint_target(self, names: list[str], positions: list[float], tol: float, timeout_sec: float) -> None:
            """Wait until /scanbot/joint_states matches the target within tol (max abs)."""

            def _reached() -> bool:
                if self.joint_state is None:
                    return False
                js = self.joint_state
                if names:
                    if len(names) != len(positions):
                        return False
                    name_to_idx = {n: i for i, n in enumerate(js.name)}
                    errs: list[float] = []
                    for name, target in zip(names, positions):
                        idx = name_to_idx.get(str(name))
                        if idx is None:
                            return False
                        if idx >= len(js.position):
                            return False
                        errs.append(abs(float(js.position[idx]) - float(target)))
                    return bool(errs) and max(errs) <= tol

                # unnamed: assume simulator joint order matches js.position order
                if len(js.position) != len(positions):
                    return False
                errs = [abs(float(a) - float(b)) for a, b in zip(js.position, positions)]
                return bool(errs) and max(errs) <= tol

            if not self._spin_until(_reached, timeout_sec):
                raise TimeoutError("Timed out waiting for joint target to apply")

        def wait_for_tcp_target(
            self,
            target_pos: list[float],
            target_quat_wxyz: list[float],
            pos_tol: float,
            rot_tol_rad: float,
            timeout_sec: float,
        ) -> None:
            """Wait until /scanbot/tcp_pose is within tolerance of the target (base frame)."""

            def _quat_angle(q0_wxyz: list[float], q1_wxyz: list[float]) -> float:
                dot = abs(
                    float(q0_wxyz[0]) * float(q1_wxyz[0])
                    + float(q0_wxyz[1]) * float(q1_wxyz[1])
                    + float(q0_wxyz[2]) * float(q1_wxyz[2])
                    + float(q0_wxyz[3]) * float(q1_wxyz[3])
                )
                dot = max(0.0, min(1.0, dot))
                return 2.0 * float(np.arccos(dot))

            def _reached() -> bool:
                if self.tcp_pose is None:
                    return False
                pose = self.tcp_pose.pose
                pos = [float(pose.position.x), float(pose.position.y), float(pose.position.z)]
                quat = [float(pose.orientation.w), float(pose.orientation.x), float(pose.orientation.y), float(pose.orientation.z)]
                pos_err = float(np.linalg.norm(np.array(pos, dtype=float) - np.array(target_pos, dtype=float)))
                rot_err = _quat_angle(quat, target_quat_wxyz)
                return pos_err <= pos_tol and rot_err <= rot_tol_rad

            if not self._spin_until(_reached, timeout_sec):
                raise TimeoutError("Timed out waiting for TCP target to apply")

        def _spin_until(self, condition, timeout_sec: float) -> bool:
            end = time.monotonic() + timeout_sec
            while time.monotonic() < end:
                if condition():
                    return True
                time.sleep(0.01)
            return False

    rclpy.init()
    # Reset before subscribing to high-bandwidth topics to avoid starving the service response.
    reset_node = rclpy.create_node("scanbot_reset_client")
    try:
        reset_client = reset_node.create_client(Trigger, "/scanbot/reset_env")
        start = time.monotonic()
        while not reset_client.wait_for_service(timeout_sec=0.2):
            if time.monotonic() - start > args.timeout_sec:
                raise TimeoutError("Service not available: /scanbot/reset_env")
        req = Trigger.Request()
        fut = reset_client.call_async(req)
        deadline = time.monotonic() + args.timeout_sec
        while time.monotonic() < deadline and not fut.done():
            rclpy.spin_once(reset_node, timeout_sec=0.1)
        resp = fut.result() if fut.done() else None
        if resp is None or not resp.success:
            raise RuntimeError(f"reset_env failed: {resp.message if resp else 'timeout'}")
    finally:
        reset_node.destroy_node()

    node = Collector()
    try:
        node.wait_for_services(timeout_sec=args.timeout_sec)
        # reset_env is executed as a queued hook inside the simulator loop, so the service response
        # does not guarantee the reset has completed. Give the sim a moment to run the hook and
        # publish fresh sensor state before starting the first capture.
        time.sleep(2.0)
        for _ in range(max(0, int(args.discard_topic_sets)) + 1):
            min_stamp = node.get_clock().now().to_msg()
            node.clear_latest()
            node.wait_for_all_topics(timeout_sec=args.timeout_sec, min_stamp=min_stamp)

        for idx, legacy_json in enumerate(legacy_jsons, start=1):
            timestamp = _timestamp_from_filename(legacy_json)
            node.get_logger().info(f"[{idx}/{len(legacy_jsons)}] Capturing timestamp={timestamp} ({args.mode})")

            legacy_meta = json.loads(legacy_json.read_text())

            if args.mode == "teleport_joint":
                names = legacy_meta.get("joint_names", [])
                positions = legacy_meta.get("joint_positions", [])
                node.call_teleport_joint(names, positions, tol=1e-3, timeout_sec=args.timeout_sec)
                # Give rendering a moment to catch up (teleport can take a few frames to propagate to RTX sensors).
                time.sleep(1.0)
            else:
                tcp_pos = legacy_meta.get("tcp_position_base")
                tcp_q_wxyz = legacy_meta.get("tcp_orientation_base_wxyz")
                if tcp_pos is None or tcp_q_wxyz is None:
                    raise RuntimeError(f"Legacy frame missing tcp fields: {legacy_json.name}")
                target = PoseStamped()
                target.header.frame_id = "base"
                target.pose.position.x = float(tcp_pos[0])
                target.pose.position.y = float(tcp_pos[1])
                target.pose.position.z = float(tcp_pos[2])
                target.pose.orientation.w = float(tcp_q_wxyz[0])
                target.pose.orientation.x = float(tcp_q_wxyz[1])
                target.pose.orientation.y = float(tcp_q_wxyz[2])
                target.pose.orientation.z = float(tcp_q_wxyz[3])
                # Use slightly looser tolerances than joint-teleport to avoid false timeouts due to small
                # residual TCP error after IK + sim writeback.
                if args.mode == "teleport_tcp":
                    node.call_teleport_tcp(target, pos_tol=5e-3, rot_tol=2e-2, timeout_sec=args.timeout_sec)
                else:
                    node.call_target_tcp(target, pos_tol=5e-3, rot_tol=2e-2, timeout_sec=args.timeout_sec)
                time.sleep(1.0)

            # Wait for fresh data. RTX sensors can lag by ~1 publish interval after a teleport, so discard a
            # configurable number of complete topic sets before saving.
            for _ in range(max(0, int(args.discard_topic_sets)) + 1):
                min_stamp = node.get_clock().now().to_msg()
                node.clear_latest()
                node.wait_for_all_topics(timeout_sec=args.timeout_sec, min_stamp=min_stamp)

            # Save files and metadata.
            meta_out: dict[str, object] = {
                "timestamp": timestamp,
                "joint_names": list(node.joint_state.name),
                "joint_positions": [float(x) for x in node.joint_state.position],
            }
            if node.tcp_pose is not None:
                p = node.tcp_pose.pose.position
                q = node.tcp_pose.pose.orientation
                meta_out["tcp_position_base"] = [float(p.x), float(p.y), float(p.z)]
                meta_out["tcp_orientation_base_wxyz"] = [float(q.w), float(q.x), float(q.y), float(q.z)]

            cameras_meta: dict[str, object] = {}
            for cam_name in args.cameras:
                cam = node._cams[cam_name]
                file_prefix = _camera_prefix(cam_name)

                # RGB
                rgb_msg = cam.rgb
                if rgb_msg is None:
                    raise RuntimeError(f"Missing RGB for camera: {cam_name}")
                rgb_np = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(rgb_msg.height, rgb_msg.width, 3)
                rgb_path = out_dir / f"{file_prefix}rgb_{timestamp}.png"
                PilImage.fromarray(rgb_np).save(rgb_path)

                # Depth
                depth_msg = cam.depth
                if depth_msg is None:
                    raise RuntimeError(f"Missing depth for camera: {cam_name}")
                depth_np = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(depth_msg.height, depth_msg.width)
                depth_raw_path = out_dir / f"{file_prefix}depth_raw_{timestamp}.npy"
                np.save(depth_raw_path, depth_np)

                # Visual depth (legacy-style min/max normalization, ignoring inf)
                finite = np.isfinite(depth_np)
                if finite.any():
                    dmin = float(depth_np[finite].min())
                    dmax = float(depth_np[finite].max())
                else:
                    dmin, dmax = 0.0, 0.0
                if dmax > dmin:
                    depth_vis = (np.clip(depth_np, dmin, dmax) - dmin) / (dmax - dmin) * 255.0
                else:
                    depth_vis = np.zeros_like(depth_np)
                depth_vis_path = out_dir / f"{file_prefix}depth_visual_{timestamp}.png"
                PilImage.fromarray(depth_vis.astype(np.uint8)).save(depth_vis_path)

                # Camera pose + intrinsics
                pose_msg = cam.pose_world
                info_msg = cam.info
                if pose_msg is None or info_msg is None:
                    raise RuntimeError(f"Missing pose/info for camera: {cam_name}")
                cp = pose_msg.pose.position
                cq = pose_msg.pose.orientation
                k = list(info_msg.k)
                cam_intr = [k[0:3], k[3:6], k[6:9]] if len(k) == 9 else []

                cam_meta = {
                    "rgb_image_path": rgb_path.name,
                    "depth_raw_path": depth_raw_path.name,
                    "depth_visual_path": depth_vis_path.name,
                    "camera_position_world": [float(cp.x), float(cp.y), float(cp.z)],
                    "camera_orientation_world_ros": [float(cq.w), float(cq.x), float(cq.y), float(cq.z)],
                    "camera_intrinsics": cam_intr,
                }
                cameras_meta[cam_name] = cam_meta

                # Mirror legacy top-level fields for wrist_camera.
                if cam_name == "wrist_camera":
                    meta_out.update(
                        {
                            "rgb_image_path": rgb_path.name,
                            "depth_raw_path": depth_raw_path.name,
                            "depth_visual_path": depth_vis_path.name,
                            "camera_position_world": cam_meta["camera_position_world"],
                            "camera_orientation_world_ros": cam_meta["camera_orientation_world_ros"],
                            "camera_intrinsics": cam_intr,
                        }
                    )

            meta_out["cameras"] = cameras_meta

            out_json = out_dir / f"data_{timestamp}.json"
            out_json.write_text(json.dumps(meta_out, indent=2))

            if not args.no_pcd:
                try:
                    import open3d as o3d

                    ply_path = out_pcd_dir / f"data_{timestamp}_pcd.ply"
                    if args.pcd_source == "rgbd":
                        from scipy.spatial.transform import Rotation

                        wrist = cameras_meta.get("wrist_camera")
                        if not isinstance(wrist, dict):
                            raise RuntimeError("Missing wrist_camera metadata for rgbd PCD")

                        rgb_path = out_dir / wrist["rgb_image_path"]
                        depth_path = out_dir / wrist["depth_raw_path"]
                        color_raw = o3d.io.read_image(str(rgb_path))
                        depth_raw = o3d.geometry.Image(np.load(depth_path))
                        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                            color_raw,
                            depth_raw,
                            depth_scale=1.0,
                            convert_rgb_to_intensity=False,
                        )
                        k = np.array(wrist["camera_intrinsics"], dtype=float)
                        color_np = np.asarray(color_raw)
                        height, width, _ = color_np.shape
                        fx, fy = float(k[0, 0]), float(k[1, 1])
                        cx, cy = float(k[0, 2]), float(k[1, 2])
                        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
                        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

                        pos = np.array(wrist["camera_position_world"], dtype=float)
                        quat_wxyz = wrist["camera_orientation_world_ros"]
                        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=float)
                        T = np.eye(4)
                        T[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
                        T[:3, 3] = pos
                        pcd.transform(T)
                    else:
                        from sensor_msgs.msg import PointField

                        if not node._pcd_camera:
                            raise RuntimeError("pcd_source=pointcloud2 but no camera selected")
                        cam = node._cams.get(node._pcd_camera)
                        if cam is None or cam.points is None:
                            raise RuntimeError(f"Missing PointCloud2 for camera: {node._pcd_camera}")

                        msg = cam.points
                        n = int(msg.width) * int(msg.height)
                        if n <= 0:
                            pcd = o3d.geometry.PointCloud()
                        else:
                            dtype_map = {
                                PointField.INT8: np.int8,
                                PointField.UINT8: np.uint8,
                                PointField.INT16: np.int16,
                                PointField.UINT16: np.uint16,
                                PointField.INT32: np.int32,
                                PointField.UINT32: np.uint32,
                                PointField.FLOAT32: np.float32,
                                PointField.FLOAT64: np.float64,
                            }
                            names = []
                            formats = []
                            offsets = []
                            for f in msg.fields:
                                if f.name not in ("x", "y", "z", "rgb", "r", "g", "b"):
                                    continue
                                np_t = dtype_map.get(int(f.datatype))
                                if np_t is None:
                                    continue
                                names.append(f.name)
                                formats.append(np_t)
                                offsets.append(int(f.offset))
                            dt = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": int(msg.point_step)})
                            cloud = np.frombuffer(msg.data, dtype=dt, count=n)

                            pts = np.stack(
                                [cloud["x"].astype(np.float64), cloud["y"].astype(np.float64), cloud["z"].astype(np.float64)],
                                axis=1,
                            )
                            if "rgb" in cloud.dtype.names:
                                rgb = cloud["rgb"]
                                if rgb.dtype == np.float32:
                                    rgb = rgb.view(np.uint32)
                                rgb = rgb.astype(np.uint32, copy=False)
                                r = ((rgb >> 16) & 0xFF).astype(np.float64)
                                g = ((rgb >> 8) & 0xFF).astype(np.float64)
                                b = (rgb & 0xFF).astype(np.float64)
                                cols = np.stack([r, g, b], axis=1) / 255.0
                            else:
                                if not all(name in cloud.dtype.names for name in ("r", "g", "b")):
                                    cols = np.zeros_like(pts)
                                else:
                                    r = cloud["r"]
                                    g = cloud["g"]
                                    b = cloud["b"]
                                    cols = np.stack(
                                        [r.astype(np.float64), g.astype(np.float64), b.astype(np.float64)], axis=1
                                    ) / 255.0

                            finite = np.isfinite(pts).all(axis=1)
                            pts = pts[finite]
                            cols = cols[finite]

                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(pts)
                            if cols.size:
                                pcd.colors = o3d.utility.Vector3dVector(cols)

                    if args.pcd_voxel > 0.0:
                        pcd = pcd.voxel_down_sample(args.pcd_voxel)

                    o3d.io.write_point_cloud(str(ply_path), pcd)
                except Exception as exc:
                    node.get_logger().error(f"PCD generation failed for {timestamp}: {exc}")

        node.get_logger().info(f"Done. Output: {out_dir}")
        return 0
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
