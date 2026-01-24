"""Camera publishing helpers for scanbot.ros2_manager."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Any

import carb
import torch
from isaaclab.utils.math import convert_camera_frame_orientation_convention
from isaaclab.sensors.camera.camera import Camera

from .camera_conversions import (
    compute_pointcloud_world,
    maybe_to_cpu,
    to_camera_info_msg,
    to_depth_msg,
    to_image_msg,
    to_pointcloud2_msg,
    to_pose_msg,
)
from .config import (
    CAMERA_COMPRESSED_FORMAT,
    CAMERA_PUB_HZ,
    CAMERA_STRIDE,
    CAMERA_TOPIC_PREFIX,
    CAMERA_USE_COMPRESSED_TRANSPORT,
    CAMERA_POINTCLOUD_SUFFIX,
    DEFAULT_CAMERA_ALIASES,
    POINTCLOUD_STRIDE,
)


class CameraBridge:
    def __init__(self, node, image_type, camera_info_type, pose_type, pointcloud2_type, pointfield_type, camera_qos) -> None:
        self._node = node
        self._image_type = image_type
        self._camera_info_type = camera_info_type
        self._pose_type = pose_type
        self._pointcloud2_type = pointcloud2_type
        self._pointfield_type = pointfield_type
        self._camera_qos = camera_qos
        self._camera_pubs: dict[str, Any] = {}
        self._camera_depth_pubs: dict[str, Any] = {}
        self._camera_info_pubs: dict[str, Any] = {}
        self._camera_pose_pubs: dict[str, Any] = {}
        self._camera_pcd_pubs: dict[str, Any] = {}
        self._camera_sensors: dict[str, Any] = {}
        self._default_camera_name: str | None = None
        self._default_camera_pub = None
        self._default_depth_pub = None
        self._default_camera_info_pub = None
        self._default_pose_pub = None
        self._default_pcd_pub = None
        self._next_camera_pub = 0.0
        self._next_camera_discover_attempt = 0.0
        self._cameras_ready = False
        self._camera_republishers: dict[str, subprocess.Popen] = {}
        self._logged_no_cameras = False

    def shutdown(self) -> None:
        self._node.destroy_publisher(self._default_camera_pub)
        self._node.destroy_publisher(self._default_depth_pub)
        self._node.destroy_publisher(self._default_camera_info_pub)
        self._node.destroy_publisher(self._default_pose_pub)
        self._node.destroy_publisher(self._default_pcd_pub)
        for pub in self._camera_pubs.values():
            self._node.destroy_publisher(pub)
        for pub in self._camera_depth_pubs.values():
            self._node.destroy_publisher(pub)
        for pub in self._camera_info_pubs.values():
            self._node.destroy_publisher(pub)
        for pub in self._camera_pose_pubs.values():
            self._node.destroy_publisher(pub)
        for pub in self._camera_pcd_pubs.values():
            self._node.destroy_publisher(pub)
        self._camera_pubs = {}
        self._camera_depth_pubs = {}
        self._camera_info_pubs = {}
        self._camera_pose_pubs = {}
        self._camera_pcd_pubs = {}
        self._camera_sensors = {}
        self._default_camera_name = None
        self._default_camera_pub = None
        self._default_depth_pub = None
        self._default_camera_info_pub = None
        self._default_pose_pub = None
        self._default_pcd_pub = None
        self._cameras_ready = False
        self._stop_camera_republishers()

    def maybe_publish(self, env) -> None:
        if not self._cameras_ready:
            now = time.monotonic()
            if now >= self._next_camera_discover_attempt:
                # Avoid hammering camera data reads while the sim is still initializing.
                self._next_camera_discover_attempt = now + 1.0
                self._discover_cameras(env)
        if not self._camera_sensors:
            return
        self._start_camera_republishers()
        if not self._has_camera_subscribers():
            return
        now = time.monotonic()
        if now < self._next_camera_pub:
            return
        self._next_camera_pub = now + (1.0 / max(CAMERA_PUB_HZ, 1.0))

        default_pub_subscribed = self._pub_has_subscribers(self._default_camera_pub)
        default_depth_subscribed = self._pub_has_subscribers(self._default_depth_pub)
        default_info_subscribed = self._pub_has_subscribers(self._default_camera_info_pub)
        default_pose_subscribed = self._pub_has_subscribers(self._default_pose_pub)
        default_pcd_subscribed = self._pub_has_subscribers(self._default_pcd_pub)

        stamp_msg = self._node.get_clock().now().to_msg()
        for name, sensor in self._camera_sensors.items():
            pub = self._camera_pubs[name]
            pub_subscribed = self._pub_has_subscribers(pub)

            depth_pub = self._camera_depth_pubs[name]
            info_pub = self._camera_info_pubs[name]
            pose_pub = self._camera_pose_pubs[name]
            pcd_pub = self._camera_pcd_pubs[name]
            depth_subscribed = self._pub_has_subscribers(depth_pub)
            info_subscribed = self._pub_has_subscribers(info_pub)
            pose_subscribed = self._pub_has_subscribers(pose_pub)
            pcd_subscribed = self._pub_has_subscribers(pcd_pub)

            any_subscribed = (
                pub_subscribed
                or depth_subscribed
                or info_subscribed
                or pose_subscribed
                or pcd_subscribed
                or (default_pub_subscribed and name == self._default_camera_name)
                or (default_depth_subscribed and name == self._default_camera_name)
                or (default_info_subscribed and name == self._default_camera_name)
                or (default_pose_subscribed and name == self._default_camera_name)
                or (default_pcd_subscribed and name == self._default_camera_name)
            )
            if not any_subscribed:
                continue
            data = sensor.data
            output = data.output

            rgb = None
            depth = None
            if isinstance(output, dict):
                if "rgb" in output:
                    rgb = output["rgb"]
                elif "rgba" in output:
                    rgb = output["rgba"]
                elif "color" in output:
                    rgb = output["color"]

                if "distance_to_image_plane" in output:
                    depth = output["distance_to_image_plane"]
                elif "depth" in output:
                    depth = output["depth"]

            need_rgb = rgb is not None and (
                pub_subscribed
                or (default_pub_subscribed and name == self._default_camera_name)
                or pcd_subscribed
                or (default_pcd_subscribed and name == self._default_camera_name)
            )
            need_depth = depth is not None and (
                depth_subscribed
                or (default_depth_subscribed and name == self._default_camera_name)
                or pcd_subscribed
                or (default_pcd_subscribed and name == self._default_camera_name)
            )
            rgb_cpu = maybe_to_cpu(rgb) if need_rgb else None
            depth_cpu = maybe_to_cpu(depth) if need_depth else None

            if rgb is not None and (pub_subscribed or (default_pub_subscribed and name == self._default_camera_name)):
                msg = to_image_msg(
                    self._image_type,
                    stamp_msg,
                    frame_id=name,
                    rgb_tensor=rgb,
                    stride=max(1, CAMERA_STRIDE),
                    rgb_cpu=rgb_cpu,
                )
                if pub_subscribed:
                    pub.publish(msg)
                if default_pub_subscribed and name == self._default_camera_name:
                    self._default_camera_pub.publish(msg)

            if depth is not None and (
                depth_subscribed or (default_depth_subscribed and name == self._default_camera_name)
            ):
                msg = to_depth_msg(
                    self._image_type,
                    stamp_msg,
                    frame_id=name,
                    depth_tensor=depth,
                    stride=max(1, CAMERA_STRIDE),
                    depth_cpu=depth_cpu,
                )
                if depth_subscribed:
                    depth_pub.publish(msg)
                if (
                    default_depth_subscribed
                    and name == self._default_camera_name
                ):
                    self._default_depth_pub.publish(msg)

            if isinstance(output, dict) and (
                info_subscribed or (default_info_subscribed and name == self._default_camera_name)
            ):
                k = data.intrinsic_matrices
                h_w = data.image_shape
                height, width = h_w
                msg = to_camera_info_msg(
                    self._camera_info_type,
                    stamp_msg,
                    frame_id=name,
                    width=int(width),
                    height=int(height),
                    intrinsic_matrix=k,
                )
                if info_subscribed:
                    info_pub.publish(msg)
                if (
                    default_info_subscribed
                    and name == self._default_camera_name
                ):
                    self._default_camera_info_pub.publish(msg)

            if pose_subscribed or (default_pose_subscribed and name == self._default_camera_name):
                pos_w, quat_wxyz = self._extract_pose_world_ros(sensor)
                msg = to_pose_msg(self._pose_type, stamp_msg, frame_id="world", pos=pos_w, quat_wxyz=quat_wxyz)
                if pose_subscribed:
                    pose_pub.publish(msg)
                if (
                    default_pose_subscribed
                    and name == self._default_camera_name
                ):
                    self._default_pose_pub.publish(msg)

            if (
                (pcd_subscribed or (default_pcd_subscribed and name == self._default_camera_name))
                and rgb_cpu is not None
                and depth_cpu is not None
                and isinstance(output, dict)
            ):
                k = data.intrinsic_matrices
                pos_w, quat_wxyz = self._extract_pose_world_ros(sensor)
                pts, cols = compute_pointcloud_world(
                    depth=depth_cpu,
                    rgb=rgb_cpu,
                    intrinsic=k,
                    pos_world=pos_w,
                    quat_world_wxyz=quat_wxyz,
                    stride=POINTCLOUD_STRIDE,
                )
                pc_msg = to_pointcloud2_msg(
                    self._pointcloud2_type,
                    self._pointfield_type,
                    stamp_msg,
                    frame_id="world",
                    points_xyz=pts,
                    colors_rgb=cols,
                )
                if pcd_subscribed:
                    pcd_pub.publish(pc_msg)
                if (
                    default_pcd_subscribed
                    and name == self._default_camera_name
                ):
                    self._default_pcd_pub.publish(pc_msg)

    def _discover_cameras(self, env) -> None:
        self._camera_sensors = {}
        self._camera_pubs = {}
        self._camera_depth_pubs = {}
        self._camera_info_pubs = {}
        self._camera_pose_pubs = {}
        self._camera_pcd_pubs = {}
        self._default_camera_name = None
        self._default_camera_pub = None
        self._default_depth_pub = None
        self._default_camera_info_pub = None
        self._default_pose_pub = None
        self._default_pcd_pub = None

        scene = env.scene
        sensors = scene.sensors

        for name, sensor in sensors.items():
            if not isinstance(sensor, Camera):
                continue
            if self._extract_rgb(sensor) is None:
                continue
            self._camera_sensors[name] = sensor
            topic = f"{CAMERA_TOPIC_PREFIX}/{name}/image_raw"
            self._camera_pubs[name] = self._node.create_publisher(self._image_type, topic, self._camera_qos)
            topic = f"{CAMERA_TOPIC_PREFIX}/{name}/depth_raw"
            self._camera_depth_pubs[name] = self._node.create_publisher(self._image_type, topic, self._camera_qos)
            topic = f"{CAMERA_TOPIC_PREFIX}/{name}/camera_info"
            self._camera_info_pubs[name] = self._node.create_publisher(
                self._camera_info_type, topic, self._camera_qos
            )
            topic = f"{CAMERA_TOPIC_PREFIX}/{name}/pose_world"
            self._camera_pose_pubs[name] = self._node.create_publisher(self._pose_type, topic, self._camera_qos)
            topic = f"{CAMERA_TOPIC_PREFIX}/{name}/{CAMERA_POINTCLOUD_SUFFIX}"
            self._camera_pcd_pubs[name] = self._node.create_publisher(
                self._pointcloud2_type, topic, self._camera_qos
            )

        for alias in DEFAULT_CAMERA_ALIASES:
            if alias in self._camera_sensors:
                self._default_camera_name = alias
                break
        if self._default_camera_name is None and self._camera_sensors:
            self._default_camera_name = next(iter(self._camera_sensors.keys()))

        topic = f"{CAMERA_TOPIC_PREFIX}/default/image_raw"
        self._default_camera_pub = self._node.create_publisher(self._image_type, topic, self._camera_qos)
        topic = f"{CAMERA_TOPIC_PREFIX}/default/depth_raw"
        self._default_depth_pub = self._node.create_publisher(self._image_type, topic, self._camera_qos)
        topic = f"{CAMERA_TOPIC_PREFIX}/default/camera_info"
        self._default_camera_info_pub = self._node.create_publisher(
            self._camera_info_type, topic, self._camera_qos
        )
        topic = f"{CAMERA_TOPIC_PREFIX}/default/pose_world"
        self._default_pose_pub = self._node.create_publisher(self._pose_type, topic, self._camera_qos)
        topic = f"{CAMERA_TOPIC_PREFIX}/default/{CAMERA_POINTCLOUD_SUFFIX}"
        self._default_pcd_pub = self._node.create_publisher(self._pointcloud2_type, topic, self._camera_qos)

        if not self._camera_sensors:
            # Cameras may not be ready until the sim starts playing; keep trying.
            if not self._logged_no_cameras:
                sensor_names = list(sensors.keys())
                msg = f"[scanbot.ros2_manager] No camera sensors discovered yet. scene.sensors={sensor_names}"
                carb.log_warn(msg)
                self._logged_no_cameras = True
            self._cameras_ready = False
            return

        if not self._logged_no_cameras:
            msg = f"[scanbot.ros2_manager] Discovered camera sensors: {list(self._camera_sensors.keys())}"
            carb.log_info(msg)

        self._start_camera_republishers()
        self._cameras_ready = True

    def _extract_rgb(self, sensor):
        data = sensor.data
        output = data.output
        if not isinstance(output, dict):
            return None
        if "rgb" in output:
            return output["rgb"]
        if "rgba" in output:
            return output["rgba"]
        if "color" in output:
            return output["color"]
        return None

    def _extract_pose_world_ros(self, sensor):
        view = sensor._view
        poses, quat_gl = view.get_world_poses([0])
        if not torch.is_tensor(poses):
            poses = torch.tensor(poses)
        if not torch.is_tensor(quat_gl):
            quat_gl = torch.tensor(quat_gl)
        quat_ros = convert_camera_frame_orientation_convention(quat_gl, origin="opengl", target="ros")
        return poses[0], quat_ros[0]

    def _has_camera_subscribers(self) -> bool:
        if self._pub_has_subscribers(self._default_camera_pub):
            return True
        if self._pub_has_subscribers(self._default_depth_pub):
            return True
        if self._pub_has_subscribers(self._default_camera_info_pub):
            return True
        if self._pub_has_subscribers(self._default_pose_pub):
            return True
        if self._pub_has_subscribers(self._default_pcd_pub):
            return True
        for pub in self._camera_pubs.values():
            if self._pub_has_subscribers(pub):
                return True
        for pub in self._camera_depth_pubs.values():
            if self._pub_has_subscribers(pub):
                return True
        for pub in self._camera_info_pubs.values():
            if self._pub_has_subscribers(pub):
                return True
        for pub in self._camera_pose_pubs.values():
            if self._pub_has_subscribers(pub):
                return True
        for pub in self._camera_pcd_pubs.values():
            if self._pub_has_subscribers(pub):
                return True
        return False

    def _pub_has_subscribers(self, pub) -> bool:
        return pub.get_subscription_count() > 0

    def _start_camera_republishers(self) -> None:
        if not CAMERA_USE_COMPRESSED_TRANSPORT:
            return
        republish_bin = "/opt/ros/humble/lib/image_transport/republish"
        if os.path.isfile(republish_bin) and os.access(republish_bin, os.X_OK):
            base_cmd = [republish_bin]
        elif shutil.which("ros2") is not None:
            base_cmd = ["ros2", "run", "image_transport", "republish"]
        else:
            msg = "[scanbot.ros2_manager] republish binary/ros2 CLI not found; skipping compressed republishers."
            carb.log_warn(msg)
            return
        topics = []
        for name in self._camera_sensors:
            topics.append(f"{CAMERA_TOPIC_PREFIX}/{name}/image_raw")
        topics.append(f"{CAMERA_TOPIC_PREFIX}/default/image_raw")
        for topic in topics:
            if topic in self._camera_republishers:
                existing = self._camera_republishers[topic]
                if existing.poll() is None:
                    continue
                self._camera_republishers.pop(topic, None)
            node_name = "scanbot_image_republish_" + topic.strip("/").replace("/", "_")
            compressed_topic = f"{topic}/compressed"
            cmd = base_cmd + [
                "raw",
                "compressed",
                "--ros-args",
                "--param",
                f"format:={CAMERA_COMPRESSED_FORMAT}",
                "--remap",
                f"/in:={topic}",
                "--remap",
                f"/out/compressed:={compressed_topic}",
                "--remap",
                f"__node:={node_name}",
            ]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=os.environ.copy(),
            )
            self._camera_republishers[topic] = proc
            carb.log_info(f"[scanbot.ros2_manager] Started compressed republisher for {topic}.")

    def _stop_camera_republishers(self) -> None:
        if not self._camera_republishers:
            return
        for topic, proc in list(self._camera_republishers.items()):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
            self._camera_republishers.pop(topic, None)
