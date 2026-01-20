"""Camera publishing helpers for scanbot.ros2_manager."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Any

import carb
import torch

try:
    from isaaclab.utils.math import convert_camera_frame_orientation_convention
except Exception:  # pragma: no cover - Isaac Lab runtime only
    convert_camera_frame_orientation_convention = None

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
        if self._node is not None:
            if self._default_camera_pub is not None:
                try:
                    self._node.destroy_publisher(self._default_camera_pub)
                except Exception:
                    pass
            if self._default_depth_pub is not None:
                try:
                    self._node.destroy_publisher(self._default_depth_pub)
                except Exception:
                    pass
            if self._default_camera_info_pub is not None:
                try:
                    self._node.destroy_publisher(self._default_camera_info_pub)
                except Exception:
                    pass
            if self._default_pose_pub is not None:
                try:
                    self._node.destroy_publisher(self._default_pose_pub)
                except Exception:
                    pass
            if self._default_pcd_pub is not None:
                try:
                    self._node.destroy_publisher(self._default_pcd_pub)
                except Exception:
                    pass
            for pub in self._camera_pubs.values():
                try:
                    self._node.destroy_publisher(pub)
                except Exception:
                    pass
            for pub in self._camera_depth_pubs.values():
                try:
                    self._node.destroy_publisher(pub)
                except Exception:
                    pass
            for pub in self._camera_info_pubs.values():
                try:
                    self._node.destroy_publisher(pub)
                except Exception:
                    pass
            for pub in self._camera_pose_pubs.values():
                try:
                    self._node.destroy_publisher(pub)
                except Exception:
                    pass
            for pub in self._camera_pcd_pubs.values():
                try:
                    self._node.destroy_publisher(pub)
                except Exception:
                    pass
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
        if self._node is None or self._image_type is None or self._camera_qos is None:
            return
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
            pub = self._camera_pubs.get(name)
            pub_subscribed = self._pub_has_subscribers(pub)

            depth_pub = self._camera_depth_pubs.get(name)
            info_pub = self._camera_info_pubs.get(name)
            pose_pub = self._camera_pose_pubs.get(name)
            pcd_pub = self._camera_pcd_pubs.get(name)
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
            try:
                data = getattr(sensor, "data", None)
                output = getattr(data, "output", None) if data is not None else None

                rgb = None
                depth = None
                if isinstance(output, dict):
                    if "rgb" in output:
                        rgb = output.get("rgb")
                    elif "rgba" in output:
                        rgb = output.get("rgba")
                    elif "color" in output:
                        rgb = output.get("color")

                    if "distance_to_image_plane" in output:
                        depth = output.get("distance_to_image_plane")
                    elif "depth" in output:
                        depth = output.get("depth")

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
                    if pub_subscribed and pub is not None:
                        pub.publish(msg)
                    if default_pub_subscribed and name == self._default_camera_name and self._default_camera_pub is not None:
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
                    if depth_subscribed and depth_pub is not None:
                        depth_pub.publish(msg)
                    if (
                        default_depth_subscribed
                        and name == self._default_camera_name
                        and self._default_depth_pub is not None
                    ):
                        self._default_depth_pub.publish(msg)

                if isinstance(output, dict) and (
                    info_subscribed or (default_info_subscribed and name == self._default_camera_name)
                ):
                    k = getattr(data, "intrinsic_matrices", None)
                    h_w = getattr(data, "image_shape", None)
                    if k is not None and h_w is not None:
                        height, width = h_w
                        msg = to_camera_info_msg(
                            self._camera_info_type,
                            stamp_msg,
                            frame_id=name,
                            width=int(width),
                            height=int(height),
                            intrinsic_matrix=k,
                        )
                        if info_subscribed and info_pub is not None:
                            info_pub.publish(msg)
                        if (
                            default_info_subscribed
                            and name == self._default_camera_name
                            and self._default_camera_info_pub is not None
                        ):
                            self._default_camera_info_pub.publish(msg)

                if pose_subscribed or (default_pose_subscribed and name == self._default_camera_name):
                    pose = self._extract_pose_world_ros(sensor)
                    if pose is not None:
                        pos_w, quat_wxyz = pose
                        msg = to_pose_msg(self._pose_type, stamp_msg, frame_id="world", pos=pos_w, quat_wxyz=quat_wxyz)
                        if pose_subscribed and pose_pub is not None:
                            pose_pub.publish(msg)
                        if (
                            default_pose_subscribed
                            and name == self._default_camera_name
                            and self._default_pose_pub is not None
                        ):
                            self._default_pose_pub.publish(msg)

                if (
                    self._pointcloud2_type is not None
                    and self._pointfield_type is not None
                    and (pcd_subscribed or (default_pcd_subscribed and name == self._default_camera_name))
                    and rgb_cpu is not None
                    and depth_cpu is not None
                    and isinstance(output, dict)
                ):
                    k = getattr(data, "intrinsic_matrices", None)
                    if k is None:
                        continue
                    pose = self._extract_pose_world_ros(sensor)
                    if pose is None:
                        continue
                    pos_w, quat_wxyz = pose
                    pts_cols = compute_pointcloud_world(
                        depth=depth_cpu,
                        rgb=rgb_cpu,
                        intrinsic=k,
                        pos_world=pos_w,
                        quat_world_wxyz=quat_wxyz,
                        stride=POINTCLOUD_STRIDE,
                    )
                    if pts_cols is None:
                        continue
                    pts, cols = pts_cols
                    pc_msg = to_pointcloud2_msg(
                        self._pointcloud2_type,
                        self._pointfield_type,
                        stamp_msg,
                        frame_id="world",
                        points_xyz=pts,
                        colors_rgb=cols,
                    )
                    if pcd_subscribed and pcd_pub is not None:
                        pcd_pub.publish(pc_msg)
                    if (
                        default_pcd_subscribed
                        and name == self._default_camera_name
                        and self._default_pcd_pub is not None
                    ):
                        self._default_pcd_pub.publish(pc_msg)
            except Exception:
                continue

    def _discover_cameras(self, env) -> None:
        if self._node is None or self._image_type is None or self._camera_qos is None:
            return
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

        sensors = getattr(env.scene, "sensors", None)
        if sensors is None:
            self._cameras_ready = True
            return

        for name, sensor in sensors.items():
            if self._extract_rgb(sensor) is None:
                continue
            self._camera_sensors[name] = sensor
            topic = f"{CAMERA_TOPIC_PREFIX}/{name}/image_raw"
            try:
                self._camera_pubs[name] = self._node.create_publisher(self._image_type, topic, self._camera_qos)
            except Exception as exc:
                msg = f"[scanbot.ros2_manager] Failed to create publisher for {topic}: {exc}"
                try:
                    carb.log_warn(msg)
                except Exception:
                    pass
            try:
                topic = f"{CAMERA_TOPIC_PREFIX}/{name}/depth_raw"
                self._camera_depth_pubs[name] = self._node.create_publisher(self._image_type, topic, self._camera_qos)
            except Exception as exc:
                msg = f"[scanbot.ros2_manager] Failed to create publisher for {topic}: {exc}"
                try:
                    carb.log_warn(msg)
                except Exception:
                    pass
            if self._camera_info_type is not None:
                try:
                    topic = f"{CAMERA_TOPIC_PREFIX}/{name}/camera_info"
                    self._camera_info_pubs[name] = self._node.create_publisher(
                        self._camera_info_type, topic, self._camera_qos
                    )
                except Exception as exc:
                    msg = f"[scanbot.ros2_manager] Failed to create publisher for {topic}: {exc}"
                    try:
                        carb.log_warn(msg)
                    except Exception:
                        pass
            if self._pose_type is not None:
                try:
                    topic = f"{CAMERA_TOPIC_PREFIX}/{name}/pose_world"
                    self._camera_pose_pubs[name] = self._node.create_publisher(self._pose_type, topic, self._camera_qos)
                except Exception as exc:
                    msg = f"[scanbot.ros2_manager] Failed to create publisher for {topic}: {exc}"
                    try:
                        carb.log_warn(msg)
                    except Exception:
                        pass
            if self._pointcloud2_type is not None:
                try:
                    topic = f"{CAMERA_TOPIC_PREFIX}/{name}/{CAMERA_POINTCLOUD_SUFFIX}"
                    self._camera_pcd_pubs[name] = self._node.create_publisher(
                        self._pointcloud2_type, topic, self._camera_qos
                    )
                except Exception as exc:
                    msg = f"[scanbot.ros2_manager] Failed to create publisher for {topic}: {exc}"
                    try:
                        carb.log_warn(msg)
                    except Exception:
                        pass

        for alias in DEFAULT_CAMERA_ALIASES:
            if alias in self._camera_sensors:
                self._default_camera_name = alias
                break
        if self._default_camera_name is None and self._camera_sensors:
            self._default_camera_name = next(iter(self._camera_sensors.keys()))

        if self._default_camera_name is not None:
            try:
                topic = f"{CAMERA_TOPIC_PREFIX}/default/image_raw"
                self._default_camera_pub = self._node.create_publisher(self._image_type, topic, self._camera_qos)
            except Exception:
                self._default_camera_pub = None
            try:
                topic = f"{CAMERA_TOPIC_PREFIX}/default/depth_raw"
                self._default_depth_pub = self._node.create_publisher(self._image_type, topic, self._camera_qos)
            except Exception:
                self._default_depth_pub = None
            if self._camera_info_type is not None:
                try:
                    topic = f"{CAMERA_TOPIC_PREFIX}/default/camera_info"
                    self._default_camera_info_pub = self._node.create_publisher(
                        self._camera_info_type, topic, self._camera_qos
                    )
                except Exception:
                    self._default_camera_info_pub = None
            if self._pose_type is not None:
                try:
                    topic = f"{CAMERA_TOPIC_PREFIX}/default/pose_world"
                    self._default_pose_pub = self._node.create_publisher(self._pose_type, topic, self._camera_qos)
                except Exception:
                    self._default_pose_pub = None
            if self._pointcloud2_type is not None:
                try:
                    topic = f"{CAMERA_TOPIC_PREFIX}/default/{CAMERA_POINTCLOUD_SUFFIX}"
                    self._default_pcd_pub = self._node.create_publisher(self._pointcloud2_type, topic, self._camera_qos)
                except Exception:
                    self._default_pcd_pub = None

        if not self._camera_sensors:
            # Cameras may not be ready until the sim starts playing; keep trying.
            if not self._logged_no_cameras:
                try:
                    sensor_names = list(getattr(sensors, "keys", lambda: [])())
                except Exception:
                    sensor_names = []
                msg = f"[scanbot.ros2_manager] No camera sensors discovered yet. scene.sensors={sensor_names}"
                try:
                    carb.log_warn(msg)
                except Exception:
                    pass
                self._logged_no_cameras = True
            self._cameras_ready = False
            return

        if not self._logged_no_cameras:
            msg = f"[scanbot.ros2_manager] Discovered camera sensors: {list(self._camera_sensors.keys())}"
            try:
                carb.log_info(msg)
            except Exception:
                pass

        self._start_camera_republishers()
        self._cameras_ready = True

    def _extract_rgb(self, sensor):
        try:
            data = getattr(sensor, "data", None)
            output = getattr(data, "output", None)
            if not isinstance(output, dict):
                return None
            if "rgb" in output:
                return output.get("rgb")
            if "rgba" in output:
                return output.get("rgba")
            if "color" in output:
                return output.get("color")
        except Exception:
            return None
        return None

    def _extract_pose_world_ros(self, sensor):
        if convert_camera_frame_orientation_convention is None:
            return None
        try:
            view = getattr(sensor, "_view", None)
            if view is None:
                return None
            poses, quat_gl = view.get_world_poses([0])
            # convert to torch tensors
            if not torch.is_tensor(poses):
                poses = torch.tensor(poses)
            if not torch.is_tensor(quat_gl):
                quat_gl = torch.tensor(quat_gl)
            quat_ros = convert_camera_frame_orientation_convention(quat_gl, origin="opengl", target="ros")
            return poses[0], quat_ros[0]
        except Exception:
            return None

    def _has_camera_subscribers(self) -> bool:
        if self._default_camera_pub is not None and self._pub_has_subscribers(self._default_camera_pub):
            return True
        if self._default_depth_pub is not None and self._pub_has_subscribers(self._default_depth_pub):
            return True
        if self._default_camera_info_pub is not None and self._pub_has_subscribers(self._default_camera_info_pub):
            return True
        if self._default_pose_pub is not None and self._pub_has_subscribers(self._default_pose_pub):
            return True
        if self._default_pcd_pub is not None and self._pub_has_subscribers(self._default_pcd_pub):
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
        if pub is None:
            return False
        try:
            return pub.get_subscription_count() > 0
        except Exception:
            # If API is unavailable, assume subscribers exist to avoid silent drop.
            return True

    def _start_camera_republishers(self) -> None:
        if not CAMERA_USE_COMPRESSED_TRANSPORT:
            return
        if self._node is None:
            return
        republish_bin = "/opt/ros/humble/lib/image_transport/republish"
        if os.path.isfile(republish_bin) and os.access(republish_bin, os.X_OK):
            base_cmd = [republish_bin]
        elif shutil.which("ros2") is not None:
            base_cmd = ["ros2", "run", "image_transport", "republish"]
        else:
            msg = "[scanbot.ros2_manager] republish binary/ros2 CLI not found; skipping compressed republishers."
            try:
                carb.log_warn(msg)
            except Exception:
                pass
            return
        topics = []
        for name in self._camera_sensors:
            topics.append(f"{CAMERA_TOPIC_PREFIX}/{name}/image_raw")
        if self._default_camera_pub is not None:
            topics.append(f"{CAMERA_TOPIC_PREFIX}/default/image_raw")
        for topic in topics:
            existing = self._camera_republishers.get(topic)
            if existing is not None:
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
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=os.environ.copy(),
                )
            except Exception as exc:
                msg = f"[scanbot.ros2_manager] Failed to start republisher for {topic}: {exc}"
                try:
                    carb.log_warn(msg)
                except Exception:
                    pass
                continue
            self._camera_republishers[topic] = proc
            try:
                carb.log_info(f"[scanbot.ros2_manager] Started compressed republisher for {topic}.")
            except Exception:
                pass

    def _stop_camera_republishers(self) -> None:
        if not self._camera_republishers:
            return
        for topic, proc in list(self._camera_republishers.items()):
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            except Exception:
                pass
            self._camera_republishers.pop(topic, None)
