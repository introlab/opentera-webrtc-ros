#!/usr/bin/env python3

from typing import Callable, Optional

import rclpy
import rclpy.exceptions
import rclpy.node
import rclpy.task
from geometry_msgs.msg import PoseStamped, Quaternion
from opentera_webrtc_ros_msgs.msg import Waypoint

# from opentera_webrtc_ros_msgs.srv import MapGoalToImageGoal
from opentera_webrtc_ros_msgs.srv import ImageGoalToMapGoal
from transforms3d.euler import euler2quat, quat2euler


def euler_from_quaternion(quaternion: Quaternion, axes="sxyz") -> tuple:
    x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
    return quat2euler((w, x, y, z), axes)


def quaternion_from_euler(roll, pitch, yaw, axes="sxyz") -> Quaternion:
    w, x, y, z = euler2quat(roll, pitch, yaw, axes)
    return Quaternion(x=x, y=y, z=z, w=w)


class PoseWaypointConverter:
    def __init__(self, node: rclpy.node.Node) -> None:
        self._node = node
        # self._map_goal_to_image_goal_client = self._node.create_client(
        #     MapGoalToImageGoal, "map_goal_to_image_goal"
        # )
        self._image_goal_to_map_goal_client = self._node.create_client(
            ImageGoalToMapGoal, "image_goal_to_map_goal"
        )

    # def convert_pose_to_waypoint(self, pose: PoseStamped, callback: Callable) -> Optional[rclpy.task.Future]:
    #     def cb(future: rclpy.task.Future):
    #         try:
    #             result: MapGoalToImageGoal.Response = future.result()  # type: ignore

    #             if not result.success:
    #                 self._node.get_logger().error(f"Failed to convert pose ({pose}) to waypoint")
    #                 return

    #             waypoint = Waypoint()
    #             waypoint.x = pose.pose.position.x
    #             waypoint.y = pose.pose.position.y
    #             waypoint.yaw = euler_from_quaternion(pose.pose.orientation)[2]
    #             callback(waypoint)
    #         except Exception as e:
    #             self._node.get_logger().error(f"Failed to convert pose to waypoint: {e}")
    #             return

    #     return self._call_image_goal_to_map_goal_client(pose, cb)

    def convert_waypoint_to_pose(
        self, waypoint: Waypoint, callback: Callable
    ) -> Optional[rclpy.task.Future]:
        def cb(future: rclpy.task.Future):
            try:
                result: ImageGoalToMapGoal.Response = future.result()  # type: ignore

                if not result.success:
                    self._node.get_logger().error(
                        f"Failed to convert waypoint ({waypoint}) to pose"
                    )
                    return

                callback(result.map_goal)
            except Exception as e:
                self._node.get_logger().error(
                    f"Failed to convert waypoint to pose: {e}"
                )
                return

        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.pose.position.x = waypoint.x
        pose.pose.position.y = waypoint.y
        pose.pose.position.z = 0.0
        yaw = -waypoint.yaw
        pose.pose.orientation = quaternion_from_euler(0.0, 0.0, yaw)

        return self._call_image_goal_to_map_goal_client(pose, cb)

    # def _call_map_goal_to_image_goal_client(
    #     self, map_goal: PoseStamped, callback: Callable
    # ) -> Optional[rclpy.task.Future]:

    #     try:
    #         req = MapGoalToImageGoal.Request()
    #         req.map_goal = map_goal
    #         if not self._map_goal_to_image_goal_client.wait_for_service(timeout_sec=2.0):
    #             self._node.get_logger().error(f"{self._map_goal_to_image_goal_client.srv_name}: service not available")
    #             return None
    #         future = self._map_goal_to_image_goal_client.call_async(req).
    #         future.add_done_callback(callback)
    #         return future
    #     except Exception as e:
    #         self._node.get_logger().warn(f"{self._map_goal_to_image_goal_client.srv_name} service call failed: {e}")

    def _call_image_goal_to_map_goal_client(
        self, image_goal: PoseStamped, callback: Callable
    ) -> Optional[rclpy.task.Future]:
        try:
            req = ImageGoalToMapGoal.Request()
            req.image_goal = image_goal
            if not self._image_goal_to_map_goal_client.wait_for_service(
                timeout_sec=2.0
            ):
                self._node.get_logger().error(
                    f"{self._image_goal_to_map_goal_client.srv_name}: service not available"
                )
                return None
            future = self._image_goal_to_map_goal_client.call_async(req)
            future.add_done_callback(callback)
            return future
        except Exception as e:
            self._node.get_logger().warn(
                f"{self._image_goal_to_map_goal_client.srv_name} service call failed: {e}"
            )
