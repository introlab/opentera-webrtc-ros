#!/usr/bin/env python3

from typing import Optional

import rclpy
import rclpy.exceptions
import rclpy.node
from geometry_msgs.msg import PoseStamped, Quaternion

# from opentera_webrtc_ros_msgs.srv import MapGoalToImageGoal
from opentera_webrtc_ros_msgs.srv import ImageGoalToMapGoal
from opentera_webrtc_ros_msgs.msg import Waypoint
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

    # def convert_pose_to_waypoint(self, pose: PoseStamped) -> Optional[Waypoint]:
    #     waypoint_pose = self._call_map_goal_to_image_goal_client(pose)
    #     if waypoint_pose is None:
    #         return None
    #     waypoint = Waypoint()
    #     waypoint.x = waypoint_pose.pose.position.x
    #     waypoint.y = waypoint_pose.pose.position.y
    #     waypoint.yaw = euler_from_quaternion(waypoint_pose.pose.orientation)[2]
    #     return waypoint

    def convert_waypoint_to_pose(self, waypoint: Waypoint) -> Optional[PoseStamped]:
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.pose.position.x = waypoint.x
        pose.pose.position.y = waypoint.y
        pose.pose.position.z = 0
        yaw = -waypoint.yaw
        pose.pose.orientation = quaternion_from_euler(0, 0, yaw)

        return self._call_image_goal_to_map_goal_client(pose)

    # def _call_map_goal_to_image_goal_client(
    #     self, map_goal: PoseStamped
    # ) -> Optional[PoseStamped]:
    #     self._map_goal_to_image_goal_client.wait_for_service(timeout_sec=2.0)

    #     try:
    #         req = MapGoalToImageGoal.Request()
    #         req.map_goal = map_goal
    #         result = self._map_goal_to_image_goal_client.call(req)
    #         return result.image_goal
    #     except Exception as e:
    #         self._node.get_logger().warn(f"Service call failed: {e}")

    def _call_image_goal_to_map_goal_client(
        self, image_goal: PoseStamped
    ) -> Optional[PoseStamped]:
        self._image_goal_to_map_goal_client.wait_for_service(timeout_sec=2.0)
        try:
            req = ImageGoalToMapGoal.Request()
            req.image_goal = image_goal
            result = self._image_goal_to_map_goal_client.call(req)
            return result.map_goal
        except Exception as e:
            self._node.get_logger().warn(f"Service call failed: {e}")
