#!/usr/bin/env python3

import json
from typing import Callable, Generator, Optional

import rclpy
import rclpy.action
import rclpy.exceptions
import rclpy.node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.task import Future
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool


class WaypointNavigationClient:
    def __init__(self, node: rclpy.node.Node, stop_cb: Optional[Callable[..., Generator]] = None, stop_token=None) -> None:
        self._node = node

        self._navigate_future: Optional[Future] = None

        self.move_base_client = rclpy.action.ActionClient(self._node, NavigateToPose, "move_base")
        self.move_base_client.wait_for_server()

        self._clear_goals_client = self._node.create_client(SetBool, "map_image_drawer/clear_goals")
        self._clear_global_path_client = self._node.create_client(SetBool, "clear_global_path")

        self.stop_sub = self._node.create_subscription(
            Bool, "stop", self._stop_callback, 1)

        self.waypoint_reached_pub = self._node.create_publisher(
            String, "waypoint_reached", 1)
        self.remove_waypoint_from_image_pub = self._node.create_publisher(
            PoseStamped, "map_image_drawer/remove_goal", 10)
        self.add_waypoint_to_image_pub = self._node.create_publisher(
            PoseStamped, "map_image_drawer/add_goal", 10)

        self.stop_cb = stop_cb or WaypointNavigationClient.__noop
        self.stop_token = stop_token or WaypointNavigationClient.__false
        self.stop = False

        self.cancel_all_goals()

    @staticmethod
    def __noop(*_, **__) -> Generator:
        while True:
            yield

    @staticmethod
    def __false(*_, **__): return False

    def add_to_image(self, pose_goal):
        self.add_waypoint_to_image_pub.publish(pose_goal)

    def navigate_to_goal(self, pose_goal, reached_index):
        goal = NavigateToPose.Goal()
        goal.pose = pose_goal
        self.stop = False

        self._navigate_future = self.move_base_client.send_goal_async(goal)
        self._navigate_future.add_done_callback(lambda f: self._done_callback(pose_goal, reached_index, f))

    def _done_callback(self, pose_goal, reached_index, future: Future):
        if future.exception() is not None:
            print(f"Goal failed with exception: {future.exception()}")
        elif future.cancelled():
            print("Goal was cancelled.")
        else:
            result: NavigateToPose.Result = future.result()  # type: ignore
            if not self.stop and not self.stop_token(result.result):
                self._publish_waypoint_reached(
                    reached_index, pose_goal)

    def cancel_all_goals(self, clear_goals=True):
        if self._navigate_future is not None:
            self._navigate_future.cancel()
        if (clear_goals):
            self._clear_goals_client.wait_for_service(timeout_sec=2.0)
            try:
                request = SetBool.Request()
                request.data = True
                self._clear_goals_client.call(request)
            except Exception as e:
                self._node.get_logger().warn(f"Service call failed: {e}")

    def clear_global_path(self):
        self._clear_global_path_client.wait_for_service(timeout_sec=2.0)
        try:
            request = SetBool.Request()
            request.data = True
            self._clear_global_path_client.call(request)
        except Exception as e:
            self._node.get_logger().warn(f"Service call failed: {e}")

    def _publish_waypoint_reached(self, waypoint_index, goal_pose):
        waypoint_reached_json_message = {
            "type": "waypointReached", "waypointNumber": waypoint_index}
        waypoint_reached_msg = json.dumps(waypoint_reached_json_message)
        self.waypoint_reached_pub.publish(waypoint_reached_msg)
        self.remove_waypoint_from_image_pub.publish(goal_pose)

    def _stop_callback(self, msg):
        if msg.data == True:
            self.stop = True
            cb = self.stop_cb(msg)
            next(cb)
            self.cancel_all_goals()
            self.clear_global_path()
            next(cb)
