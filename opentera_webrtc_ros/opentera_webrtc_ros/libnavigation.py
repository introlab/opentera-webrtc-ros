#!/usr/bin/env python3

import json
from typing import Callable, Generator, Optional, cast

import rclpy
import rclpy.action
import rclpy.client
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
        self._navigate_future_goal: Optional[Future] = None
        self._goal_handle: Optional[rclpy.action.client.ClientGoalHandle] = None

        self.nav2_client = rclpy.action.ActionClient(self._node, NavigateToPose, "navigate_to_pose")

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
    
    def _pub_goal(self, pose_goal, done_cb):
        goal = NavigateToPose.Goal()
        goal.pose = pose_goal
        self.stop = False

        self._navigate_future = self.nav2_client.send_goal_async(goal)
        self._navigate_future.add_done_callback(done_cb)

    def navigate_through_goals(self, pose_goals):
        goals_count = len(pose_goals)

        def _done_cb(done_future: Future):
            self._goal_handle = None
            if done_future.exception() is not None:
                self._node.get_logger().error(f"Goal failed with exception: {done_future.exception()}")
                return
            elif done_future.cancelled():
                self._node.get_logger().info("Goal was cancelled.")
                return

            result: NavigateToPose.Result = done_future.result()  # type: ignore
            self._navigate_future_goal = None

            if not self.stop and not self.stop_token(result.result):
                reached_index = goals_count - (len(pose_goals) - 1)
                self._node.get_logger().debug(f"Waypoint reached: {reached_index}")
                self._publish_waypoint_reached(
                    reached_index, pose_goals[0])
                pose_goals.pop(0)
                if len(pose_goals) > 0:
                    self._pub_goal(pose_goals[0], _cb)
                else:
                    self.clear_global_path()
            else:
                self._node.get_logger().debug(f"Stopped or cancelled")

        def _cb(future: Future):
            goal_handle = cast(rclpy.action.client.ClientGoalHandle, future.result())
            self._navigate_future = None
            if not goal_handle.accepted:
                self._node.get_logger().error(f"Goal {pose_goals[0]} rejected with id {goal_handle.goal_id}")
                return

            self._goal_handle = goal_handle
            self._navigate_future_goal = cast(Future, self._goal_handle.get_result_async())
            self._navigate_future_goal.add_done_callback(_done_cb)

        self._pub_goal(pose_goals[0], _cb)

    def _make_service_callback(self, service_name: str) -> Callable:
        def callback(future: Future):
            try:
                response: SetBool.Response = future.result()  # type: ignore
                if response.success:
                    self._node.get_logger().debug(f"{service_name} service called: success")
                else:
                    self._node.get_logger().warn(f"{service_name} service called, returned failure: {response.message}")
            except Exception as e:
                self._node.get_logger().warn(f"{service_name} service call failed: {e}")
        return callback

    def cancel_all_goals(self, clear_goals=True):
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()
            self._goal_handle = None

        if clear_goals:
            try:
                request = SetBool.Request()
                request.data = True
                if self._clear_goals_client.service_is_ready():
                    future = self._clear_goals_client.call_async(request)
                    future.add_done_callback(self._make_service_callback(self._clear_goals_client.srv_name))
            except Exception as e:
                self._node.get_logger().warn(f"{self._clear_goals_client.srv_name} service call failed: {e}")

    def clear_global_path(self):
        try:
            request = SetBool.Request()
            request.data = True
            if self._clear_global_path_client.service_is_ready():
                future = self._clear_global_path_client.call_async(request)
                future.add_done_callback(self._make_service_callback(self._clear_global_path_client.srv_name))
        except Exception as e:
            self._node.get_logger().warn(f"{self._clear_global_path_client.srv_name} service call failed: {e}")

    def _publish_waypoint_reached(self, waypoint_index, goal_pose):
        waypoint_reached_json_message = {
            "type": "waypointReached", "waypointNumber": waypoint_index}
        waypoint_reached_msg = json.dumps(waypoint_reached_json_message)
        self.waypoint_reached_pub.publish(String(data=waypoint_reached_msg))
        self.remove_waypoint_from_image_pub.publish(goal_pose)

    def _stop_callback(self, msg: Bool):
        if msg.data == True:
            self.stop = True
            cb = self.stop_cb(msg)
            next(cb)
            self.cancel_all_goals()
            self.clear_global_path()
            next(cb)
