#!/usr/bin/env python3

import rclpy
import rclpy.exceptions
import rclpy.node
from typing import List
from opentera_webrtc_ros.libmapimageconverter import PoseWaypointConverter
from opentera_webrtc_ros.libnavigation import WaypointNavigationClient
from opentera_webrtc_ros_msgs.msg import WaypointArray
import rclpy.qos
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped


class GoalManager(rclpy.node.Node):
    def __init__(self):
        super().__init__("goal_manager")  # type: ignore

        self._pose_waypoint_converter = PoseWaypointConverter(self)

        self.nav_client = WaypointNavigationClient(self,
            stop_cb=self.__stop_cb, stop_token=self.__stop_token
        )

        self.should_stop = False
        self.pose_goals: List[PoseStamped] = []

        # Subscribers and publishers
        self.waypoints_sub = self.create_subscription(
            WaypointArray, "waypoints", self.waypoints_cb, 1
        )
        self.start_sub = self.create_subscription(
            Bool, "start", self.start_cb, 1
        )

        self.get_logger().info("Goal manager ready")

    def __stop_token(self, *_, **__):
        return self.should_stop

    def waypoints_cb(self, msg: WaypointArray):
        for waypoint in msg.waypoints:
            pose_goal = self._pose_waypoint_converter.convert_waypoint_to_pose(waypoint)
            if pose_goal is not None:
                pose_goal.pose.position.z = 1 + len(self.pose_goals)
                self.nav_client.add_to_image(pose_goal)
                self.pose_goals.append(pose_goal)

    def __stop_cb(self, _):
        self.get_logger().info("Stopping")
        self.should_stop = True
        yield
        self.pose_goals.clear()
        yield

    def start_cb(self, msg: Bool):
        self.get_logger().info("Starting")
        if msg.data == True:
            self.nav_client.cancel_all_goals(False)
            self.nav_client.clear_global_path()
            self.should_stop = False

            for pose_goal in self.pose_goals:
                self.nav_client.navigate_to_goal(
                    pose_goal, round(pose_goal.pose.position.z)
                )
                if self.should_stop:
                    break
            self.nav_client.clear_global_path()
            self.pose_goals.clear()


if __name__ == '__main__':
    rclpy.init()
    try:
        goal_manager = GoalManager()
        rclpy.spin(goal_manager)
    except KeyboardInterrupt:
        pass
