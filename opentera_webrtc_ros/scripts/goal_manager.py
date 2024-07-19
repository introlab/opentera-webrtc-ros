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
        def cb(pose_goal: PoseStamped):
            pose_goal.pose.position.z = float(1 + len(self.pose_goals))
            self.nav_client.add_to_image(pose_goal)
            self.pose_goals.append(pose_goal)

        for waypoint in msg.waypoints:
            self._pose_waypoint_converter.convert_waypoint_to_pose(waypoint, cb)

    def __stop_cb(self, _):
        self.get_logger().debug("Stopping")
        self.should_stop = True
        yield
        self.pose_goals.clear()
        yield

    def start_cb(self, msg: Bool):
        self.get_logger().debug("Starting")
        if msg.data == True:
            self.nav_client.cancel_all_goals(False)
            self.nav_client.clear_global_path()
            self.should_stop = False

            self.nav_client.navigate_through_goals(self.pose_goals)
            self.pose_goals = []


def main():
    rclpy.init()
    goal_manager = GoalManager()
    
    try:
        rclpy.spin(goal_manager)
    except KeyboardInterrupt:
        pass

    goal_manager.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
