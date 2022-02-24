#!/usr/bin/env python3

import rospy
from opentera_webrtc_ros_msgs.msg import WaypointArray
from std_msgs.msg import Bool
from opentera_webrtc_ros.libmapimageconverter import convert_waypoint_to_pose
from opentera_webrtc_ros.libnavigation import WaypointNavigationClient


class GoalManager():
    def __init__(self):
        self.nav_client = WaypointNavigationClient(
            stop_cb=self.__stop_cb, stop_token=self.__stop_token)

        self.should_stop = False
        self.pose_goals = []

        # Subscribers and publishers
        self.waypoints_sub = rospy.Subscriber(
            "waypoints", WaypointArray, self.waypoints_cb)
        self.start_sub = rospy.Subscriber(
            "start", Bool, self.start_cb, queue_size=1)

        rospy.loginfo("Goal manager ready")

    def __stop_token(self, *_, **__):
        return self.should_stop

    def waypoints_cb(self, msg):
        for waypoint in msg.waypoints:
            pose_goal = convert_waypoint_to_pose(waypoint)
            if pose_goal is not None:
                pose_goal.pose.position.z = 1 + len(self.pose_goals)
                self.nav_client.add_to_image(pose_goal)
                self.pose_goals.append(pose_goal)

    def __stop_cb(self, _):
        rospy.loginfo("Stopping")
        self.should_stop = True
        yield
        self.pose_goals.clear()
        yield

    def start_cb(self, msg):
        rospy.loginfo("Starting")
        if msg.data == True:
            self.nav_client.cancel_all_goals(False)
            self.nav_client.clear_global_path()
            self.should_stop = False

            for pose_goal in self.pose_goals:
                self.nav_client.navigate_to_goal(
                    pose_goal, round(pose_goal.pose.position.z))
                if self.should_stop:
                    break
            self.nav_client.clear_global_path()
            self.pose_goals.clear()


if __name__ == '__main__':
    rospy.init_node("goal_manager")
    try:
        goal_manager = GoalManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
