#!/usr/bin/python

import rospy
import actionlib
from math import pi
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import PoseStamped, PoseArray
from opentera_webrtc_ros_msgs.msg import Waypoint, WaypointArray
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus

class GoalManager():
    def __init__(self):
        rospy.init_node("goal_manager")  

        # Global action client used to send waypoints to move_base
        print("creting action client")
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        print("Waiting for server")
        self.move_base_client.wait_for_server()
        # Make sure no goals are active
        print("Cancelling goals")
        self.move_base_client.cancel_all_goals()

        # Subscribers and publishers
        self.waypoints_sub = rospy.Subscriber("waypoints", WaypointArray, self.waypoints_cb)
        self.waypoints_sub = rospy.Subscriber("map_goal", PoseStamped, self.map_goal_cb)
        self.map_image_goals_pub = rospy.Publisher("map_image_goal", PoseStamped, queue_size=1)

    def waypoints_cb(self, msg):
        for waypoint in msg.waypoints:
            self.transform_waypoint_to_pose(waypoint)

    def transform_waypoint_to_pose(self, waypoint):
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.pose.position.x = waypoint.x
        pose.pose.position.y = waypoint.y
        yaw = wrap_angle_pi(waypoint.yaw - pi)
        quaternion = quaternion_from_euler(0, 0, yaw)
        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]

        self.map_image_goals_pub.publish(pose)

    def map_goal_cb(self, msg):
        print("Map goal:")
        print(msg)
        goal = MoveBaseGoal()
        goal.target_pose = msg
        self.move_base_client.send_goal(goal)


def wrap_angle_pi(angle):
    return (angle + pi) % (2 * pi) - pi

if __name__ == '__main__':
    print("Goal manager ready")
    try:
        goal_manager = GoalManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
