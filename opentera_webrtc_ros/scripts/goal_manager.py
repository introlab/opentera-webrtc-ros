#!/usr/bin/python

import rospy
import actionlib
from math import pi
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import PoseStamped, PoseArray
from opentera_webrtc_ros_msgs.msg import Waypoint, WaypointArray
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from map_image_generator.srv import ImageGoalToMapGoal

class GoalManager():
    def __init__(self):
        rospy.init_node("goal_manager")  

        # Global action client used to send waypoints to move_base
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.move_base_client.wait_for_server()
        # Make sure no goals are active
        self.move_base_client.cancel_all_goals()

        # Subscribers and publishers
        self.waypoints_sub = rospy.Subscriber("waypoints", WaypointArray, self.waypoints_cb)

    def waypoints_cb(self, msg):
        for waypoint in msg.waypoints:
            pose_goal = self.transform_waypoint_to_pose(waypoint)
            self.send_goal(pose_goal)

    def transform_waypoint_to_pose(self, waypoint):
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.pose.position.x = waypoint.x
        pose.pose.position.y = waypoint.y
        yaw = -waypoint.yaw
        quaternion = quaternion_from_euler(0, 0, yaw)
        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]

        return self.image_goal_to_map_goal_client(pose)

    def image_goal_to_map_goal_client(self, image_goal):
        try:
            image_goal_to_map_goal = rospy.ServiceProxy('image_goal_to_map_goal', ImageGoalToMapGoal)
            res = image_goal_to_map_goal(image_goal)
            return res.map_goal
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def send_goal(self, pose_goal):
        goal = MoveBaseGoal()
        goal.target_pose = pose_goal
        self.move_base_client.send_goal(goal)

if __name__ == '__main__':
    print("Goal manager ready")
    try:
        goal_manager = GoalManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
