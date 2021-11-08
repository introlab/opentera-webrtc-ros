#!/usr/bin/env python3

import rospy
import actionlib
import json
from math import pi
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import PoseStamped
from opentera_webrtc_ros_msgs.msg import WaypointArray
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from map_image_generator.srv import ImageGoalToMapGoal
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool

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
        self.stop_sub = rospy.Subscriber("stop", Bool, self.stop_cb)
        self.dock_action_sub = rospy.Subscriber("dock_action", Bool, self.dock_cb)
        self.waypoint_reached_pub = rospy.Publisher("waypoint_reached", String, queue_size=1)

        self.should_stop = False

        # Dock pose in map
        self.dock_pose = PoseStamped()
        self.dock_pose.header.frame_id = "map"
        self.dock_pose.pose.position.x = 5
        self.dock_pose.pose.position.y = 3
        self.dock_pose.pose.position.z = 0
        self.dock_pose.pose.orientation.x = 0
        self.dock_pose.pose.orientation.y = 0
        self.dock_pose.pose.orientation.z = 0
        self.dock_pose.pose.orientation.w = 1

    def waypoints_cb(self, msg):
        self.move_base_client.cancel_all_goals()
        self.should_stop = False
        i = 1
        for waypoint in msg.waypoints:
            pose_goal = self.transform_waypoint_to_pose(waypoint)
            self.send_goal(pose_goal)
            if self.should_stop:
                break
            else:
                self.publishWaypointReached(i)
                i += 1
        self.clearGlobalPathClient()

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
        self.move_base_client.wait_for_result()

    def stop_cb(self, msg):
        if msg.data == True:
            self.should_stop = True
            self.move_base_client.cancel_all_goals()
            self.clearGlobalPathClient()

    def dock_cb(self, msg):
        if msg.data == True:
            self.send_goal(self.dock_pose)
        else:
            self.move_base_client.cancel_all_goals()
            self.clearGlobalPathClient()

    def publishWaypointReached(self, i):
        waypoint_reached_json_message = {"type": "waypointReached", "waypointNumber": i}
        waypoint_reached_msg = json.dumps(waypoint_reached_json_message)
        self.waypoint_reached_pub.publish(waypoint_reached_msg)

    def clearGlobalPathClient(self):
        rospy.wait_for_service('clear_global_path')
        try:
            clear_global_path = rospy.ServiceProxy('clear_global_path', SetBool)
            clear_global_path(True)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)


if __name__ == '__main__':
    print("Goal manager ready")
    try:
        goal_manager = GoalManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
