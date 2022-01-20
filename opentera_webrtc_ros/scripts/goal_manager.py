#!/usr/bin/env python3

import rospy
import actionlib
import json
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import PoseStamped
from opentera_webrtc_ros_msgs.msg import WaypointArray
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool
from libmapimageconverter import convert_waypoint_to_pose


class GoalManager():
    def __init__(self):

        # Global action client used to send waypoints to move_base
        self.move_base_client = actionlib.SimpleActionClient(
            'move_base', MoveBaseAction)
        self.move_base_client.wait_for_server()
        # Make sure no goals are active
        self.cancelAllGoalsClient()

        self.should_stop = False
        self.pose_goals = []

        # Subscribers and publishers
        self.waypoints_sub = rospy.Subscriber(
            "waypoints", WaypointArray, self.waypoints_cb)
        self.stop_sub = rospy.Subscriber("stop", Bool, self.stop_cb)
        self.start_sub = rospy.Subscriber("start", Bool, self.start_cb)
        self.waypoint_reached_pub = rospy.Publisher(
            "waypoint_reached", String, queue_size=1)
        self.remove_waypoint_from_image_pub = rospy.Publisher(
            "map_image_drawer/remove_goal", PoseStamped, queue_size=10)
        self.add_waypoint_to_image_pub = rospy.Publisher(
            "map_image_drawer/add_goal", PoseStamped, queue_size=10)

        rospy.loginfo("Goal manager ready")

    def waypoints_cb(self, msg):
        for waypoint in msg.waypoints:
            pose_goal = convert_waypoint_to_pose(waypoint)
            if pose_goal is not None:
                pose_goal.pose.position.z = 1 + len(self.pose_goals)
                self.add_waypoint_to_image_pub.publish(pose_goal)
                self.pose_goals.append(pose_goal)

    def send_goal(self, pose_goal):
        goal = MoveBaseGoal()
        goal.target_pose = pose_goal
        self.move_base_client.send_goal(goal)
        self.move_base_client.wait_for_result()
        return self.move_base_client.get_state()

    def stop_cb(self, msg):
        rospy.loginfo("Stopping")
        if msg.data == True:
            self.should_stop = True
            self.cancelAllGoalsClient()
            self.clearGlobalPathClient()
            self.pose_goals.clear()

    def start_cb(self, msg):
        rospy.loginfo("Starting")
        if msg.data == True:
            self.cancelAllGoalsClient(False)
            self.clearGlobalPathClient()
            self.should_stop = False

            for pose_goal in self.pose_goals:
                self.send_goal(pose_goal)
                if self.should_stop:
                    break
                else:
                    self.publishWaypointReached(
                        round(pose_goal.pose.position.z), pose_goal)
            self.clearGlobalPathClient()
            self.pose_goals.clear()

    def publishWaypointReached(self, i, goal_pose):
        waypoint_reached_json_message = {
            "type": "waypointReached", "waypointNumber": i}
        waypoint_reached_msg = json.dumps(waypoint_reached_json_message)
        self.waypoint_reached_pub.publish(waypoint_reached_msg)
        self.remove_waypoint_from_image_pub.publish(goal_pose)

    def cancelAllGoalsClient(self, clear_goals=True):
        self.move_base_client.cancel_all_goals()
        if (clear_goals):
            rospy.wait_for_service('map_image_drawer/clear_goals')
            try:
                image_goal_clear_waypoints = rospy.ServiceProxy(
                    'map_image_drawer/clear_goals', SetBool)
                res = image_goal_clear_waypoints(True)
            except rospy.ServiceException as e:
                rospy.logwarn("Service call failed: %s" % e)

    def clearGlobalPathClient(self):
        rospy.wait_for_service('clear_global_path')
        try:
            clear_global_path = rospy.ServiceProxy(
                'clear_global_path', SetBool)
            clear_global_path(True)
        except rospy.ServiceException as e:
            rospy.logwarn("Service call failed: %s" % e)


if __name__ == '__main__':
    rospy.init_node("goal_manager")
    try:
        goal_manager = GoalManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
