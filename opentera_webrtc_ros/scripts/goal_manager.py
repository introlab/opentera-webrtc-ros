#!/usr/bin/env python3

import rospy
import actionlib
import json
import dynamic_reconfigure.client
from math import pi, sqrt
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import PoseStamped, Twist
from opentera_webrtc_ros_msgs.msg import WaypointArray
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from map_image_generator.srv import ImageGoalToMapGoal
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool

class GoalManager():
    def __init__(self):  
        self.dr_client = dynamic_reconfigure.client.Client("/move_base/DWAPlannerROS")

        # Global action client used to send waypoints to move_base
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.move_base_client.wait_for_server()
        # Make sure no goals are active
        self.move_base_client.cancel_all_goals()

        # Subscribers and publishers
        self.waypoints_sub = rospy.Subscriber("waypoints", WaypointArray, self.waypoints_cb)
        self.stop_sub = rospy.Subscriber("stop", Bool, self.stop_cb)
        self.pre_docking_pose_sub = rospy.Subscriber("pre_docking_pose", PoseStamped, self.pre_docking_pose_cb)
        self.waypoint_reached_pub = rospy.Publisher("waypoint_reached", String, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)

        self.should_stop = False

        rospy.loginfo("Goal manager ready")

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
            rospy.logwarn("Service call failed: %s"%e)

    def send_goal(self, pose_goal):
        goal = MoveBaseGoal()
        goal.target_pose = pose_goal
        self.move_base_client.send_goal(goal)
        self.move_base_client.wait_for_result()

    def stop_cb(self, msg):
        rospy.loginfo("Stopping")
        if msg.data == True:
            self.should_stop = True
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
            rospy.logwarn("Service call failed: %s" % e)
    
    def pre_docking_pose_cb(self, pose):
        # Need to change goal tolerances to something small to make sure the docking is precise.
        rospy.loginfo("Got pre-docking pose")
        prev_dwa_config = self.dr_client.get_configuration()
        new_dwa_config = prev_dwa_config.copy()
        new_dwa_config["xy_goal_tolerance"] = 0.06
        new_dwa_config["yaw_goal_tolerance"] = 0.0349066
        self.dr_client.update_configuration(new_dwa_config)
        rospy.loginfo("Changed goal tolerances")
        rospy.loginfo("Sending goal to move_base")
        self.should_stop = False
        self.send_goal(pose)
        # Reset DWA config to what it was previously
        self.dr_client.update_configuration(prev_dwa_config)

        # Go backwards until connected to charger
        # TODO: refactor. Should this be handled by move_base? If not should it be in docking.py instead?
        # TODO: should not use timer for determining when to stop backing up
        if not self.should_stop:
            cmd = Twist()
            cmd.linear.x = -0.1
            r = rospy.Rate(10)
            start_time = rospy.Time.now()
            elapsed_time = 0
            max_time = 3
            rospy.loginfo("Backing for %.1f seconds" % max_time)
            # TODO: find better way of knowing when to stop backing up
            while not rospy.is_shutdown() and elapsed_time < max_time:
                self.cmd_vel_pub.publish(cmd)
                elapsed_time = (rospy.Time.now() - start_time).to_sec()
                r.sleep()
        

if __name__ == '__main__':
    rospy.init_node("goal_manager")
    try:
        goal_manager = GoalManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
