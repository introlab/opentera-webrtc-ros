#!/usr/bin/env python3

import rospy
import actionlib
import json
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_srvs.srv import SetBool
from std_msgs.msg import Bool, String
from geometry_msgs.msg import PoseStamped
from typing import Optional, Generator, Callable


class WaypointNavigationClient:
    def __init__(self, stop_cb: Optional[Callable[..., Generator]] = None, stop_token=None) -> None:
        self.move_base_client = actionlib.SimpleActionClient(
            'move_base', MoveBaseAction)
        self.move_base_client.wait_for_server()

        self.stop_sub = rospy.Subscriber(
            "stop", Bool, self._stop_callback, queue_size=1)

        self.waypoint_reached_pub = rospy.Publisher(
            "waypoint_reached", String, queue_size=1)
        self.remove_waypoint_from_image_pub = rospy.Publisher(
            "map_image_drawer/remove_goal", PoseStamped, queue_size=10)
        self.add_waypoint_to_image_pub = rospy.Publisher(
            "map_image_drawer/add_goal", PoseStamped, queue_size=10)

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
        goal = MoveBaseGoal()
        goal.target_pose = pose_goal
        self.stop = False
        self.move_base_client.send_goal(goal)
        self.move_base_client.wait_for_result()
        state = self.move_base_client.get_state()
        if not self.stop and not self.stop_token(state):
            self._publish_waypoint_reached(reached_index, pose_goal)

    def cancel_all_goals(self, clear_goals=True):
        self.move_base_client.cancel_all_goals()
        if (clear_goals):
            rospy.wait_for_service('map_image_drawer/clear_goals')
            try:
                image_goal_clear_waypoints = rospy.ServiceProxy(
                    'map_image_drawer/clear_goals', SetBool)
                image_goal_clear_waypoints(True)
            except rospy.ServiceException as e:
                rospy.logwarn("Service call failed: %s" % e)

    def clear_global_path(self):
        rospy.wait_for_service('clear_global_path')
        try:
            clear_global_path = rospy.ServiceProxy(
                'clear_global_path', SetBool)
            clear_global_path(True)
        except rospy.ServiceException as e:
            rospy.logwarn("Service call failed: %s" % e)

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
