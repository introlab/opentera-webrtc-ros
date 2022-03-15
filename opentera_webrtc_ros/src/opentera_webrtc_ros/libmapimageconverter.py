#!/usr/bin/env python3

import rospy
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import PoseStamped
from opentera_webrtc_ros_msgs.msg import Waypoint
from typing import Optional
from map_image_generator.srv import ImageGoalToMapGoal, ImageGoalToMapGoalResponse
# from map_image_generator.srv import MapGoalToImageGoal, MapGoalToImageGoalResponse


def convert_pose_to_waypoint(pose: PoseStamped) -> Optional[Waypoint]:
    waypoint_pose = _map_goal_to_image_goal_client(pose)
    if waypoint_pose is None:
        return None
    waypoint = Waypoint()
    waypoint.x = waypoint_pose.pose.position.x
    waypoint.y = waypoint_pose.pose.position.y
    waypoint.yaw = euler_from_quaternion(waypoint_pose.pose.orientation)[2]
    return waypoint


def convert_waypoint_to_pose(waypoint: Waypoint) -> Optional[PoseStamped]:
    pose = PoseStamped()
    pose.header.frame_id = "map"
    pose.pose.position.x = waypoint.x
    pose.pose.position.y = waypoint.y
    pose.pose.position.z = 0
    yaw = -waypoint.yaw
    quaternion = quaternion_from_euler(0, 0, yaw)
    pose.pose.orientation.x = quaternion[0]
    pose.pose.orientation.y = quaternion[1]
    pose.pose.orientation.z = quaternion[2]
    pose.pose.orientation.w = quaternion[3]

    return _image_goal_to_map_goal_client(pose)


def _map_goal_to_image_goal_client(map_goal: PoseStamped) -> Optional[PoseStamped]:
    pass
    # rospy.wait_for_service('map_goal_to_image_goal')
    # try:
    #     map_goal_to_image_goal = rospy.ServiceProxy(
    #         'map_goal_to_image_goal', MapGoalToImageGoal)
    #     res: MapGoalToImageGoalResponse = map_goal_to_image_goal(
    #         map_goal)
    #     return res.map_goal
    # except rospy.ServiceException as e:
    #     rospy.logwarn("Service call failed: %s" % e)


def _image_goal_to_map_goal_client(image_goal: PoseStamped) -> Optional[PoseStamped]:
    rospy.wait_for_service('image_goal_to_map_goal')
    try:
        image_goal_to_map_goal = rospy.ServiceProxy(
            'image_goal_to_map_goal', ImageGoalToMapGoal)
        res: ImageGoalToMapGoalResponse = image_goal_to_map_goal(
            image_goal)
        return res.map_goal
    except rospy.ServiceException as e:
        rospy.logwarn("Service call failed: %s" % e)
