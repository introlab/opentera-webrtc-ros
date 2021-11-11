#!/usr/bin/env python3

import rospy
import actionlib
import json
from math import pi, sqrt
from tf.transformations import quaternion_from_euler, quaternion_multiply, quaternion_conjugate
from tf import TransformListener, LookupException, ExtrapolationException
from geometry_msgs.msg import PoseStamped
from opentera_webrtc_ros_msgs.msg import WaypointArray
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from map_image_generator.srv import ImageGoalToMapGoal
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool
from fiducial_msgs.msg import FiducialTransformArray

class Fiducial:
    def __init__(self, id, pose):
        self.id = id
        self.pose = pose

class GoalManager():
    def __init__(self):
        rospy.init_node("goal_manager") 

        self.tf = TransformListener()

        # TODO: get frames with parameters
        self.camera_frame = "d455_color_optical_frame"
        self.map_frame = "map"
        self.fiducial_distance_tolerance = 0.25  # m

        # Global action client used to send waypoints to move_base
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.move_base_client.wait_for_server()
        # Make sure no goals are active
        self.move_base_client.cancel_all_goals()

        # Subscribers and publishers
        self.waypoints_sub = rospy.Subscriber("waypoints", WaypointArray, self.waypoints_cb)
        self.stop_sub = rospy.Subscriber("stop", Bool, self.stop_cb)
        self.dock_action_sub = rospy.Subscriber("dock_action", Bool, self.dock_cb)
        self.fiducial_sub = rospy.Subscriber("fiducial_transforms", FiducialTransformArray, self.fiducial_cb)
        self.waypoint_reached_pub = rospy.Publisher("waypoint_reached", String, queue_size=1)
        self.dock_fiducial_pose_pub = rospy.Publisher("dock_fiducial_pose", PoseStamped, queue_size=1)  # For debugging
        self.pre_docking_pose_pub = rospy.Publisher("pre_docking_pose", PoseStamped, queue_size=1)  # For debugging

        self.should_stop = False

        # Dock pose in map
        self.dock_pose = PoseStamped()
        self.dock_pose.header.frame_id = "map"
        self.dock_pose.pose.position.x = 0
        self.dock_pose.pose.position.y = 0
        self.dock_pose.pose.position.z = 0
        self.dock_pose.pose.orientation.x = 0
        self.dock_pose.pose.orientation.y = 0
        self.dock_pose.pose.orientation.z = 0
        self.dock_pose.pose.orientation.w = 1

        self.dock_fiducial = Fiducial(-1, None)

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

    def fiducial_cb(self, msg):
        if len(msg.transforms) == 0:
            return
        print("Fiducial found")
        self.move_base_client.cancel_all_goals()
        for fiducial in msg.transforms:
            pose_map_frame = self.transform_to_map_frame(fiducial.transform, msg.header)
            if pose_map_frame == None:
                rospy.logwarn("Failed to transform fiducial pose from %s to %s" % (self.camera_frame, self.map_frame))
                continue

            current_fiducial = Fiducial(fiducial.fiducial_id, pose_map_frame)
            if self.dock_fiducial.id == -1:
                # First time seeing the fiducial
                self.dock_fiducial = current_fiducial
            else:
                # Not first time seeing a fiducial, check if it is the same one
                is_same = self.is_same_fiducial(self.dock_fiducial, current_fiducial)
                if not is_same:
                    # TODO: handle this case
                    rospy.logwarn("Current fiducial is not the same as the first one")
                else:
                    print("Same fiducial")
                    self.dock_fiducial_pose_pub.publish(current_fiducial.pose)

                    pre_docking_pose = self.calculate_pre_docking_pose(current_fiducial.pose)
                    self.pre_docking_pose_pub.publish(pre_docking_pose)
                    

    def is_same_fiducial(self, f1, f2):
        # TODO: compare orientation too
        dist = sqrt((f1.pose.pose.position.x - f2.pose.pose.position.x) ** 2 + \
                    (f1.pose.pose.position.y - f2.pose.pose.position.y) ** 2 + \
                    (f1.pose.pose.position.z - f2.pose.pose.position.z) ** 2)
        if f1.id == f2.id and dist < self.fiducial_distance_tolerance:
            return True
        else:
            return False

    def transform_to_map_frame(self, transform, header):
        pose_cam_frame = self.transform_to_pose_stamped(transform, header)
        try:
            pose_map_frame = self.tf.transformPose(self.map_frame, pose_cam_frame)
            return pose_map_frame
        except (LookupException, ExtrapolationException):
            return None

    def transform_to_pose_stamped(self, transform, header):
        pose = PoseStamped()
        pose.header = header
        pose.pose.position.x = transform.translation.x
        pose.pose.position.y = transform.translation.y
        pose.pose.position.z = transform.translation.z
        pose.pose.orientation.x = transform.rotation.x
        pose.pose.orientation.y = transform.rotation.y
        pose.pose.orientation.z = transform.rotation.z
        pose.pose.orientation.w = transform.rotation.w
        return pose

    def calculate_pre_docking_pose(self, dock_fiducial_pose):
        # Rotate 90 degrees so that the x axis points towards the dock
        # Translate back by a certain amount before the dock
        # TODO: backward translation should be a parameter
        pre_docking_pose = dock_fiducial_pose
        rotation = quaternion_from_euler(0, 0, pi/2)
        fiducial_orientation = [dock_fiducial_pose.pose.orientation.x, \
                                dock_fiducial_pose.pose.orientation.y, \
                                dock_fiducial_pose.pose.orientation.z, \
                                dock_fiducial_pose.pose.orientation.w]
        resulting_orientation = quaternion_multiply(fiducial_orientation, rotation)
        pre_docking_pose.pose.orientation.x = resulting_orientation[0]
        pre_docking_pose.pose.orientation.y = resulting_orientation[1]
        pre_docking_pose.pose.orientation.z = resulting_orientation[2]
        pre_docking_pose.pose.orientation.w = resulting_orientation[3]

        v1 = [-0.5, 0, 0, 1]
        v2 = quaternion_multiply(quaternion_multiply(resulting_orientation, v1), \
                                    quaternion_conjugate(resulting_orientation))
    
        pre_docking_pose.pose.position.x += v2[0]
        pre_docking_pose.pose.position.y += v2[1]
        pre_docking_pose.pose.position.z += v2[2]

        return pre_docking_pose


if __name__ == '__main__':
    print("Goal manager ready")
    try:
        goal_manager = GoalManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
