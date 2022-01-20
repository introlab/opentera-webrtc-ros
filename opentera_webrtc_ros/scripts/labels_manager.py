#!/usr/bin/env python3

from __future__ import annotations

import rospy
from rospy_message_converter.message_converter import convert_ros_message_to_dictionary as ros2dict
from rospy_message_converter.message_converter import convert_dictionary_to_ros_message as dict2ros
from opentera_webrtc_ros_msgs.msg import LabelSimple, LabelSimpleArray, LabelSimpleEdit
from opentera_webrtc_ros_msgs.msg import Label, LabelArray, LabelEdit
from opentera_webrtc_ros_msgs.msg import Waypoint
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from libmapimageconverter import convert_waypoint_to_pose as wp2pose
from libmapimageconverter import convert_pose_to_waypoint as pose2wp

from libyamldatabase import YamlDatabase
from pathlib import Path
from typing import cast


class ConversionError(Exception):
    pass


class LabelData:
    yaml_tag = "!label"

    def __init__(self, label: Label) -> None:
        self.data = ros2dict(label)

    @property
    def label(self) -> Label:
        pose = PoseStamped()
        pose.header.frame_id = str(self.data["pose"]["header"]["frame_id"])
        pose.pose.position.x = float(
            self.data["pose"]["pose"]["position"]["x"])
        pose.pose.position.y = float(
            self.data["pose"]["pose"]["position"]["y"])
        pose.pose.position.z = float(
            self.data["pose"]["pose"]["position"]["z"])
        pose.pose.orientation.x = float(
            self.data["pose"]["pose"]["orientation"]["x"])
        pose.pose.orientation.y = float(
            self.data["pose"]["pose"]["orientation"]["y"])
        pose.pose.orientation.z = float(
            self.data["pose"]["pose"]["orientation"]["z"])
        pose.pose.orientation.w = float(
            self.data["pose"]["pose"]["orientation"]["w"])

        return Label(name=self.data["name"], description=self.data["description"], pose=pose)


class LabelsManager:
    def __init__(self) -> None:

        self.add_label_simple_sub = rospy.Subscriber(
            "add_label_simple", LabelSimple, self.add_label_simple_callback, queue_size=1)
        self.remove_label_by_name_sub = rospy.Subscriber(
            "remove_label_by_name", String, self.remove_label_by_name_callback, queue_size=1)
        self.edit_label_simple_sub = rospy.Subscriber(
            "edit_label_simple", LabelSimpleEdit, self.edit_label_simple_callback, queue_size=1)

        self.stored_labels_pub = rospy.Publisher(
            "stored_labels", LabelArray, queue_size=1)

        self.database_path: str = rospy.get_param(
            "~database_path", "~/.ros/labels.yaml")
        self.db: YamlDatabase[LabelData] = YamlDatabase(
            Path(self.database_path), LabelData)

        self.pub_timer = rospy.Timer(rospy.Duration(
            1), self.publish_stored_labels_simple)

        rospy.loginfo("Labels manager initialized")

    def publish_stored_labels_simple(self, _: rospy.timer.TimerEvent) -> None:
        labels = tuple(e.label for e in self.db.values())
        self.stored_labels_pub.publish(labels)

    def add_label_simple_callback(self, msg: LabelSimple) -> None:
        try:
            label = LabelData(self.simple2label(msg))
            self.db.add(msg.name, label)
            self.db.commit()
        except (IndexError, ConversionError) as e:
            rospy.logerr(f"Adding label to database failed: {e}")

    def remove_label_by_name_callback(self, msg: String) -> None:
        try:
            self.db.remove(msg.data)
            self.db.commit()
        except IndexError as e:
            rospy.logerr(f"Removing label from database failed: {e}")

    def edit_label_simple_callback(self, msg: LabelSimpleEdit) -> None:
        try:
            self.db.replace(msg.current_name, LabelData(
                self.simple2label(msg.updated)))
            if msg.current_name != msg.updated.name:
                self.db.rename(msg.current_name, msg.updated.name)
            self.db.commit()
        except (IndexError, ConversionError) as e:
            rospy.logerr(f"Editing label in database failed: {e}")

    @staticmethod
    def label2simple(label: Label) -> LabelSimple:
        waypoint = pose2wp(label.pose)
        if waypoint is None:
            raise ConversionError(
                f"Conversion of pose to waypoint for label {label.name} failed")
        return LabelSimple(name=label.name, description=label.description, waypoint=waypoint)

    @staticmethod
    def simple2label(label_simple: LabelSimple) -> Label:
        pose = wp2pose(label_simple.waypoint)
        if pose is None:
            raise ConversionError(
                f"Conversion of waypoint to pose for label {label_simple.name} failed")
        return Label(name=label_simple.name, description=label_simple.description, pose=pose)


rospy.loginfo("Labels manager ready")


if __name__ == '__main__':
    rospy.init_node("labels_manager")
    try:
        labels_manager = LabelsManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
