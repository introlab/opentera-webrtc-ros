#!/usr/bin/env python3

import rospy
import json
from pathlib import Path
from opentera_webrtc_ros_msgs.msg import LabelSimple, LabelSimpleArray, LabelSimpleEdit
from opentera_webrtc_ros_msgs.msg import Label, LabelArray, LabelEdit
from opentera_webrtc_ros_msgs.msg import Waypoint
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray, Marker
from opentera_webrtc_ros.libmapimageconverter import convert_waypoint_to_pose as wp2pose
from opentera_webrtc_ros.libmapimageconverter import convert_pose_to_waypoint as pose2wp
from opentera_webrtc_ros.libyamldatabase import YamlDatabase
from opentera_webrtc_ros.libnavigation import WaypointNavigationClient


class ConversionError(Exception):
    pass


class LabelData:
    yaml_tag = "!label"

    def __init__(self, label: Label) -> None:
        self.data = {
            "name": label.name,
            "description": label.description,
            "pose": {
                "header": {
                    "frame_id": label.pose.header.frame_id,
                },
                "pose": {
                    "position": {
                        "x": label.pose.pose.position.x,
                        "y": label.pose.pose.position.y,
                        "z": label.pose.pose.position.z,
                    },
                    "orientation": {
                        "x": label.pose.pose.orientation.x,
                        "y": label.pose.pose.orientation.y,
                        "z": label.pose.pose.orientation.z,
                        "w": label.pose.pose.orientation.w,
                    },
                },
            },
        }

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
        self.navigate_to_label_sub = rospy.Subscriber(
            "navigate_to_label", String, self.navigate_to_label_callback, queue_size=1)

        self.stored_labels_pub = rospy.Publisher(
            "stored_labels", LabelArray, queue_size=1)
        self.stored_labels_text_pub = rospy.Publisher(
            "stored_labels_text", String, queue_size=1)
        self.stored_labels_marker_pub = rospy.Publisher(
            "stored_labels_marker", MarkerArray, queue_size=1)

        self.database_path: str = rospy.get_param(
            "~database_path", "~/.ros/labels.yaml")
        self.db: YamlDatabase[LabelData] = YamlDatabase(
            Path(self.database_path), LabelData)

        self.pub_timer_stored_labels = rospy.Timer(rospy.Duration(
            1), self.publish_stored_labels)
        self.pub_timer_stored_labels_text = rospy.Timer(rospy.Duration(
            1), self.publish_stored_labels_text)
        self.pub_timer_stored_labels_marker = rospy.Timer(rospy.Duration(
            1), self.publish_stored_labels_marker)

        self.nav_client = WaypointNavigationClient()

        rospy.loginfo("Labels manager initialized")

    def publish_stored_labels_text(self, _: rospy.timer.TimerEvent) -> None:
        labels_text = [
            {"name": e.label.name, "description": e.label.description} for e in self.db.values()]
        labels_text_json_message = {
            "type": "labels", "labels": labels_text}
        labels_text_msg = json.dumps(labels_text_json_message)
        self.stored_labels_text_pub.publish(labels_text_msg)

    def publish_stored_labels_marker(self, _: rospy.timer.TimerEvent) -> None:
        markers = MarkerArray()
        markers.markers = [self._get_marker_from_label(e.label, i)
                           for i, e in enumerate(self.db.values())]
        self.stored_labels_marker_pub.publish(markers)

    def publish_stored_labels(self, _: rospy.timer.TimerEvent) -> None:
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
            if msg.current_name != msg.updated.name:
                self.db.rename(msg.current_name, msg.updated.name)

            updated = self.simple2label(msg.updated)
            if msg.ignore_waypoint is True:
                updated.pose = self.db[msg.updated.name].label.pose
            self.db.replace(msg.updated.name, LabelData(updated))

            self.db.commit()
        except (IndexError, ConversionError) as e:
            rospy.logerr(f"Editing label in database failed: {e}")

    def navigate_to_label_callback(self, msg: String) -> None:
        if not msg.data in self.db:
            rospy.logerr(
                f"Navigation to label failed: Label [{msg.data}] not in database")
            return

        label = self.db[msg.data].label
        self.nav_client.add_to_image(label.pose)
        self.nav_client.navigate_to_goal(label.pose, 1)

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

    def _get_marker_from_label(self, label: Label, id: int) -> Marker:
        marker = Marker()
        marker.header = label.pose.header
        marker.ns = "labels"
        marker.id = id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose = label.pose.pose
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 255.0
        marker.color.r = 255.0
        marker.color.g = 0.0
        marker.color.b = 255.0
        return marker


rospy.loginfo("Labels manager ready")


if __name__ == '__main__':
    rospy.init_node("labels_manager")
    try:
        labels_manager = LabelsManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
