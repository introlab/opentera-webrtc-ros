#!/usr/bin/env python3

import rclpy
import rclpy.exceptions
import rclpy.node
import rclpy.timer
import json
from pathlib import Path
from opentera_webrtc_ros_msgs.msg import LabelSimple, LabelSimpleEdit
from opentera_webrtc_ros_msgs.msg import Label, LabelArray
import rclpy.timer
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray, Marker
from opentera_webrtc_ros.libmapimageconverter import PoseWaypointConverter
from opentera_webrtc_ros.libyamldatabase import YamlDatabase
from opentera_webrtc_ros.libnavigation import WaypointNavigationClient
from typing import Callable


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


class LabelsManager(rclpy.node.Node):
    def __init__(self) -> None:
        super().__init__("labels_manager")  # type: ignore

        self._pose_waypoint_converter = PoseWaypointConverter(self)

        self.add_label_simple_sub = self.create_subscription(
            LabelSimple, "add_label_simple", self.add_label_simple_callback, 1)
        self.remove_label_by_name_sub = self.create_subscription(
            String, "remove_label_by_name", self.remove_label_by_name_callback, 1)
        self.edit_label_simple_sub = self.create_subscription(
            LabelSimpleEdit, "edit_label_simple", self.edit_label_simple_callback, 1)
        self.navigate_to_label_sub = self.create_subscription(
            String, "navigate_to_label", self.navigate_to_label_callback, 1)

        self.stored_labels_pub = self.create_publisher(
            LabelArray, "stored_labels", 1)
        self.stored_labels_text_pub = self.create_publisher(
            String, "stored_labels_text", 1)
        self.stored_labels_marker_pub = self.create_publisher(
            MarkerArray, "stored_labels_marker", 1)

        self.database_path: str = self.declare_parameter("~database_path", "~/.ros/labels.yaml").get_parameter_value().string_value
        self.db: YamlDatabase[LabelData] = YamlDatabase(
            Path(self.database_path), LabelData)

        self.pub_timer_stored_labels = self.create_timer(1, self.publish_stored_labels)
        self.pub_timer_stored_labels_text = self.create_timer(1, self.publish_stored_labels_text)
        self.pub_timer_stored_labels_marker = self.create_timer(1, self.publish_stored_labels_marker)

        self.nav_client = WaypointNavigationClient(self)

        self.get_logger().info("Labels manager initialized")

    def publish_stored_labels_text(self) -> None:
        labels_text = [
            {"name": e.label.name, "description": e.label.description} for e in self.db.values()]
        labels_text_json_message = {
            "type": "labels", "labels": labels_text}
        labels_text_msg = json.dumps(labels_text_json_message)
        self.stored_labels_text_pub.publish(String(data=labels_text_msg))

    def publish_stored_labels_marker(self) -> None:
        markers = MarkerArray()
        markers.markers = [self._get_marker_from_label(e.label, i)
                           for i, e in enumerate(self.db.values())]
        self.stored_labels_marker_pub.publish(markers)

    def publish_stored_labels(self) -> None:
        labels = tuple(e.label for e in self.db.values())
        self.stored_labels_pub.publish(LabelArray(labels=labels))

    def add_label_simple_callback(self, msg: LabelSimple) -> None:
        def cb(label: Label):
            try:
                self.db.add(msg.name, LabelData(label))
                self.db.commit()
            except IndexError as e:
                self.get_logger().error(f"Adding label to database failed: {e}")

        try:
            self._simple2label(msg, cb)
        except ConversionError as e:
            self.get_logger().error(f"Adding label to database failed: {e}")

    def remove_label_by_name_callback(self, msg: String) -> None:
        try:
            self.db.remove(msg.data)
            self.db.commit()
        except IndexError as e:
            self.get_logger().error(f"Removing label from database failed: {e}")

    def edit_label_simple_callback(self, msg: LabelSimpleEdit) -> None:
        def update_label(label: Label):
            try:
                if msg.current_name != msg.updated.name:
                    self.db.rename(msg.current_name, msg.updated.name)
    
                if msg.ignore_waypoint is True:
                    label.pose = self.db[msg.updated.name].label.pose
                self.db.replace(msg.updated.name, LabelData(label))

                self.db.commit()
            except IndexError as e:
                self.get_logger().error(f"Editing label in database failed: {e}")
        
        try:
            self._simple2label(msg.updated, update_label)
        except ConversionError as e:
            self.get_logger().error(f"Editing label in database failed: {e}")

    def navigate_to_label_callback(self, msg: String) -> None:
        if not msg.data in self.db:
            self.get_logger().error(
                f"Navigation to label failed: Label [{msg.data}] not in database")
            return

        label = self.db[msg.data].label
        self.nav_client.add_to_image(label.pose)
        self.nav_client.navigate_through_goals([label.pose])

    # def _label2simple(self, label: Label, callback: Callable) -> None:
    #     def cb(waypoint: Waypoint):
    #         label_simple = LabelSimple(name=label.name, description=label.description, waypoint=waypoint)
    #         callback(label_simple)

    #     if self._pose_waypoint_converter.convert_pose_to_waypoint(label.pose, cb) is None:
    #         raise ConversionError(
    #             f"Conversion of pose to waypoint for label {label.name} failed")

    def _simple2label(self, label_simple: LabelSimple, callback: Callable) -> None:
        def cb(pose: PoseStamped):
            label = Label(name=label_simple.name, description=label_simple.description, pose=pose)
            callback(label)

        if self._pose_waypoint_converter.convert_waypoint_to_pose(label_simple.waypoint, cb) is None:
            raise ConversionError(
                f"Conversion of waypoint to pose for label {label_simple.name} failed")

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


def main():
    rclpy.init()
    labels_manager = LabelsManager()
    
    try:
        rclpy.spin(labels_manager)
    except KeyboardInterrupt:
        pass

    labels_manager.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
