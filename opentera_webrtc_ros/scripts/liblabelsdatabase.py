from __future__ import annotations

import rospy
import sqlite3
from typing import Union
from pathlib import Path
from opentera_webrtc_ros_msgs.msg import Label
from geometry_msgs.msg import PoseStamped, Pose, Vector3, Quaternion
from std_msgs.msg import Header
from dataclasses import dataclass
from typing import Generator


@dataclass
class LabelData:
    label: Label
    node_id: int
    node_tf: Pose


class LabelsDb:
    def __init__(self, db_path: Union[str, Path]) -> None:
        self.path = Path(db_path).expanduser()
        self.db = LabelsDb._make_db(self.path)
        self._make_table()
        self.commit()

    def __getitem__(self, __name: str) -> LabelData:
        return self._get(__name)

    def __setitem__(self, __name: str, label: LabelData) -> None:
        self._set(__name, label)

    def __delitem__(self, __name: str) -> None:
        self.db.execute(
            "DELETE FROM OpenTeraLabels WHERE name=:name", {"name": __name})

    def __contains__(self, __name: str) -> bool:
        val = self.db.execute(
            "SELECT EXISTS (SELECT name FROM OpenTeraLabels WHERE name=:name)", {"name": __name}).fetchone()
        return bool(val[0])

    def get(self, name: str) -> LabelData:
        return self._get(name)

    def remove(self, name) -> None:
        if name not in self:
            raise IndexError(f"{name} not found in database")
        else:
            del self[name]

    def add(self, name: str, label: LabelData) -> None:
        if name in self:
            raise IndexError(f"{name} already exists in database")
        else:
            self._add(name, label)

    def rename(self, name: str, new_name: str) -> None:
        if name not in self:
            raise IndexError(f"{name} not found in database")
        elif new_name in self:
            raise IndexError(f"{new_name} already exists in database")
        else:
            self.db.execute("UPDATE OpenTeraLabels SET name=:new_name WHERE name=:name",
                            {"name": name, "new_name": new_name})

    def replace(self, name: str, label: LabelData) -> None:
        if name not in self:
            raise IndexError(f"{name} not found in database")
        else:
            self._edit(name, label)

    def commit(self) -> None:
        self.db.commit()

    def rollback(self) -> LabelsDb:
        self.db.rollback()
        return self

    def clear(self) -> LabelsDb:
        self._clear_table()
        self._make_table()
        return self

    def values(self) -> Generator[LabelData, None, None]:
        return (v for v in self.db.execute("SELECT * FROM OpenTeraLabels"))

    def _fetch(self, __name: str) -> sqlite3.Cursor:
        return self.db.execute("SELECT * FROM OpenTeraLabels WHERE name=:name",  {"name": __name})

    def _set(self, __name: str, label: LabelData) -> None:
        self._add(__name, label)
        self._edit(__name, label)

    def _edit(self, __name: str, label: LabelData) -> None:
        self.db.execute("""
        UPDATE OpenTeraLabels SET (
            name=:name,
            node_id=:node_id,
            description=:description,
            frame_id=:frame_id,
            position_x=:position_x,
            position_y=:position_y,
            position_z=:position_z,
            orientation_x=:orientation_x,
            orientation_y=:orientation_y,
            orientation_z=:orientation_z,
            orientation_w=:orientation_w,
            tf_position_x=:tf_position_x,
            tf_position_y=:tf_position_y,
            tf_position_z=:tf_position_z,
            tf_orientation_x=:tf_orientation_x,
            tf_orientation_y=:tf_orientation_y,
            tf_orientation_z=:tf_orientation_z,
            tf_orientation_w=:tf_orientation_w
        ) WHERE name=:old_name
        """, {
            "name": label.label.name,
            "node_id": label.node_id,
            "description": label.label.description,
            "frame_id": label.label.pose.header.frame_id,
            "position_x": label.label.pose.pose.position.x,
            "position_y": label.label.pose.pose.position.y,
            "position_z": label.label.pose.pose.position.z,
            "orientation_x": label.label.pose.pose.orientation.x,
            "orientation_y": label.label.pose.pose.orientation.y,
            "orientation_z": label.label.pose.pose.orientation.z,
            "orientation_w": label.label.pose.pose.orientation.w,
            "tf_position_x": label.node_tf.position.x,
            "tf_position_y": label.node_tf.position.y,
            "tf_position_z": label.node_tf.position.z,
            "tf_orientation_x": label.node_tf.orientation.x,
            "tf_orientation_y": label.node_tf.orientation.y,
            "tf_orientation_z": label.node_tf.orientation.z,
            "tf_orientation_w": label.node_tf.orientation.w,
            "old_name": __name,
        })

    def _add(self, __name: str, label: LabelData) -> None:
        self.db.execute("""
            INSERT OR IGNORE INTO OpenTeraLabels (
                name,
                node_id,
                description,
                frame_id,
                position_x,
                position_y,
                position_z,
                orientation_x,
                orientation_y,
                orientation_z,
                orientation_w,
                tf_position_x,
                tf_position_y,
                tf_position_z,
                tf_orientation_x,
                tf_orientation_y,
                tf_orientation_z,
                tf_orientation_w
            ) VALUES (
                :name,
                :node_id,
                :description,
                :frame_id,
                :position_x,
                :position_y,
                :position_z,
                :orientation_x,
                :orientation_y,
                :orientation_z,
                :orientation_w,
                :tf_position_x,
                :tf_position_y,
                :tf_position_z,
                :tf_orientation_x,
                :tf_orientation_y,
                :tf_orientation_z,
                :tf_orientation_w
            )
            """, {
            "name": __name,
            "node_id": label.node_id,
            "description": label.label.description,
            "frame_id": label.label.pose.header.frame_id,
            "position_x": label.label.pose.pose.position.x,
            "position_y": label.label.pose.pose.position.y,
            "position_z": label.label.pose.pose.position.z,
            "orientation_x": label.label.pose.pose.orientation.x,
            "orientation_y": label.label.pose.pose.orientation.y,
            "orientation_z": label.label.pose.pose.orientation.z,
            "orientation_w": label.label.pose.pose.orientation.w,
            "tf_position_x": label.node_tf.position.x,
            "tf_position_y": label.node_tf.position.y,
            "tf_position_z": label.node_tf.position.z,
            "tf_orientation_x": label.node_tf.orientation.x,
            "tf_orientation_y": label.node_tf.orientation.y,
            "tf_orientation_z": label.node_tf.orientation.z,
            "tf_orientation_w": label.node_tf.orientation.w,
        })

    def _get(self, __name: str) -> LabelData:
        val = self._fetch(__name).fetchone()
        if val is None:
            raise IndexError(f"{__name} not found in database")

        header = Header(frame_id=val["frame_id"])
        position = Vector3(x=val["position_x"],
                           y=val["position_y"], z=val["position_z"])
        orientation = Quaternion(
            x=val["orientation_x"], y=val["orientation_y"], z=val["orientation_z"], w=val["orientation_w"])
        pose = Pose(position=position, orientation=orientation)
        pose_stamped = PoseStamped(header=header, pose=pose)
        label = Label(name=val["name"], description=val["description"],
                      pose=pose_stamped)

        tf_position = Vector3(
            x=val["tf_position_x"], y=val["tf_position_y"], z=val["tf_position_z"])
        tf_orientation = Quaternion(
            x=val["tf_orientation_x"], y=val["tf_orientation_y"], z=val["tf_orientation_z"], w=val["tf_orientation_w"])
        node_tf = Pose(position=tf_position, orientation=tf_orientation)

        return LabelData(label=label, node_id=val["node_id"], node_tf=node_tf)

    @staticmethod
    def _make_db(sb_path: Path) -> sqlite3.Connection:

        for i in range(10):
            if not sb_path.exists():
                rospy.logwarn(
                    "Labels database not found. Trying again in 2 seconds.")
                rospy.sleep(2)
            else:
                break
        else:
            rospy.logerr(
                "Could not find labels database at {}".format(sb_path))
            raise FileNotFoundError

        db = sqlite3.connect(sb_path)
        db.row_factory = sqlite3.Row

        return db

    def _make_table(self) -> None:
        self.db.execute("""CREATE TABLE IF NOT EXISTS "OpenTeraLabels" (
            "name"                 TEXT NOT NULL UNIQUE,
            "node_id"              INTEGER NOT NULL,
            "description"          TEXT DEFAULT "",
            "frame_id"             TEXT DEFAULT "map",
            "position_x"           REAL DEFAULT 0,
            "position_y"           REAL DEFAULT 0,
            "position_z"           REAL DEFAULT 0,
            "orientation_x"        REAL DEFAULT 0,
            "orientation_y"        REAL DEFAULT 0,
            "orientation_z"        REAL DEFAULT 0,
            "orientation_w"        REAL DEFAULT 0,
            "tf_position_x"        REAL DEFAULT 0,
            "tf_position_y"        REAL DEFAULT 0,
            "tf_position_z"        REAL DEFAULT 0,
            "tf_orientation_x"     REAL DEFAULT 0,
            "tf_orientation_y"     REAL DEFAULT 0,
            "tf_orientation_z"     REAL DEFAULT 0,
            "tf_orientation_w"     REAL DEFAULT 0,
            PRIMARY KEY("name")
        )""")

    def _clear_table(self) -> None:
        self.db.execute("DROP TABLE IF EXISTS OpenTeraLabels")
