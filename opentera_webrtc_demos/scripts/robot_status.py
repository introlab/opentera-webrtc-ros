#!/usr/bin/env python

import rospy
import json
from std_msgs.msg import String

class RobotStatus():
    def __init__(self):
        rospy.init_node("robot_status")
        self.status_pub = rospy.Publisher('/robot_status', String, queue_size=0)
        self.pub_rate = 1

    def run(self):
        r = rospy.Rate(self.pub_rate)
        while not rospy.is_shutdown():
            for i in range(100, -1, -5):
                battery_json_message = {"type": "batteryStatus", "level": i}
                signal_json_message = {"type": "signalStatus", "strength": i}
                battery_serialized_msg = json.dumps(battery_json_message)
                signal_serialized_msg = json.dumps(signal_json_message)
                self.status_pub.publish(battery_serialized_msg)
                self.status_pub.publish(signal_serialized_msg)
                print("published msg")
                r.sleep()


if __name__ == '__main__':
    print("Robot Status Ready")
    try:
        robot_status = RobotStatus()
        robot_status.run()
    except rospy.ROSInterruptException:
        pass
