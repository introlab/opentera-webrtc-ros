#!/usr/bin/env python3

import rospy
import psutil
import os
import subprocess
import re
from opentera_webrtc_ros_msgs.msg import RobotStatus
from std_msgs.msg import String
import json

class RobotStatusPublisher():
    def __init__(self):
        rospy.init_node("robot_status_publisher")
        self.status_pub = rospy.Publisher('/robot_status', RobotStatus, queue_size=10)
        self.status_webrtc_pub = rospy.Publisher('/webrtc_data_outgoing', String, queue_size=10)
        self.pub_rate = 1


    def get_ip_address(self, ifname: str):
        try:
            address = os.popen('ip addr show ' + ifname).read().split("inet ")[1].split("/")[0]
        except Exception as e:
            address = '127.0.0.1'
        finally:
            return address


    def get_disk_usage(self, mount_point='/'):
        result=os.statvfs(mount_point)
        block_size=result.f_frsize
        total_blocks=result.f_blocks
        free_blocks=result.f_bfree
        return 100 - (free_blocks * 100 / total_blocks)

    def run(self):
        r = rospy.Rate(self.pub_rate)
        while not rospy.is_shutdown():
            for i in range(100, -1, -5):
                # Fill timestamp
                status = RobotStatus()
                status.header.stamp = rospy.Time.now()

                # Fill (mostly fake) robot info
                status.battery_voltage = float(i)
                status.battery_current = 1.0
                status.cpu_usage = psutil.cpu_percent()
                status.mem_usage = 100 - (psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
                status.disk_usage = self.get_disk_usage()


                subprocess_result = subprocess.Popen('iwgetid',shell=True,stdout=subprocess.PIPE)
                subprocess_output = subprocess_result.communicate()[0],subprocess_result.returncode
                network_name = subprocess_output[0].decode('utf-8')
                status.wifi_network = network_name
                if status.wifi_network:
                    wifi_interface_name = status.wifi_network.split()[0]

                    command = "iwconfig %s | grep 'Link Quality='" % wifi_interface_name
                    subprocess_result = subprocess.Popen(command, shell=True,stdout=subprocess.PIPE)
                    subprocess_output = subprocess_result.communicate()[0],subprocess_result.returncode
                    decoded_output = subprocess_output[0].decode('utf-8')
                    numerator = int(re.search('=(.+?)/', decoded_output).group(1))
                    denominator = int(re.search('/(.+?) ', decoded_output).group(1))
                    status.wifi_strength = numerator / denominator * 100
                    status.local_ip = self.get_ip_address(wifi_interface_name)
                else:
                    status.wifi_strength = 0
                    status.local_ip = '127.0.0.1'



                # Publish for ROS
                self.status_pub.publish(status)

                # Publish for webrtc
                status_dict = {
                    'type': 'robotStatus',
                    'timestamp': status.header.stamp.secs,
                    'status': {
                        'isCharging': status.is_charging,
                        'batteryVoltage': status.battery_voltage,
                        'batteryCurrent': status.battery_current,
                        'cpuUsage': status.cpu_usage,
                        'memUsage': status.mem_usage,
                        'diskUsage': status.disk_usage,
                        'wifiNetwork': status.wifi_network,
                        'wifiStrength': status.wifi_strength,
                        'localIp': status.local_ip
                    }
                }
                self.status_webrtc_pub.publish(json.dumps(status_dict))
                r.sleep()


if __name__ == '__main__':
    print("Robot Status Publisher Starting")
    try:
        robot_status = RobotStatusPublisher()
        robot_status.run()
    except rospy.ROSInterruptException as e:
        print(e)
        pass
