#!/usr/bin/env python3

import json
import os
import re
import subprocess

import psutil
import rclpy
import rclpy.exceptions
import rclpy.node
from opentera_webrtc_ros_msgs.msg import RobotStatus
from std_msgs.msg import Bool, Float32, String


class RobotStatusPublisher(rclpy.node.Node):
    def __init__(self):
        super().__init__("robot_status_publisher")  # type: ignore

        self.status_pub = self.create_publisher(
            RobotStatus, '/robot_status', 10)
        self.status_webrtc_pub = self.create_publisher(
            String, '/webrtc_data_outgoing', 10)
        self.pub_rate = 1
        self.mic_volume_sub = self.create_subscription(
            Float32, 'mic_volume', self._set_mic_volume_cb, 10)
        self.mic_volume = 1.0
        self.enable_camera_sub = self.create_subscription(
            Bool, 'enable_camera', self._set_enable_camera_cb, 10)
        self.camera_enabled = True
        self.volume_sub = self.create_subscription(
            Float32, 'volume', self._set_volume_cb, 10)
        self.volume = 1.0
        self.io = psutil.net_io_counters(pernic=True)
        self.bytes_sent = 0
        self.bytes_recv = 0

        self._status_timer = self.create_timer(1.0 / self.pub_rate, self._publish_status_callback)
        self._status_generator = self.get_status()

    def get_ip_address(self, ifname: str):
        try:
            address = os.popen('ip addr show ' +
                               ifname).read().split("inet ")[1].split("/")[0]
        except Exception:
            address = '127.0.0.1'

        return address

    def get_disk_usage(self, mount_point='/'):
        result = os.statvfs(mount_point)
        block_size = result.f_frsize
        total_blocks = result.f_blocks
        free_blocks = result.f_bfree
        return 100 - (free_blocks * 100 / total_blocks)

    def _set_mic_volume_cb(self, msg):
        self.mic_volume = msg.data

    def _set_enable_camera_cb(self, msg):
        self.camera_enabled = msg.data

    def _set_volume_cb(self, msg):
        self.volume = msg.data

    def _publish_status_callback(self):
        self.status_webrtc_pub.publish(String(data=json.dumps(next(self._status_generator))))

    def get_status(self):
        while True:
            for i in range(100, -1, -5):
                    # Fill timestamp
                    status = RobotStatus()
                    status.header.stamp = self.get_clock().now().to_msg()

                    # Fill (mostly fake) robot info
                    status.battery_level = float(i)
                    status.battery_voltage = 12.0
                    status.battery_current = 1.0
                    status.cpu_usage = psutil.cpu_percent()
                    status.mem_usage = 100 - \
                        (psutil.virtual_memory().available *
                        100 / psutil.virtual_memory().total)
                    status.disk_usage = self.get_disk_usage()

                    status.mic_volume = self.mic_volume
                    status.is_camera_on = self.camera_enabled
                    status.volume = self.volume

                    subprocess_result = subprocess.Popen(
                        'iwgetid', shell=True, stdout=subprocess.PIPE)
                    subprocess_output = subprocess_result.communicate()[
                        0], subprocess_result.returncode
                    network_name = subprocess_output[0].decode('utf-8')
                    if network_name:
                        wifi_interface_name = network_name.split()[0]
                        status.wifi_network = network_name.split('"')[1]

                        command = "iwconfig %s | grep 'Link Quality='" % wifi_interface_name
                        subprocess_result = subprocess.Popen(
                            command, shell=True, stdout=subprocess.PIPE)
                        subprocess_output = subprocess_result.communicate()[
                            0], subprocess_result.returncode
                        decoded_output = subprocess_output[0].decode('utf-8')
                        numerator = int(
                            re.search('=(.+?)/', decoded_output).group(1))  # type: ignore
                        denominator = int(
                            re.search('/(.+?) ', decoded_output).group(1))  # type: ignore
                        status.wifi_strength = numerator / denominator * 100
                        status.local_ip = self.get_ip_address(wifi_interface_name)

                        io_2 = psutil.net_io_counters(pernic=True)
                        status.upload_speed = (io_2[wifi_interface_name].bytes_sent - self.bytes_sent) * 8.0
                        status.download_speed = (io_2[wifi_interface_name].bytes_recv - self.bytes_recv) * 8.0
                        self.bytes_sent = io_2[wifi_interface_name].bytes_sent
                        self.bytes_recv = io_2[wifi_interface_name].bytes_recv
                    else:
                        status.wifi_network = ""
                        status.wifi_strength = 0.0
                        status.local_ip = '127.0.0.1'

                    # Publish for ROS
                    self.status_pub.publish(status)

                    # Publish for webrtc
                    status_dict = {
                        'type': 'robotStatus',
                        'timestamp': status.header.stamp.sec,
                        'status': {
                            'isCharging': status.is_charging,
                            'batteryVoltage': status.battery_voltage,
                            'batteryCurrent': status.battery_current,
                            'batteryLevel': status.battery_level,
                            'cpuUsage': status.cpu_usage,
                            'memUsage': status.mem_usage,
                            'diskUsage': status.disk_usage,
                            'wifiNetwork': status.wifi_network,
                            'wifiStrength': status.wifi_strength,
                            'uploadSpeed': status.upload_speed,
                            'downloadSpeed': status.download_speed,
                            'localIp': status.local_ip,
                            'micVolume':status.mic_volume,
                            'isCameraOn':status.is_camera_on,
                            'volume':status.volume
                        }
                    }

                    yield status_dict

    def run(self):
        rclpy.spin(self)


def main():
    print("Robot Status Publisher Starting")
    rclpy.init()
    robot_status = RobotStatusPublisher()
    
    try:
        robot_status.run()
    except KeyboardInterrupt:
        pass

    robot_status.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

if __name__ == '__main__':
    main()
