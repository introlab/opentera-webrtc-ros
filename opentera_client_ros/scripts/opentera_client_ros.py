#!/usr/bin/env python3
# PYTHONPATH is set properly when loading a workspace.

# Asyncio
import aiohttp
import asyncio
import threading
import json
import os
from signal import SIGINT, SIGTERM
from pathlib import Path

# ROS
import rclpy
import rclpy.node
import rclpy.parameter
from std_msgs.msg import String
from opentera_webrtc_ros_msgs.msg import OpenTeraEvent
from opentera_webrtc_ros_msgs.msg import DatabaseEvent
from opentera_webrtc_ros_msgs.msg import DeviceEvent
from opentera_webrtc_ros_msgs.msg import JoinSessionEvent
from opentera_webrtc_ros_msgs.msg import JoinSessionReplyEvent
from opentera_webrtc_ros_msgs.msg import LeaveSessionEvent
from opentera_webrtc_ros_msgs.msg import LogEvent
from opentera_webrtc_ros_msgs.msg import ParticipantEvent
from opentera_webrtc_ros_msgs.msg import StopSessionEvent
from opentera_webrtc_ros_msgs.msg import UserEvent
from opentera_webrtc_ros_msgs.msg import RobotStatus

# OpenTera
import opentera_protobuf_messages as messages
from google.protobuf.json_format import ParseDict, ParseError
from google.protobuf.json_format import MessageToJson


class OpenTeraROSClient(rclpy.node.Node):

    login_api_endpoint = '/api/device/login'
    status_api_endpoint = '/api/device/status'
    stop_session_endpoint = '/robot/api/session/manager'

    def __init__(self, url: str, token: str):
        super().__init__('opentera_client_ros')  # type: ignore

        self.__base_url = url
        self.__token = token
        self.__event_publisher = self.create_publisher(
            OpenTeraEvent, 'events', 10)
        self.__robot_status_subscriber = self.create_subscription(
            RobotStatus, 'robot_status', self.robot_status_callback, 1)
        self.__manage_session_subscriber = self.create_subscription(
            String, 'manage_session', self.manage_session_callback, 10)
        self.__robot_status = {}
        self.__stop_session_json = {}
        self.__client = {}
        self.__eventLoop = {}

        self.__current_device_uuid = ''
        self.__current_device_name = ''

    def robot_status_callback(self, status: RobotStatus):
        # Update internal status
        # Will be sent by _opentera_send_device_status task as json
        self.__robot_status = {
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

    def manage_session_callback(self, msg: String):
        asyncio.run_coroutine_threadsafe(self._opentera_send_manage_session(msg.data), self.__eventLoop)

    def __set_stop_session(self, session_uuid):
        self.__stop_session_json = {
            'session_uuid': session_uuid,
            'action': 'stop'
        }

    async def _fetch(self, client, url, params=None):
        if params is None:
            params = {}
        async with client.get(url, params=params, verify_ssl=False) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                return {}

    async def _opentera_main_loop(self):

        async with aiohttp.ClientSession() as self.__client:
            params = {'token': self.__token}
            login_info = await self._fetch(self.__client, self.__base_url + OpenTeraROSClient.login_api_endpoint, params)
            self.get_logger().info(str(login_info))

            if 'device_info' in login_info:
                device_info = login_info['device_info']
                if 'device_uuid' in device_info:
                    self.__current_device_uuid = device_info['device_uuid']
                if 'device_name' in device_info:
                    self.__current_device_name = device_info['device_name']

            if 'websocket_url' in login_info:
                websocket_url = login_info['websocket_url']

                ws = await self.__client.ws_connect(url=websocket_url, ssl=False,  autoping=True, autoclose=True)
                self.get_logger().info(str(ws))

                # Create alive publishing task
                status_task = self.__eventLoop.create_task(
                    self._opentera_send_device_status(self.__base_url + OpenTeraROSClient.status_api_endpoint))

                while rclpy.ok():
                    msg = await ws.receive()

                    if msg.type == aiohttp.WSMsgType.text:
                        await self._parse_message(self.__client, msg.json())
                    if msg.type == aiohttp.WSMsgType.closed:
                        self.get_logger().info('websocket closed')
                        break
                    if msg.type == aiohttp.WSMsgType.error:
                        self.get_logger().info('websocket error')
                        break

                status_task.cancel()
                await status_task

            self.get_logger().warning('cancel task')

    def ros_thread_run(self):
        try:
            rclpy.spin(self)
        except rclpy.executors.ExternalShutdownException:
            pass

    def run(self):
        try:

            self.__ros_thread = threading.Thread(target=self.ros_thread_run, args=[])
            self.__ros_thread.start()

            self.__eventLoop = asyncio.get_event_loop()

            main_task = asyncio.ensure_future(
                self._opentera_main_loop())

            for signal in [SIGINT, SIGTERM]:
                self.__eventLoop.add_signal_handler(signal, main_task.cancel)

            self.__eventLoop.run_until_complete(main_task)
        except asyncio.CancelledError as e:
            self.get_logger().error(f'Main Task cancelled: {e}')
            # Exit ROS loop
            rclpy.shutdown()
            self.__ros_thread.join()

    async def _opentera_send_device_status(self, url: str):
        while rclpy.ok():
            try:
                # Every 10 seconds
                await asyncio.sleep(10)

                if len(self.__robot_status) > 0:
                    params = {'token': self.__token}
                    async with self.__client.post(url, params=params, json=self.__robot_status, verify_ssl=False) as response:
                        if response.status != 200:
                            self.get_logger().warning('Send status failed')
            except asyncio.CancelledError as e:
                self.get_logger().error('_opentera_send_device_status', e)
                # Exit loop
                break

    async def _opentera_send_manage_session(self, action):
        params = {'token': self.__token}
        if (action == 'stop'):
            try:
                async with self.__client.post(self.__base_url + OpenTeraROSClient.stop_session_endpoint,
                    params=params, json=self.__stop_session_json, verify_ssl=False) as response:
                        if response.status != 200:
                            self.get_logger().warning('Send stop session failed')
            except asyncio.CancelledError as e:
                    self.get_logger().error(f'_opentera_send_manage_session: {e}')
        else:
            raise NotImplementedError(f"Action {action} not implemented")

    async def _parse_message(self, client: aiohttp.ClientSession, msg_dict: dict):
        try:
            if 'message' in msg_dict:
                message = ParseDict(
                    msg_dict['message'], messages.TeraEvent(), ignore_unknown_fields=True)

                # All events in same message
                opentera_events = OpenTeraEvent()
                opentera_events.current_device_uuid = self.__current_device_uuid
                opentera_events.current_device_name = self.__current_device_name

                for any_msg in message.events:
                    # Test for DeviceEvent
                    device_event = messages.DeviceEvent()
                    if any_msg.Unpack(device_event):
                        event = DeviceEvent()
                        event.device_uuid = device_event.device_uuid
                        event.type = device_event.type
                        event.device_name = device_event.device_name
                        event.device_status = device_event.device_status
                        opentera_events.device_events.append(event)
                        continue

                    # Test for JoinSessionEvent
                    join_session_event = messages.JoinSessionEvent()
                    if any_msg.Unpack(join_session_event):
                        event = JoinSessionEvent()
                        event.session_url = join_session_event.session_url
                        event.session_creator_name = join_session_event.session_creator_name
                        event.session_uuid = join_session_event.session_uuid
                        event.session_participants = join_session_event.session_participants
                        event.session_users = join_session_event.session_users
                        event.session_devices = join_session_event.session_devices
                        event.join_msg = join_session_event.join_msg
                        event.session_parameters = join_session_event.session_parameters
                        event.service_uuid = join_session_event.service_uuid
                        opentera_events.join_session_events.append(event)

                        self.__set_stop_session(event.session_uuid)
                        continue

                    # Test for ParticipantEvent
                    participant_event = messages.ParticipantEvent()
                    if any_msg.Unpack(participant_event):
                        event = ParticipantEvent()
                        event.participant_uuid = participant_event.participant_uuid
                        event.type = participant_event.type
                        event.participant_name = participant_event.participant_name
                        event.participant_project_name = participant_event.participant_project_name
                        event.participant_site_name = participant_event.participant_site_name
                        opentera_events.participant_events.append(event)
                        continue

                    # Test for StopSessionEvent
                    stop_session_event = messages.StopSessionEvent()
                    if any_msg.Unpack(stop_session_event):
                        event = StopSessionEvent()
                        event.session_uuid = stop_session_event.session_uuid
                        event.service_uuid = stop_session_event.service_uuid
                        opentera_events.stop_session_events.append(event)
                        continue

                    # Test for UserEvent
                    user_event = messages.UserEvent()
                    if any_msg.Unpack(user_event):
                        event = UserEvent()
                        event.user_uuid = user_event.user_uuid
                        event.type = user_event.type
                        event.user_fullname = user_event.user_fullname
                        opentera_events.user_events.append(event)
                        continue

                    # Test for LeaveSessionEvent
                    leave_session_event = messages.LeaveSessionEvent()
                    if any_msg.Unpack(leave_session_event):
                        event = LeaveSessionEvent()
                        event.session_uuid = leave_session_event.session_uuid
                        event.service_uuid = leave_session_event.service_uuid
                        event.leaving_participants = leave_session_event.leaving_participants
                        event.leaving_users = leave_session_event.leaving_users
                        event.leaving_devices = leave_session_event.leaving_devices
                        opentera_events.leave_session_events.append(event)
                        continue

                    # Test for JoinSessionReply
                    join_session_reply = messages.JoinSessionReplyEvent()
                    if any_msg.Unpack(join_session_reply):
                        event = JoinSessionReplyEvent()
                        event.session_uuid = join_session_reply.session_uuid
                        event.user_uuid = join_session_reply.user_uuid
                        event.participant_uuid = join_session_reply.participant_uuid
                        event.device_uuid = join_session_reply.device_uuid
                        event.join_reply = join_session_reply.join_reply
                        event.reply_msg = join_session_reply.reply_msg
                        opentera_events.join_session_reply_events.append(event)
                        continue

                    # TODO Handle other events if required.
                    self.get_logger().error(f"Unknown message type: {any_msg}")

                self.__event_publisher.publish(opentera_events)

        except ParseError as e:
            self.get_logger().error(e)

        return


class ConfigFileParam(rclpy.node.Node):
    @staticmethod
    def get_config_path():
        node = ConfigFileParam('__opentera_client_ros_config_file_param')  # type: ignore
        config_file_param = node.declare_parameter('config_file', rclpy.parameter.Parameter.Type.STRING).get_parameter_value()

        if config_file_param.type == rclpy.parameter.Parameter.Type.NOT_SET:
            node.get_logger().error('No config file provided')
            node.destroy_node()
            return None

        config_file_path = Path(config_file_param.string_value).expanduser().resolve()
        node.destroy_node()

        return config_file_path


if __name__ == '__main__':
    # Init ROS
    rclpy.init()

    config_file_name = ConfigFileParam.get_config_path()

    if config_file_name is None:
        exit(-1)

    # Read config file
    # Should be a param for this node
    with config_file_name.open() as json_file:
        data = json.load(json_file)
        if 'url' in data and 'client_token' in data:
            url = data['url']
            token = data['client_token']
        else:
            exit(-1)

        client = OpenTeraROSClient(url=url, token=token)
        client.run()
