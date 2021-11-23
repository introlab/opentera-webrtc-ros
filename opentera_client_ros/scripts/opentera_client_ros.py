#!/usr/bin/env python3
# PYTHONPATH is set properly when loading a workspace.

# Asyncio
import aiohttp
import asyncio
import json
import os
from signal import SIGINT, SIGTERM

# ROS
import rospy
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
import opentera.messages.python as messages
from google.protobuf.json_format import ParseDict, ParseError
from google.protobuf.json_format import MessageToJson


class OpenTeraROSClient:

    login_api_endpoint = '/api/device/login'
    status_api_endpoint = '/api/device/status'

    def __init__(self, url: str, token: str):
        self.__base_url = url
        self.__token = token
        self.__event_publisher = rospy.Publisher('events', OpenTeraEvent, queue_size=10)
        self.__robot_status_subscriber = rospy.Subscriber('robot_status', RobotStatus, self.robot_status_callback)
        self.__robot_status = {}

    def robot_status_callback(self, status: RobotStatus):
        # Update internal status
        # Will be sent by _opentera_send_device_status task as json
        self.__robot_status = {
            'timestamp': status.header.stamp.secs,
            'status': {
                'is_charging': status.is_charging,
                'battery_voltage': status.battery_voltage,
                'battery_current': status.battery_current,
                'cpu_usage': status.cpu_usage,
                'mem_usage': status.mem_usage,
                'disk_usage': status.disk_usage,
                'wifi_network': status.wifi_network,
                'wifi_strength': status.wifi_strength,
                'local_ip': status.local_ip
            }
        }

    async def _fetch(self, client, url, params=None):
        if params is None:
            params = {}
        async with client.get(url, params=params, verify_ssl=False) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                return {}

    async def _opentera_main_loop(self, url, token):

        async with aiohttp.ClientSession() as client:
            params = {'token': token}
            login_info = await self._fetch(client, url + OpenTeraROSClient.login_api_endpoint, params)
            print(login_info)
            if 'websocket_url' in login_info:
                websocket_url = login_info['websocket_url']

                ws = await client.ws_connect(url=websocket_url, ssl=False,  autoping=True, autoclose=True)
                print(ws)

                # Create alive publishing task
                status_task = asyncio.get_event_loop().create_task(
                              self._opentera_send_device_status(client, url + OpenTeraROSClient.status_api_endpoint, token))

                while not rospy.is_shutdown():
                    msg = await ws.receive()

                    if msg.type == aiohttp.WSMsgType.text:
                        await self._parse_message(client, msg.json())
                    if msg.type == aiohttp.WSMsgType.closed:
                        print('websocket closed')
                        break
                    if msg.type == aiohttp.WSMsgType.error:
                        print('websocket error')
                        break

                status_task.cancel()
                await status_task

            rospy.logwarn('cancel task')

    def run(self):
        try:
            loop = asyncio.get_event_loop()

            main_task = asyncio.ensure_future(self._opentera_main_loop(self.__base_url, self.__token))

            for signal in [SIGINT, SIGTERM]:
                loop.add_signal_handler(signal, main_task.cancel)

            loop.run_until_complete(main_task)
        except asyncio.CancelledError as e:
            print('Main Task cancelled', e)
            # Exit ROS loop
            rospy.signal_shutdown('SIGINT/SIGTERM Detected')

    async def _opentera_send_device_status(self, client: aiohttp.ClientSession, url: str, token: str):
        while not rospy.is_shutdown():
            try:
                # Every 10 seconds
                await asyncio.sleep(10)
                params = {'token': token}

                async with client.post(url, params=params, json=self.__robot_status, verify_ssl=False) as response:
                    if response.status != 200:
                        print('Send status failed')
                        break
            except asyncio.CancelledError as e:
                print('_opentera_send_device_status', e)
                # Exit loop
                break

    async def _parse_message(self, client: aiohttp.ClientSession, msg_dict: dict):
        try:
            if 'message' in msg_dict:
                message = ParseDict(msg_dict['message'], messages.TeraEvent(), ignore_unknown_fields=True)

                # All events in same message
                opentera_events = OpenTeraEvent()

                for any_msg in message.events:
                    # Test for DeviceEvent
                    device_event = messages.DeviceEvent()
                    if any_msg.Unpack(device_event):
                        # print('device_event')
                        event = DeviceEvent()
                        event.device_uuid = device_event.device_uuid
                        event.type = device_event.type
                        event.device_name = device_event.device_name
                        event.device_status = device_event.device_status
                        opentera_events.device_events.append(event)

                    # Test for JoinSessionEvent
                    join_session_event = messages.JoinSessionEvent()
                    if any_msg.Unpack(join_session_event):
                        # print('join_session_event')
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

                    # Test for ParticipantEvent
                    participant_event = messages.ParticipantEvent()
                    if any_msg.Unpack(participant_event):
                        # print('participant_event')
                        event = ParticipantEvent()
                        event.participant_uuid = participant_event.participant_uuid
                        event.type = participant_event.type
                        event.participant_name = participant_event.participant_name
                        event.participant_project_name = participant_event.participant_project_name
                        event.participant_site_name = participant_event.participant_site_name
                        opentera_events.participant_events.append(event)

                    # Test for StopSessionEvent
                    stop_session_event = messages.StopSessionEvent()
                    if any_msg.Unpack(stop_session_event):
                        # print('stop_session_event')
                        event = StopSessionEvent()
                        event.session_uuid = stop_session_event.session_uuid
                        event.service_uuid = stop_session_event.service_uuid
                        opentera_events.stop_session_events.append(event)

                    # Test for UserEvent
                    user_event = messages.UserEvent()
                    if any_msg.Unpack(user_event):
                        # print('user_event')
                        event = UserEvent()
                        event.user_uuid = user_event.user_uuid
                        event.type = user_event.type
                        event.user_fullname = user_event.user_fullname
                        opentera_events.user_events.append(event)

                    # Test for LeaveSessionEvent
                    leave_session_event = messages.LeaveSessionEvent()
                    if any_msg.Unpack(leave_session_event):
                        # print('leave_session_event')
                        event = LeaveSessionEvent()
                        event.session_uuid = leave_session_event.session_uuid
                        event.service_uuid = leave_session_event.service_uuid
                        event.leaving_participants = leave_session_event.leaving_participants
                        event.leaving_users = leave_session_event.leaving_users
                        event.leaving_devices = leave_session_event.leaving_devices
                        opentera_events.leave_session_events.append(event)

                    # Test for JoinSessionReply
                    join_session_reply = messages.JoinSessionReplyEvent()
                    if any_msg.Unpack(join_session_reply):
                        # print('join_session_reply')
                        event = JoinSessionReplyEvent()
                        event.session_uuid = join_session_reply.session_uuid
                        event.user_uuid = join_session_reply.user_uuid
                        event.participant_uuid = join_session_reply.participant_uuid
                        event.device_uuid = join_session_reply.device_uuid
                        event.join_reply = join_session_reply.join_reply
                        event.reply_msg = join_session_reply.reply_msg
                        opentera_events.join_session_reply_events.append(event)

                    # TODO Handle other events if required.

                self.__event_publisher.publish(opentera_events)

        except ParseError as e:
            print(e)

        return


if __name__ == '__main__':
    # Init ROS
    rospy.init_node('opentera_client_ros', anonymous=True)

    # Guessing config file path
    base_folder = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_folder, '../config/client_config.json')
    config_file_name = rospy.get_param('~config_file', config_path)

    # Read config file
    # Should be a param for this node
    with open(config_file_name) as json_file:
        data = json.load(json_file)
        if 'url' in data and 'client_token' in data:
            url = data['url']
            token = data['client_token']
        else:
            exit(-1)

        client = OpenTeraROSClient(url=url, token=token)
        client.run()

