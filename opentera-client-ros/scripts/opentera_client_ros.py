#!/usr/bin/env python
# PYTHONPATH is set properly when loading a workspace.

# Asyncio
import aiohttp
import asyncio
import json

# ROS
import rospy
from std_msgs.msg import String

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
        self.__event_publisher = rospy.Publisher('events', String, queue_size=10)

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

                ws = await client.ws_connect(url=websocket_url, ssl=False)
                print(ws)

                # Create alive publishing task
                status_task = asyncio.create_task(
                    self._opentera_send_device_status(client, url + OpenTeraROSClient.status_api_endpoint, token))

                while True:
                    msg = await ws.receive()

                    if msg.type == aiohttp.WSMsgType.text:
                        await self._parse_message(client, msg.json())
                    if msg.type == aiohttp.WSMsgType.closed:
                        print('websocket closed')
                        break
                    if msg.type == aiohttp.WSMsgType.error:
                        print('websocket error')
                        break

            print('cancel task')
            status_task.cancel()
            await status_task

    def run(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._opentera_main_loop(self.__base_url, self.__token))

    async def _opentera_send_device_status(self, client: aiohttp.ClientSession, url: str, token: str):
        while True:
            # Every 10 seconds
            await asyncio.sleep(10)
            params = {'token': token}

            from datetime import datetime

            # This can be anything...
            status = {
                'status': {'battery': 10.4, 'flag': False},
                'timestamp': datetime.now().timestamp()
            }

            async with client.post(url, params=params, json=status, verify_ssl=False) as response:
                if response.status == 200:
                    print('Sent status')
                else:
                    print('Send status failed')

    async def _parse_message(self, client: aiohttp.ClientSession, msg_dict: dict):
        try:
            if 'message' in msg_dict:
                message = ParseDict(msg_dict['message'], messages.TeraEvent(), ignore_unknown_fields=True)
                for any_msg in message.events:
                    # Test for DeviceEvent
                    device_event = messages.DeviceEvent()
                    if any_msg.Unpack(device_event):
                        # TODO Handle device_event
                        print('device_event:', device_event)
                        # TODO convert this to ROS message
                        self.__event_publisher.publish(
                            String(MessageToJson(device_event, including_default_value_fields=True)))

                    # Test for JoinSessionEvent
                    join_session_event = messages.JoinSessionEvent()
                    if any_msg.Unpack(join_session_event):
                        # TODO Handle join_session_event
                        print('join_session_event:', join_session_event)
                        # TODO convert this to ROS message
                        self.__event_publisher.publish(
                            String(MessageToJson(join_session_event, including_default_value_fields=True)))

                    # Test for ParticipantEvent
                    participant_event = messages.ParticipantEvent()
                    if any_msg.Unpack(participant_event):
                        print('participant_event:', participant_event)
                        # TODO Handle participant_event
                        # TODO convert this to ROS message
                        self.__event_publisher.publish(
                            String(MessageToJson(participant_event, including_default_value_fields=True)))

                    # Test for StopSessionEvent
                    stop_session_event = messages.StopSessionEvent()
                    if any_msg.Unpack(stop_session_event):
                        print('stop_session_event:', stop_session_event)
                        # TODO Handle stop_session_event
                        # TODO convert this to ROS message
                        self.__event_publisher.publish(
                            String(MessageToJson(stop_session_event, including_default_value_fields=True)))

                    # Test for UserEvent
                    user_event = messages.UserEvent()
                    if any_msg.Unpack(user_event):
                        # TODO Handle user_event
                        print('user_event:', user_event)
                        # TODO convert this to ROS message
                        self.__event_publisher.publish(
                            String(MessageToJson(user_event, including_default_value_fields=True)))

                    # Test for LeaveSessionEvent
                    leave_session_event = messages.LeaveSessionEvent()
                    if any_msg.Unpack(leave_session_event):
                        # TODO Handle leave_session_event
                        print('leave_session_event:', leave_session_event)
                        # TODO convert this to ROS message
                        self.__event_publisher.publish(
                            String(MessageToJson(leave_session_event, including_default_value_fields=True)))

                    # Test for JoinSessionReply
                    join_session_reply = messages.JoinSessionReplyEvent()
                    if any_msg.Unpack(join_session_reply):
                        print('join_session_reply:', join_session_reply)
                        # TODO Handle join_session_reply
                        # TODO convert this to ROS message
                        self.__event_publisher.publish(
                            String(MessageToJson(join_session_event, including_default_value_fields=True)))

                    # TODO Look for useful events

        except ParseError as e:
            print(e)

        return


if __name__ == '__main__':
    # Init ROS
    rospy.init_node('opentera_client_ros', anonymous=True)
    config_file_name = rospy.get_param('/config_file', '../config/local_config.json')

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
