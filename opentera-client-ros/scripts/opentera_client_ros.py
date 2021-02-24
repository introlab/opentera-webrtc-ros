#!/usr/bin/env python
# PYTHONPATH is set properly when loading a workspace.
import rospy
import aiohttp
import asyncio
import json
from std_msgs.msg import String

import opentera.messages.python as messages
from google.protobuf.json_format import ParseDict, ParseError
from google.protobuf.json_format import MessageToJson


login_api_endpoint = '/api/device/login'

publisher = None
event_publisher = None


async def fetch(client, url, params=None):
    if params is None:
        params = {}
    async with client.get(url, params=params, verify_ssl=False) as resp:
        if resp.status == 200:
            return await resp.json()
        else:
            return {}


class OpenTeraClient:
    def __init__(self, token: str):
        pass


def setup_ros_node():
    rospy.init_node('opentera_client_ros', anonymous=True)

    # TODO get parameters

    # Setup publisher
    global publisher
    publisher = rospy.Publisher('alive', String, queue_size=10)

    global event_publisher
    event_publisher = rospy.Publisher('events', String, queue_size=10)


async def ros_alive_publisher():
    while True:
        await asyncio.sleep(1)
        publisher.publish(String('Alive!'))


async def parse_message(client, msg_dict: dict):
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
                    event_publisher.publish(
                        String(MessageToJson(device_event, including_default_value_fields=True)))

                # Test for JoinSessionEvent
                join_session_event = messages.JoinSessionEvent()
                if any_msg.Unpack(join_session_event):
                    # TODO Handle join_session_event
                    print('join_session_event:', join_session_event)
                    # TODO convert this to ROS message
                    event_publisher.publish(
                        String(MessageToJson(join_session_event, including_default_value_fields=True)))

                # Test for ParticipantEvent
                participant_event = messages.ParticipantEvent()
                if any_msg.Unpack(participant_event):
                    print('participant_event:', participant_event)
                    # TODO Handle participant_event
                    # TODO convert this to ROS message
                    event_publisher.publish(
                        String(MessageToJson(participant_event, including_default_value_fields=True)))

                # Test for StopSessionEvent
                stop_session_event = messages.StopSessionEvent()
                if any_msg.Unpack(stop_session_event):
                    print('stop_session_event:', stop_session_event)
                    # TODO Handle stop_session_event
                    # TODO convert this to ROS message
                    event_publisher.publish(
                        String(MessageToJson(stop_session_event, including_default_value_fields=True)))

                # Test for UserEvent
                user_event = messages.UserEvent()
                if any_msg.Unpack(user_event):
                    # TODO Handle user_event
                    print('user_event:', user_event)
                    # TODO convert this to ROS message
                    event_publisher.publish(
                        String(MessageToJson(user_event, including_default_value_fields=True)))

                # Test for LeaveSessionEvent
                leave_session_event = messages.LeaveSessionEvent()
                if any_msg.Unpack(leave_session_event):
                    # TODO Handle leave_session_event
                    print('leave_session_event:', leave_session_event)
                    # TODO convert this to ROS message
                    event_publisher.publish(
                        String(MessageToJson(leave_session_event, including_default_value_fields=True)))

                # Test for JoinSessionReply
                join_session_reply = messages.JoinSessionReplyEvent()
                if any_msg.Unpack(join_session_reply):
                    print('join_session_reply:', join_session_reply)
                    # TODO Handle join_session_reply
                    # TODO convert this to ROS message
                    event_publisher.publish(
                        String(MessageToJson(join_session_event, including_default_value_fields=True)))

                # TODO Look for useful events

    except ParseError as e:
        print(e)

    return


async def main(url, token):
    # Create alive publishing task
    alive_pub_task = asyncio.create_task(ros_alive_publisher())

    async with aiohttp.ClientSession() as client:
        params = {'token': token}
        login_info = await fetch(client, url + login_api_endpoint, params)
        print(login_info)
        if 'websocket_url' in login_info:
            websocket_url = login_info['websocket_url']

            ws = await client.ws_connect(url=websocket_url, ssl=False)
            print(ws)

            while True:
                msg = await ws.receive()

                if msg.type == aiohttp.WSMsgType.text:
                    await parse_message(client, msg.json())
                if msg.type == aiohttp.WSMsgType.closed:
                    print('websocket closed')
                    break
                if msg.type == aiohttp.WSMsgType.error:
                    print('websocket error')
                    break

        print('cancel task')
        alive_pub_task.cancel()
        await alive_pub_task

if __name__ == '__main__':

    url = None
    token = None

    setup_ros_node()

    # Read config file
    # Should be a param for this node
    with open('../config/client_config.json') as json_file:
        data = json.load(json_file)
        if 'url' in data:
            url = data['url']
        if 'client_token' in data:
            token = data['client_token']

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(url, token))
