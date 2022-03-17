import rospy

from opentera_webrtc.native_client import DataChannelClient, DataChannelConfiguration
from opentera_webrtc.native_client import SignalingServerConfiguration, WebrtcConfiguration
from opentera_webrtc.native_client import IceServer, Client, RoomClient

from rosbridge_library.rosbridge_protocol import RosbridgeProtocol
from rosbridge_library.util import json, bson

from typing import List

from . import ros_signaling_server_configuration as ros_ss_config


class RosbridgeWebrtc:

    def _init_signaling_server(self, signaling_server_configuration: SignalingServerConfiguration):
        ice_server_url = ros_ss_config.get_ice_server_url(
            signaling_server_configuration.url)
        rospy.loginfo(f"Fetching ice servers from: {ice_server_url}")

        try:
            ice_servers = IceServer.fetch_from_server(
                ice_server_url, signaling_server_configuration.password)
        except:
            rospy.logerr(f"Error fetching ice servers from: {ice_server_url}")
            ice_servers = []

        self.signaling_client = DataChannelClient(signaling_server_configuration=signaling_server_configuration,
                                                  webrtc_configuration=WebrtcConfiguration.create(ice_servers=ice_servers), data_channel_configuration=DataChannelConfiguration.create())

        self.signaling_client.tls_verification_enabled = rospy.get_param(  # type: ignore
            "~signaling", {}).get("verify_ssl", True)  # type: ignore

    def _init_data_channel_callback(self):
        self.signaling_client.on_data_channel_message = self.__on_data_channel_message  # type: ignore
        self.signaling_client

    def __on_data_channel_message(self, client: Client, data: str):
        pass

    def __on_data_channel_opened(self):
        pass

    def __on_data_channel_closed(self):
        pass

    def __on_data_channel_error(self, msg: str):
        pass

    def __on_room_clients_changed(self, clients: List[RoomClient]):
        pass

    def __call_acceptor(self, client: Client):
        pass

    def __on_call_rejected(self, client: Client):
        pass

    def __on_client_connected(self, client: Client):
        pass

    def __on_room_clients_changed(self, clients: List[RoomClient]):
        pass

    def __on_room_clients_changed(self, clients: List[RoomClient]):
        pass

    def __init__(self) -> None:
        if rospy.get_param("~is_stand_alone", True):  # type: ignore
            self._init_signaling_server(ros_ss_config.from_ros_param())
            self._init_data_channel_callback()
            # self.connect()
