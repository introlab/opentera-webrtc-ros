from opentera_webrtc.native_client import SignalingServerConfiguration
from typing import Optional

import rospy
from urllib.parse import urlparse


def from_ros_param() -> SignalingServerConfiguration:
    signaling_params = rospy.get_param("~signaling", {})

    server_url = signaling_params.get(
        "server_url", "http://localhost:8080")
    client_name = signaling_params.get("client_name", "rosbridge_server")
    room = signaling_params.get("room", "rosbridge")
    password = signaling_params.get("password", "abc")

    return SignalingServerConfiguration.create(url=server_url, client_name=client_name, room=room, password=password)


def from_url(url: str) -> SignalingServerConfiguration:
    address = f"{get_base_url(url)}/socket.io"

    queries = _extract_queries(url)
    password = _get_query_from("pwd", queries) or ""

    signaling = rospy.get_param("~signaling", {})
    client_name = signaling.get("client_name", "rosbridge_server")
    room = signaling.get("room", "rosbridge")

    return SignalingServerConfiguration.create(url=address, client_name=client_name, room=room, password=password)


def _get_query_from(query: str, queries: str) -> Optional[str]:
    pos1 = queries.find(f"{query}=")

    if pos1 == -1:
        rospy.logerr(f"[{query}] not found in [{queries}]")
        return None

    remaining = queries[pos1:]
    pos2 = remaining.find("&")

    return remaining[len(query) + 1:pos2] if pos2 != -1 else remaining[len(query) + 1:]


def get_base_url(url: str) -> str:
    parsed_url = urlparse(url)

    if parsed_url.scheme is None:
        rospy.logerr(f"No protocol defined in url: {url}")
        return url

    return f"{parsed_url.scheme}://{parsed_url.netloc}"


def get_ice_server_url(url: str) -> str:
    rospy.loginfo(f"get_ice_server_url from url: {url}")
    return f"{get_base_url(url)}/iceservers"


def _extract_queries(url: str) -> str:
    parsed_url = urlparse(url)

    if parsed_url.query is None:
        return "?"
    return f"?{parsed_url.query}"
