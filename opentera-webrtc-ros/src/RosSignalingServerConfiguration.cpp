#include <ros/node_handle.h>
#include <RosSignalingServerconfiguration.h>

using namespace opentera;
using namespace std;
using namespace ros;

/**
 * @brief Build a signaling server configuration from the ROS parameter server
 * Parameters are retrieved from the signaling namespace under the node private namespace
 *
 * @param defaultClientName Default name for the webrtc peer
 * @return The signaling server configuration
 */
SignalingServerConfiguration RosSignalingServerConfiguration::fromRosParam(const std::string& defaultClientName)
{
    NodeHandle pnh("~signaling");

    string serverUrl;
    pnh.param<string>("server_url", serverUrl, "http://localhost:8080");

    string clientName;
    pnh.param<string>("client_name", clientName, defaultClientName);

    string room;
    pnh.param<string>("room_name", room, "chat");

    string password;
    pnh.param<string>("room_password", password, "abc");

    return SignalingServerConfiguration::create(serverUrl, clientName, room, password);
}
