#include <ros/node_handle.h>
#include <RosSignalingServerconfiguration.h>
#include <RosStreamBridge.h>
#include <RosNodeParameters.h>

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
SignalingServerConfiguration RosSignalingServerConfiguration::fromRosParam()
{
    string serverUrl, clientName, room, password;
    RosNodeParameters::loadSignalingParams(serverUrl, clientName, room, password);

    return SignalingServerConfiguration::create(serverUrl, clientName, room, password);
}

/**
 * @brief Build a signaling server configuration from the given Url.
 * Parameters are retrieved from the queries and from the signaling namespace under the node private namespace.
 * 
 * @param url An url to parse.
 * 
 * @return The signaling server configuration.
 */ 
SignalingServerConfiguration RosSignalingServerConfiguration::fromUrl(const std::string& url) {
    size_t pos1, pos2;

    pos1 = url.find_last_of("/");
    string address = url.substr(0, pos1) + "/socket.io";

    pos1 = url.find("?");
    string queries = url.substr(pos1);

    string password = getQueryFrom("pwd", queries);

    string clientName, roomName;
    RosNodeParameters::loadSignalingParams(clientName, roomName);

    return SignalingServerConfiguration::create(address, clientName, roomName, password);
}


/**
 * @brief Get a query value among the queries arguments.
 * Queries format: "?query1=123&query2=abc"
 * 
 * @param query The parameters that contain de value.
 * @param queries The full queries in which the query argument his.
 * 
 * @return The value of the query
 */ 
std::string RosSignalingServerConfiguration::getQueryFrom(const std::string& query, const std::string& queries) {
    size_t pos1, pos2;

    pos1 = queries.find(query + "=");
    string remaining = queries.substr(pos1);
    pos2 = remaining.find_first_of("&");

    return remaining.substr(0, pos2).substr(query.size() + 1);
}