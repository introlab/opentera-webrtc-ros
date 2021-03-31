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
