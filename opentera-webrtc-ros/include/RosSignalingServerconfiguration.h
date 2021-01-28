#ifndef OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_SIGNALING_SERVER_CONFIGURATION_H
#define OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_SIGNALING_SERVER_CONFIGURATION_H

#include <OpenteraWebrtcNativeClient/Configurations/SignalingServerConfiguration.h>

namespace opentera {

    /**
     * @brief Utility to build signaling server configuration from ROS parameter server
     */
    class RosSignalingServerConfiguration: public SignalingServerConfiguration
    {
    public:
        static SignalingServerConfiguration fromRosParam(const std::string& defaultClientName);
    };
}

#endif
