#ifndef OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_SIGNALING_SERVER_CONFIGURATION_H
#define OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_SIGNALING_SERVER_CONFIGURATION_H

#include <OpenteraWebrtcNativeClient/Configurations/SignalingServerConfiguration.h>

namespace opentera
{

    /**
     * @brief Utility to build signaling server configuration from ROS parameter server
     */
    class RosSignalingServerConfiguration
    {
        static std::string getQueryFrom(const std::string& query, const std::string& queries);

    public:
        static SignalingServerConfiguration fromRosParam();
        static SignalingServerConfiguration fromUrl(const std::string& url);
        static std::string getIceServerUrl(const std::string& url);
        static std::string getBaseUrl(const std::string& url);
    };
}

#endif
