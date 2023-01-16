#ifndef OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_VIDEO_STREAM_CONFIGURATION_H
#define OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_VIDEO_STREAM_CONFIGURATION_H

#include <OpenteraWebrtcNativeClient/Configurations/VideoStreamConfiguration.h>

namespace opentera
{

    /**
     * @brief Utility to build a video stream configuration from ROS parameter server
     */
    class RosVideoStreamConfiguration
    {
    public:
        static VideoStreamConfiguration fromRosParam();
    };
}

#endif
