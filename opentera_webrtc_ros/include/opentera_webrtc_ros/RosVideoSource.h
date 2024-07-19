#ifndef OPENTERA_WEBRTC_NATIVE_CLIENT_SOURCES_ROS_VIDEO_SOURCE_H
#define OPENTERA_WEBRTC_NATIVE_CLIENT_SOURCES_ROS_VIDEO_SOURCE_H

#include <OpenteraWebrtcNativeClient/Sources/VideoSource.h>
#include <sensor_msgs/msg/image.hpp>

namespace opentera
{
    /**
     * @brief A webrtc video source that sinks images from a ROS topic
     *
     * Usage: pass an shared_ptr to an instance of this to the VideoStreamClient constructor.
     */
    class RosVideoSource : public VideoSource
    {
    public:
        RosVideoSource(bool needsDenoising, bool isScreenCast);
        void sendFrame(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
    };
}

#endif
