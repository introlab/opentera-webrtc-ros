#ifndef OPENTERA_WEBRTC_NATIVE_CLIENT_SOURCES_ROS_VIDEO_SOURCE_H
#define OPENTERA_WEBRTC_NATIVE_CLIENT_SOURCES_ROS_VIDEO_SOURCE_H

#include <OpenteraWebrtcNativeClient/Sources/VideoSource.h>
#include <sensor_msgs/Image.h>

namespace opentera
{
    /**
     * @brief A webrtc video source that sinks images from a ROS topic
     *
     * Usage: pass an shared_ptr to an instance of this to the VideoStreamClient constructor.
     * Use the imageCallback as a ROS topic subscriber callback.
     */
    class RosVideoSource : public VideoSource
    {

    public:
        RosVideoSource(bool needsDenoising, bool isScreenCast);
        void imageCallback(const sensor_msgs::ImageConstPtr& msg);
    };
}

#endif
