#ifndef OPENTERA_WEBRTC_NATIVE_CLIENT_SOURCES_ROS_AUDIO_SOURCE_H
#define OPENTERA_WEBRTC_NATIVE_CLIENT_SOURCES_ROS_AUDIO_SOURCE_H

#include <OpenteraWebrtcNativeClient/Sources/AudioSource.h>
#include <audio_utils_msgs/msg/audio_frame.hpp>
#include <rclcpp/rclcpp.hpp>

namespace opentera
{

    /**
     * @brief A webrtc audio source that gets images from a ROS topic
     *
     * Usage: pass an shared_ptr to an instance of this to the xxxx constructor.
     */
    class RosAudioSource : public AudioSource
    {
    public:
        RosAudioSource(
            rclcpp::Node& node,
            unsigned int soundCardTotalDelayMs = 40,
            bool echoCancellation = true,
            bool autoGainControl = true,
            bool noiseSuppression = true,
            bool highPassFilter = false,
            bool stereoSwapping = false,
            bool transientSuppression = true);

        void sendFrame(const audio_utils_msgs::msg::AudioFrame::ConstSharedPtr& msg);

    private:
        rclcpp::Node& m_node;
    };
}

#endif
