#include <opentera_webrtc_ros/RosAudioSource.h>
#include <rclcpp/rclcpp.hpp>

using namespace opentera;


RosAudioSource::RosAudioSource(
    rclcpp::Node& node,
    unsigned int soundCardTotalDelayMs,
    bool echoCancellation,
    bool autoGainControl,
    bool noiseSuppression,
    bool highPassFilter,
    bool stereoSwapping,
    bool transientSuppression)
    // Creating a default configuration for now, int bitsPerSample, int sampleRate, size_t numberOfChannels);
    : AudioSource(
          AudioSourceConfiguration::create(
              soundCardTotalDelayMs,
              echoCancellation,
              autoGainControl,
              noiseSuppression,
              highPassFilter,
              stereoSwapping,
              transientSuppression),
          16,
          48000,
          1),
      m_node{node}
{
}

void RosAudioSource::sendFrame(const audio_utils_msgs::msg::AudioFrame::ConstSharedPtr& msg)
{
    // Checking if frame fits default configuration and send frame
    if (msg->channel_count == 1 && msg->sampling_frequency == 48000 && msg->format == "signed_16")
    {
        // ROS_INFO("Frame size %i", msg->frame_sample_count);
        AudioSource::sendFrame(msg->data.data(), msg->frame_sample_count);
    }
    else
    {
        RCLCPP_ERROR(
            m_node.get_logger(),
            "Invalid audio frame (channel_count=%i, sampling_frequency=%i, format=%s)",
            msg->channel_count,
            msg->sampling_frequency,
            msg->format.c_str());
    }
}
