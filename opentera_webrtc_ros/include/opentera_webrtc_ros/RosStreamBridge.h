#ifndef OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_TOPIC_STREAMER_H
#define OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_TOPIC_STREAMER_H

#include <opencv2/core.hpp>
#include <opentera_webrtc_ros_msgs/msg/peer_image.hpp>
#include <opentera_webrtc_ros_msgs/msg/peer_audio.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/float32.hpp>
#include <opentera_webrtc_ros/RosVideoSource.h>
#include <opentera_webrtc_ros/RosAudioSource.h>
#include <OpenteraWebrtcNativeClient/StreamClient.h>
#include <opentera_webrtc_ros_msgs/msg/open_tera_event.hpp>

#include <opentera_webrtc_ros/RosWebRTCBridge.h>

namespace opentera
{

    /**
     * @brief A ros node that bridges streams with ROS topics
     *
     * View README.md for more details
     */
    class RosStreamBridge : public RosWebRTCBridge<StreamClient>
    {
        std::shared_ptr<RosVideoSource> m_videoSource;
        std::shared_ptr<RosAudioSource> m_audioSource;

        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr m_imageSubscriber;
        rclcpp::Subscription<audio_utils_msgs::msg::AudioFrame>::SharedPtr m_audioSubscriber;
        rclcpp::Publisher<opentera_webrtc_ros_msgs::msg::PeerImage>::SharedPtr m_imagePublisher;
        rclcpp::Publisher<opentera_webrtc_ros_msgs::msg::PeerAudio>::SharedPtr m_audioPublisher;
        rclcpp::Publisher<audio_utils_msgs::msg::AudioFrame>::SharedPtr m_mixedAudioPublisher;

        rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr m_callAllSubscriber;
        rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr m_micVolumeSubscriber;
        rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr m_enableCameraSubscriber;
        rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr m_volumeSubscriber;

        bool m_canSendAudioStream;
        bool m_canSendVideoStream;
        bool m_canReceiveAudioStream;
        bool m_canReceiveVideoStream;

        void init(
            const opentera::SignalingServerConfiguration& signalingServerConfiguration,
            const opentera::VideoStreamConfiguration& videoStreamConfiguration);

        void onJoinSessionEvents(const std::vector<opentera_webrtc_ros_msgs::msg::JoinSessionEvent>& events) override;
        void onStopSessionEvents(const std::vector<opentera_webrtc_ros_msgs::msg::StopSessionEvent>& events) override;

        void onSignalingConnectionOpened() override;
        void onSignalingConnectionClosed() override;

        void onSignalingConnectionError(const std::string& msg) override;

        void onVideoFrameReceived(const Client& client, const cv::Mat& bgrImg, uint64_t timestampUs);
        void onAudioFrameReceived(
            const Client& client,
            const void* audioData,
            int bitsPerSample,
            int sampleRate,
            size_t numberOfChannels,
            size_t numberOfFrames);
        void onMixedAudioFrameReceived(
            const void* audioData,
            int bitsPerSample,
            int sampleRate,
            size_t numberOfChannels,
            size_t numberOfFrames);

        audio_utils_msgs::msg::AudioFrame createAudioFrame(
            const void* audioData,
            int bitsPerSample,
            int sampleRate,
            size_t numberOfChannels,
            size_t numberOfFrames);

        void audioCallback(const audio_utils_msgs::msg::AudioFrame::ConstSharedPtr& msg);
        void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);

        void callAllCallBack(const std_msgs::msg::Empty::ConstSharedPtr& msg);
        void micVolumeCallback(const std_msgs::msg::Float32::ConstSharedPtr& msg);
        void enableCameraCallback(const std_msgs::msg::Bool::ConstSharedPtr& msg);
        void volumeCallback(const std_msgs::msg::Float32::ConstSharedPtr& msg);

    public:
        RosStreamBridge();
        virtual ~RosStreamBridge();
    };
}

#endif
