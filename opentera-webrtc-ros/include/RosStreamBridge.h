#ifndef OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_TOPIC_STREAMER_H
#define OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_TOPIC_STREAMER_H

#include <opencv2/core.hpp>
#include <ros/ros.h>
#include <RosVideoSource.h>
#include <RosAudioSource.h>
#include <OpenteraWebrtcNativeClient/StreamClient.h>
#include <opentera_webrtc_ros_msgs/OpenTeraEvent.h>

#include <RosWebRTCBridge.h>

namespace opentera {

    /**
     * @brief A ros node that bridges streams with ROS topics
     *
     * View README.md for more details
     */
    class RosStreamBridge: public RosWebRTCBridge<StreamClient>
    {
        std::shared_ptr<RosVideoSource> m_videoSource;
        std::shared_ptr<RosAudioSource> m_audioSource;

        ros::Subscriber m_imageSubsriber;
        ros::Publisher m_imagePublisher;

        ros::Publisher m_audioPublisher;

        bool m_canSendStream;
        bool m_canReceiveStream;

        void init(const opentera::SignalingServerConfiguration &signalingServerConfiguration);

        virtual void onJoinSessionEvents(const std::vector<opentera_webrtc_ros_msgs::JoinSessionEvent> &events);
        virtual void onStopSessionEvents(const std::vector<opentera_webrtc_ros_msgs::StopSessionEvent> &events);

        virtual void onSignalingConnectionOpened();
        virtual void onSignalingConnectionClosed();
        virtual void onSignalingConnectionError(const std::string& msg);

        void onVideoFrameReceived(const Client& client, const cv::Mat& bgrImg, uint64_t timestampUs);
        void onAudioFrameReceived(const Client& client,
            const void* audioData,
            int bitsPerSample,
            int sampleRate,
            size_t numberOfChannels,
            size_t numberOfFrames);

    public:
        RosStreamBridge();
        virtual ~RosStreamBridge();
    };
}

#endif
