#ifndef OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_TOPIC_STREAMER_H
#define OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_TOPIC_STREAMER_H

#include <opencv2/core.hpp>
#include <ros/ros.h>
#include <RosVideoSource.h>
#include <RosAudioSource.h>
#include <OpenteraWebrtcNativeClient/StreamClient.h>
#include <OpenteraWebrtcNativeClient/Sinks/VideoSink.h>

namespace opentera {

    /**
     * @brief A ros node that bridges streams with ROS topics
     *
     * View README.md for more details
     */
    class RosStreamBridge
    {
        std::string nodeName;

        ros::NodeHandle m_nh;
        std::shared_ptr<RosVideoSource> m_videoSource;
        std::shared_ptr<RosAudioSource> m_audioSource;
        std::shared_ptr<VideoSink> m_videoSink;
        std::unique_ptr<StreamClient> m_signalingClient;

        ros::Subscriber m_imageSubsriber;
        ros::Publisher m_imagePublisher;

        ros::Publisher m_audioPublisher;


        static void loadStreamParams(bool &denoise, bool &screencast);
        static void loadNodeParams(bool &canSendStream, bool &canReceiveSTream);
        void onFrameReceived(const cv::Mat& bgrImg, uint64_t timestampUs);

        template <typename T>
        void publishPeerFrame(ros::Publisher& publisher, const Client& client, const decltype(T::frame)& frame);

    public:
        RosStreamBridge();
        virtual ~RosStreamBridge();

        void run();
    };

    template <typename T>
    void RosStreamBridge::publishPeerFrame(ros::Publisher& publisher, const Client& client, const decltype(T::frame)& frame)
    {
        T peerFrameMsg;
        peerFrameMsg.sender.id = client.id();
        peerFrameMsg.sender.name = client.name();
        peerFrameMsg.frame = frame;

        publisher.publish(peerFrameMsg);
    }

}

#endif
