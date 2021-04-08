#ifndef OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_TOPIC_STREAMER_H
#define OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_TOPIC_STREAMER_H

#include <opencv2/core.hpp>
#include <ros/ros.h>
#include <RosVideoSource.h>
#include <RosAudioSource.h>
#include <OpenteraWebrtcNativeClient/StreamClient.h>
#include <OpenteraWebrtcNativeClient/Sinks/VideoSink.h>
#include <opentera_webrtc_ros_msgs/OpenTeraEvent.h>

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

        ros::Subscriber m_eventSubscriber;

        void init(bool &canSendStream,
            bool &canReceiveStream,
            bool &denoise,
            bool &screencast,
            const opentera::SignalingServerConfiguration &signalingServerConfiguration);
        void initNode();
        void initNode(const opentera::SignalingServerConfiguration &signalingServerConfiguration);

        void onVideoFrameReceived(const Client& client, const cv::Mat& bgrImg, uint64_t timestampUs);
        void onAudioFrameReceived(const Client& client,
            const void* audioData,
            int bitsPerSample,
            int sampleRate,
            size_t numberOfChannels,
            size_t numberOfFrames);
        void onEvent(const ros::MessageEvent<opentera_webrtc_ros_msgs::OpenTeraEvent const>& event);

        template <typename T>
        void publishPeerFrame(ros::Publisher& publisher, const Client& client, const decltype(T::frame)& frame);

    public:
        RosStreamBridge();
        virtual ~RosStreamBridge();

        void run();
    };

    /**
     * @brief publish a Peer message using the given node publisher
     * 
     * @param publisher ROS node publisher
     * @param client Client who sent the message
     * @param frame The message to peer with a client
     */
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
