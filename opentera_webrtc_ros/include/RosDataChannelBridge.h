#ifndef OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_DATA_CHANNEL_BRIDGE_H
#define OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_DATA_CHANNEL_BRIDGE_H

#include <OpenteraWebrtcNativeClient/DataChannelClient.h>
#include <ros/subscriber.h>
#include <ros/publisher.h>
#include <std_msgs/String.h>
#include <RosWebRTCBridge.h>

namespace opentera
{

    /**
     * @brief Implement a ROS node to bridge WebRTC data channel to ROS topics
     */
    class RosDataChannelBridge: public RosWebRTCBridge<DataChannelClient>
    {
        ros::Subscriber m_dataSubscriber;
        ros::Publisher m_dataPublisher;

        void initSignalingClient(const opentera::SignalingServerConfiguration &signalingServerConfiguration);
        void initAdvertiseTopics();
        void stopAdvertiseTopics();
        void initDataChannelCallback();
        void stopDataChannelCallback();

        void onJoinSessionEvents(const std::vector<opentera_webrtc_ros_msgs::JoinSessionEvent> &events) override;
        void onStopSessionEvents(const std::vector<opentera_webrtc_ros_msgs::StopSessionEvent> &events) override;

        void onSignalingConnectionClosed() override;
        void onSignalingConnectionError(const std::string& msg) override;

        void onRosData(const std_msgs::StringConstPtr & msg);

    public:
        RosDataChannelBridge(const ros::NodeHandle& nh);
        virtual ~RosDataChannelBridge();
    };
}

#endif
