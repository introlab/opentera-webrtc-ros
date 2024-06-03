#ifndef OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_DATA_CHANNEL_BRIDGE_H
#define OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_DATA_CHANNEL_BRIDGE_H

#include <OpenteraWebrtcNativeClient/DataChannelClient.h>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/string.hpp>
#include <opentera_webrtc_ros/RosWebRTCBridge.h>
#include <opentera_webrtc_ros_msgs/msg/peer_data.hpp>

namespace opentera
{

    /**
     * @brief Implement a ROS node to bridge WebRTC data channel to ROS topics
     */
    class RosDataChannelBridge : public RosWebRTCBridge<DataChannelClient>
    {
        rclcpp::Subscription<std_msgs::msg::String>::SharedPtr m_dataSubscriber;
        rclcpp::Publisher<opentera_webrtc_ros_msgs::msg::PeerData>::SharedPtr m_dataPublisher;
        rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr m_callAllSubscriber;

        void initSignalingClient(const opentera::SignalingServerConfiguration& signalingServerConfiguration);
        void initAdvertiseTopics();
        void stopAdvertiseTopics();
        void initDataChannelCallback();
        void stopDataChannelCallback();

        void onJoinSessionEvents(const std::vector<opentera_webrtc_ros_msgs::msg::JoinSessionEvent>& events) override;
        void onStopSessionEvents(const std::vector<opentera_webrtc_ros_msgs::msg::StopSessionEvent>& events) override;

        void onSignalingConnectionClosed() override;
        void onSignalingConnectionError(const std::string& msg) override;

        void onRosData(const std_msgs::msg::String::ConstSharedPtr& msg);
        void callAllCallBack(const std_msgs::msg::Empty::ConstSharedPtr& msg);

    public:
        RosDataChannelBridge();
        virtual ~RosDataChannelBridge();
    };
}

#endif
