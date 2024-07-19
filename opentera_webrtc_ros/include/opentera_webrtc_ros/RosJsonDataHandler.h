#ifndef OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_JSON_DATA_HANDLER_H
#define OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_JSON_DATA_HANDLER_H

#include <cmath>
#include <geometry_msgs/msg/twist.hpp>
#include <lib/json.hpp>
#include <opentera_webrtc_ros_msgs/srv/change_map_view.hpp>
#include <opentera_webrtc_ros_msgs/msg/label_simple.hpp>
#include <opentera_webrtc_ros_msgs/msg/label_simple_edit.hpp>
#include <opentera_webrtc_ros_msgs/msg/peer_data.hpp>
#include <opentera_webrtc_ros_msgs/srv/set_string.hpp>
#include <opentera_webrtc_ros_msgs/msg/waypoint_array.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_srvs/srv/empty.hpp>
#include <std_srvs/srv/set_bool.hpp>

#include "utils.h"

namespace opentera
{
    class RosJsonDataHandler : public rclcpp::Node
    {
    private:
        float m_linear_multiplier;
        float m_angular_multiplier;

        rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr m_stopPub;
        rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr m_startPub;
        rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr m_cmdVelPublisher;
        rclcpp::Publisher<opentera_webrtc_ros_msgs::msg::WaypointArray>::SharedPtr m_waypointsPub;
        rclcpp::Publisher<std_msgs::msg::String>::SharedPtr m_navigateToLabelPub;
        rclcpp::Publisher<std_msgs::msg::String>::SharedPtr m_removeLabelPub;
        rclcpp::Publisher<opentera_webrtc_ros_msgs::msg::LabelSimple>::SharedPtr m_addLabelPub;
        rclcpp::Publisher<opentera_webrtc_ros_msgs::msg::LabelSimpleEdit>::SharedPtr m_editLabelPub;
        rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr m_micVolumePub;
        rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr m_enableCameraPub;
        rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr m_volumePub;

        rclcpp::Subscription<opentera_webrtc_ros_msgs::msg::PeerData>::SharedPtr m_webrtcDataSubscriber;

        rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr m_dockingClient;
        rclcpp::Client<std_srvs::srv::Empty>::SharedPtr m_localizationModeClient;
        rclcpp::Client<std_srvs::srv::Empty>::SharedPtr m_mappingModeClient;
        rclcpp::Client<opentera_webrtc_ros_msgs::srv::ChangeMapView>::SharedPtr m_changeMapViewClient;
        rclcpp::Client<opentera_webrtc_ros_msgs::srv::SetString>::SharedPtr m_setMovementModeClient;
        rclcpp::Client<opentera_webrtc_ros_msgs::srv::SetString>::SharedPtr m_doMovementClient;

        ServiceClientPruner m_pruner;

    protected:
        virtual void onWebRTCDataReceived(const opentera_webrtc_ros_msgs::msg::PeerData::ConstSharedPtr& event);

    public:
        RosJsonDataHandler();
        virtual ~RosJsonDataHandler();

        void run();

    private:
        static opentera_webrtc_ros_msgs::msg::Waypoint getWpFromData(const nlohmann::json& data);
    };
}

#endif
