#include "opentera_webrtc_ros/RosJsonDataHandler.h"
#include "opentera_webrtc_ros/utils.h"

using namespace opentera;
using namespace std::chrono_literals;

RosJsonDataHandler::RosJsonDataHandler()
    : rclcpp::Node("json_data_handler"),
      m_linear_multiplier{static_cast<float>(this->declare_parameter("linear_multiplier", 0.15))},
      m_angular_multiplier{static_cast<float>(this->declare_parameter("angular_multiplier", 0.15))},
      m_stopPub{this->create_publisher<std_msgs::msg::Bool>("stop", 1)},
      m_startPub{this->create_publisher<std_msgs::msg::Bool>("start", 1)},
      m_cmdVelPublisher{this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 1)},
      m_waypointsPub{this->create_publisher<opentera_webrtc_ros_msgs::msg::WaypointArray>("waypoints", 1)},
      m_navigateToLabelPub{this->create_publisher<std_msgs::msg::String>("navigate_to_label", 1)},
      m_removeLabelPub{this->create_publisher<std_msgs::msg::String>("remove_label_by_name", 1)},
      m_addLabelPub{this->create_publisher<opentera_webrtc_ros_msgs::msg::LabelSimple>("add_label_simple", 1)},
      m_editLabelPub{this->create_publisher<opentera_webrtc_ros_msgs::msg::LabelSimpleEdit>("edit_label_simple", 1)},
      m_micVolumePub{this->create_publisher<std_msgs::msg::Float32>("mic_volume", 1)},
      m_enableCameraPub{this->create_publisher<std_msgs::msg::Bool>("enable_camera", 1)},
      m_volumePub{this->create_publisher<std_msgs::msg::Float32>("volume", 1)},
      m_webrtcDataSubscriber{this->create_subscription<opentera_webrtc_ros_msgs::msg::PeerData>(
          "webrtc_data",
          20,
          bind_this<opentera_webrtc_ros_msgs::msg::PeerData>(this, &RosJsonDataHandler::onWebRTCDataReceived))},
      m_dockingClient{this->create_client<std_srvs::srv::SetBool>("do_docking")},
      m_localizationModeClient{this->create_client<std_srvs::srv::Empty>("/rtabmap/rtabmap/set_mode_localization")},
      m_mappingModeClient{this->create_client<std_srvs::srv::Empty>("/rtabmap/rtabmap/set_mode_mapping")},
      m_changeMapViewClient{this->create_client<opentera_webrtc_ros_msgs::srv::ChangeMapView>("change_map_view")},
      m_setMovementModeClient{this->create_client<opentera_webrtc_ros_msgs::srv::SetString>("set_movement_mode")},
      m_doMovementClient{this->create_client<opentera_webrtc_ros_msgs::srv::SetString>("do_movement")},
      m_pruner{
          *this,
          2s,
          m_dockingClient,
          m_localizationModeClient,
          m_mappingModeClient,
          m_changeMapViewClient,
          m_setMovementModeClient,
          m_doMovementClient}
{
}

RosJsonDataHandler::~RosJsonDataHandler() = default;

opentera_webrtc_ros_msgs::msg::Waypoint RosJsonDataHandler::getWpFromData(const nlohmann::json& data)
{
    opentera_webrtc_ros_msgs::msg::Waypoint wp;
    wp.x = static_cast<float>(data["coordinate"]["x"]);
    wp.y = static_cast<float>(data["coordinate"]["y"]);
    wp.yaw = static_cast<float>(static_cast<double>(data["coordinate"]["yaw"]) * M_PI / 180);
    return wp;
}

void RosJsonDataHandler::onWebRTCDataReceived(const opentera_webrtc_ros_msgs::msg::PeerData::ConstSharedPtr& event)
{
    const opentera_webrtc_ros_msgs::msg::PeerData& msg = *event;

    nlohmann::json serializedData = nlohmann::json::parse(msg.data);
    if (serializedData["type"] == "stop")
    {
        // TODO: should this be a service instead of a topic message?
        std_msgs::msg::Bool msg;
        msg.data = serializedData["state"];
        m_stopPub->publish(msg);
    }
    else if (serializedData["type"] == "start")
    {
        // TODO: should this be a service instead of a topic message?
        std_msgs::msg::Bool msg;
        msg.data = serializedData["state"];
        m_startPub->publish(msg);
    }
    else if (serializedData["type"] == "velCmd")
    {
        geometry_msgs::msg::Twist twist;
        // Multiply by 0.15 in order to control the speed of the movement
        twist.linear.x = static_cast<double>(serializedData["x"]) * m_linear_multiplier;
        twist.angular.z = static_cast<double>(serializedData["yaw"]) * m_angular_multiplier;
        m_cmdVelPublisher->publish(twist);
    }
    else if (serializedData["type"] == "waypointArray")
    {
        opentera_webrtc_ros_msgs::msg::WaypointArray wp_array;
        for (const auto& waypoint : serializedData["array"])
        {
            wp_array.waypoints.push_back(getWpFromData(waypoint));
        }
        m_waypointsPub->publish(wp_array);
    }
    else if (serializedData["type"] == "action")
    {
        if (serializedData["action"] == "dock")
        {
            auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
            request->data = serializedData["cmd"];

            auto result = m_dockingClient->async_send_request(
                request,
                [this](rclcpp::Client<std_srvs::srv::SetBool>::SharedFuture result)
                {
                    if (!result.get()->success)
                    {
                        RCLCPP_ERROR(
                            this->get_logger(),
                            "Docking service call error: %s",
                            result.get()->message.c_str());
                    }
                });
        }
        else if (serializedData["action"] == "localizationMode")
        {
            std::cout << "Switching to localization mode" << std::endl;
            auto request = std::make_shared<std_srvs::srv::Empty::Request>();

            m_localizationModeClient->async_send_request(request);
        }
        else if (serializedData["action"] == "mappingMode")
        {
            std::cout << "Switching to mapping mode" << std::endl;
            auto request = std::make_shared<std_srvs::srv::Empty::Request>();

            m_mappingModeClient->async_send_request(request);
        }
        else if (serializedData["action"] == "setMovementMode")
        {
            auto request = std::make_shared<opentera_webrtc_ros_msgs::srv::SetString::Request>();
            request->data = serializedData["cmd"];

            auto result = m_setMovementModeClient->async_send_request(
                request,
                [this](rclcpp::Client<opentera_webrtc_ros_msgs::srv::SetString>::SharedFuture result)
                {
                    if (!result.get()->success)
                    {
                        RCLCPP_ERROR(
                            this->get_logger(),
                            "Set movement mode service call error: %s",
                            result.get()->message.c_str());
                    }
                });
        }
        else if (serializedData["action"] == "doMovement")
        {
            auto request = std::make_shared<opentera_webrtc_ros_msgs::srv::SetString::Request>();
            request->data = serializedData["cmd"];

            auto result = m_doMovementClient->async_send_request(
                request,
                [this](rclcpp::Client<opentera_webrtc_ros_msgs::srv::SetString>::SharedFuture result)
                {
                    if (!result.get()->success)
                    {
                        RCLCPP_ERROR(
                            this->get_logger(),
                            "Do movement service call error: %s",
                            result.get()->message.c_str());
                    }
                });
        }
    }
    else if (serializedData["type"] == "micVolume")
    {
        std_msgs::msg::Float32 msg;
        msg.data = serializedData["value"];
        m_micVolumePub->publish(msg);
    }
    else if (serializedData["type"] == "enableCamera")
    {
        std_msgs::msg::Bool msg;
        msg.data = serializedData["value"];
        m_enableCameraPub->publish(msg);
    }
    else if (serializedData["type"] == "volume")
    {
        std_msgs::msg::Float32 msg;
        msg.data = serializedData["value"];
        m_volumePub->publish(msg);
    }
    else if (serializedData["type"] == "changeMapView")
    {
        auto request = std::make_shared<opentera_webrtc_ros_msgs::srv::ChangeMapView::Request>();
        request->view_new = serializedData["new"];
        request->view_old = serializedData["old"];

        auto result = m_changeMapViewClient->async_send_request(
            request,
            [this](rclcpp::Client<opentera_webrtc_ros_msgs::srv::ChangeMapView>::SharedFuture result)
            {
                if (!result.get()->success)
                {
                    RCLCPP_ERROR(
                        this->get_logger(),
                        "Change map view service call error: %s",
                        result.get()->message.c_str());
                }
            });
    }
    else if (serializedData["type"] == "goToLabel")
    {
        std_msgs::msg::String msg;
        msg.data = serializedData["label"];
        m_navigateToLabelPub->publish(msg);
    }
    else if (serializedData["type"] == "removeLabel")
    {
        std_msgs::msg::String msg;
        msg.data = serializedData["label"];
        m_removeLabelPub->publish(msg);
    }
    else if (serializedData["type"] == "addLabel")
    {
        const auto& data = serializedData["label"];

        opentera_webrtc_ros_msgs::msg::LabelSimple label;

        label.name = data["name"];
        label.description = data["description"];
        label.waypoint = getWpFromData(data);

        m_addLabelPub->publish(label);
    }
    else if (serializedData["type"] == "editLabel")
    {
        const auto& data = serializedData["newLabel"];

        opentera_webrtc_ros_msgs::msg::LabelSimpleEdit labelEdit;

        labelEdit.current_name = serializedData["currentLabel"];
        labelEdit.updated.name = data["name"];
        labelEdit.updated.description = data["description"];
        if (data["coordinate"].is_null())
        {
            labelEdit.ignore_waypoint = true;
            labelEdit.updated.waypoint = opentera_webrtc_ros_msgs::msg::Waypoint();
        }
        else
        {
            labelEdit.ignore_waypoint = false;
            labelEdit.updated.waypoint = getWpFromData(data);
        }

        m_editLabelPub->publish(labelEdit);
    }
}

/**
 * @brief Connect to server and process images forever
 */
void RosJsonDataHandler::run()
{
    rclcpp::spin(this->shared_from_this());
}

/**
 * @brief runs a ROS data channel bridge
 *
 * @param argc ROS argument count
 * @param argv ROS argument values
 * @return nothing
 */
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<RosJsonDataHandler>();
    node->run();
}
