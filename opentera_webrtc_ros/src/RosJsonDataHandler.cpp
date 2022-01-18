#include <RosJsonDataHandler.h>

using namespace opentera;

RosJsonDataHandler::RosJsonDataHandler(const ros::NodeHandle& nh,
                                       const ros::NodeHandle& p_nh)
    : m_nh(nh), m_p_nh(p_nh)
{
    m_p_nh.param<float>("linear_multiplier", m_linear_multiplier, 0.15);
    m_p_nh.param<float>("angular_multiplier", m_angular_multiplier, 0.15);

    m_stopPub = m_nh.advertise<std_msgs::Bool>("stop", 1);
    m_startPub = m_nh.advertise<std_msgs::Bool>("start", 1);
    m_cmdVelPublisher = m_nh.advertise<geometry_msgs::Twist>("cmd_vel", 1);
    m_waypointsPub =
        m_nh.advertise<opentera_webrtc_ros_msgs::WaypointArray>("waypoints", 1);

    m_webrtcDataSubscriber =
        m_nh.subscribe("webrtc_data", 1, &RosJsonDataHandler::onWebRTCDataReceived, this);

    m_dockingClient = m_nh.serviceClient<std_srvs::SetBool>("do_docking");
    m_muteClient = m_nh.serviceClient<std_srvs::SetBool>("mute");
    m_enableCameraClient = m_nh.serviceClient<std_srvs::SetBool>("enableCamera");

    m_localizationModeClient =
        m_nh.serviceClient<std_srvs::Empty>("/rtabmap/set_mode_localization");
    m_mappingModeClient =
        m_nh.serviceClient<std_srvs::Empty>("/rtabmap/set_mode_mapping");

    m_changeMapViewClient =
        m_nh.serviceClient<map_image_generator::ChangeMapView>("change_map_view");
}

RosJsonDataHandler::~RosJsonDataHandler() = default;

void RosJsonDataHandler::onWebRTCDataReceived(
    const ros::MessageEvent<opentera_webrtc_ros_msgs::PeerData const>& event)
{
    const opentera_webrtc_ros_msgs::PeerData msg = *(event.getMessage());

    nlohmann::json serializedData = nlohmann::json::parse(msg.data);
    if (serializedData["type"] == "stop")
    {
        // TODO: should this be a service instead of a topic message?
        std_msgs::Bool msg;
        msg.data = serializedData["state"];
        m_stopPub.publish(msg);
    }
    else if (serializedData["type"] == "start")
    {
        // TODO: should this be a service instead of a topic message?
        std_msgs::Bool msg;
        msg.data = serializedData["state"];
        m_startPub.publish(msg);
    }
    else if (serializedData["type"] == "velCmd")
    {
        geometry_msgs::Twist twist;
        // Multiply by 0.15 in order to control the speed of the movement
        twist.linear.x = static_cast<double>(serializedData["x"]) * m_linear_multiplier;
        twist.angular.z =
            static_cast<double>(serializedData["yaw"]) * m_angular_multiplier;
        m_cmdVelPublisher.publish(twist);
    }
    else if (serializedData["type"] == "waypointArray")
    {
        opentera_webrtc_ros_msgs::WaypointArray wp_array;
        for (auto waypoint : serializedData["array"])
        {
            // Received waypoints are in pixel coordinates in the image frame
            opentera_webrtc_ros_msgs::Waypoint wp;
            wp.x = static_cast<float>(waypoint["coordinate"]["x"]);
            wp.y = static_cast<float>(waypoint["coordinate"]["y"]);
            wp.yaw = static_cast<float>(static_cast<double>(waypoint["coordinate"]["yaw"])
                                        * M_PI / 180);
            wp_array.waypoints.push_back(wp);
        }
        m_waypointsPub.publish(wp_array);
    }
    else if (serializedData["type"] == "action")
    {
        if (serializedData["action"] == "dock")
        {
            std_srvs::SetBool srv;
            srv.request.data = serializedData["cmd"];
            if (!m_dockingClient.call(srv))
            {
                ROS_ERROR("Docking service call error: %s", srv.response.message.c_str());
            }
        }
        else if (serializedData["action"] == "localizationMode")
        {
            std::cout << "Switching to localization mode" << std::endl;
            std_srvs::Empty srv;
            if (!m_localizationModeClient.call(srv))
            {
                ROS_ERROR("Localization mode service call error");
            }
        }
        else if (serializedData["action"] == "mappingMode")
        {
            std::cout << "Switching to mapping mode" << std::endl;
            std_srvs::Empty srv;
            if (!m_mappingModeClient.call(srv))
            {
                ROS_ERROR("Mapping mode service call error");
            }
        }
    }
    else if (serializedData["type"] == "mute")
    {
        std_srvs::SetBool srv;
        srv.request.data = serializedData["value"];
        if (!m_muteClient.call(srv))
        {
            ROS_ERROR("Mute service call error: %s", srv.response.message.c_str());
        }
    }
    else if (serializedData["type"] == "enableCamera")
    {
        std_srvs::SetBool srv;
        srv.request.data = serializedData["value"];
        if (!m_enableCameraClient.call(srv))
        {
            ROS_ERROR("EnableCamera service call error: %s",
                      srv.response.message.c_str());
        }
    }
    else if (serializedData["type"] == "changeMapView")
    {
        map_image_generator::ChangeMapView srv;
        srv.request.view_new = serializedData["new"];
        srv.request.view_old = serializedData["old"];
        if (!m_changeMapViewClient.call(srv))
        {
            ROS_ERROR("change_map_view service call error: %s",
                      srv.response.message.c_str());
        }
    }
}

/**
 * @brief Connect to server and process images forever
 */
void RosJsonDataHandler::run()
{
    ros::spin();
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
    ros::init(argc, argv, "json_data_handler");
    ros::NodeHandle nh;
    ros::NodeHandle p_nh("~");

    RosJsonDataHandler node(nh, p_nh);
    node.run();
}
