#include <RosJsonDataHandler.h>

using namespace opentera;

RosJsonDataHandler::RosJsonDataHandler(ros::NodeHandle nh, ros::NodeHandle p_nh) : 
    m_nh(nh),
    m_p_nh(p_nh)
{
    m_webrtcDataSubscriber = m_nh.subscribe("webrtc_data", 1, &RosJsonDataHandler::onWebRTCDataReceived, this);
    m_stopPub = m_nh.advertise<std_msgs::Bool>("stop", 1);
    m_cmdVelPublisher = m_nh.advertise<geometry_msgs::Twist>("cmd_vel", 1);
    m_waypointsPub = m_nh.advertise<opentera_webrtc_ros_msgs::WaypointArray>("waypoints", 1);
    m_p_nh.param<float>("linear_multiplier", m_linear_multiplier, 0.15);
    m_p_nh.param<float>("angular_multiplier", m_angular_multiplier, 0.15);
    m_dockingClient = m_nh.serviceClient<std_srvs::SetBool>("do_docking");
}

RosJsonDataHandler::~RosJsonDataHandler()
{

}

void RosJsonDataHandler::onWebRTCDataReceived(const ros::MessageEvent<opentera_webrtc_ros_msgs::PeerData const>& event)
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
    else if (serializedData["type"] == "velCmd")
    {
        geometry_msgs::Twist twist;
        // Multiply by 0.15 in order to control the speed of the movement
        twist.linear.x = (double)serializedData["x"] * m_linear_multiplier;
        twist.angular.z = (double)serializedData["yaw"] * m_angular_multiplier;
        m_cmdVelPublisher.publish(twist);
    }
    else if (serializedData["type"] == "waypointArray")
    {
        opentera_webrtc_ros_msgs::WaypointArray wp_array;
        for(auto waypoint : serializedData["array"])
        {
            // Received waypoints are in pixel coordinates in the image frame
            opentera_webrtc_ros_msgs::Waypoint wp;
            wp.x = (double)waypoint["coordinate"]["x"];
            wp.y = (double)waypoint["coordinate"]["y"];
            wp.yaw = (double)waypoint["coordinate"]["yaw"] * M_PI/180;
            wp_array.waypoints.push_back(wp);
        }
        m_waypointsPub.publish(wp_array);
    }
    else if (serializedData["type"] == "action")
    {
        if(serializedData["action"] == "dock")
        {
            std_srvs::SetBool srv;
            srv.request.data = serializedData["cmd"];            
            if (!m_dockingClient.call(srv))
            {
                ROS_INFO("Error: %s", srv.response.message.c_str());
            }
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