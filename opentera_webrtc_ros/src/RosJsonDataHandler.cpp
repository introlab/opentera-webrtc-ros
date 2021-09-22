#include <RosJsonDataHandler.h>

#include <json.hpp>

#include <geometry_msgs/Twist.h>

using namespace opentera;

RosJsonDataHandler::RosJsonDataHandler(ros::NodeHandle nh): m_nh(nh)
{
    m_webrtcDataSubscriber = m_nh.subscribe("webrtc_data", 1, &RosJsonDataHandler::onWebRTCDataReceived, this);
    m_cmdVelPublisher = m_nh.advertise<geometry_msgs::Twist>("cmd_vel", 1);
}

RosJsonDataHandler::~RosJsonDataHandler()
{

}

void RosJsonDataHandler::onWebRTCDataReceived(const ros::MessageEvent<opentera_webrtc_ros_msgs::PeerData const>& event)
{
    const opentera_webrtc_ros_msgs::PeerData msg = *(event.getMessage());

    nlohmann::json serializedData = nlohmann::json::parse(msg.data);

    if (serializedData["type"] == "velCmd")
    {
        geometry_msgs::Twist twist;
        // Multiply by 0.15 in order to control the speed of the movement
        twist.linear.x = (double)serializedData["x"] * 0.15;
        twist.angular.z = (double)serializedData["z"] * 0.15;
        m_cmdVelPublisher.publish(twist);
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

    RosJsonDataHandler node(nh);
    node.run();
}