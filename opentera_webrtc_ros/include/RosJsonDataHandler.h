#ifndef OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_JSON_DATA_HANDLER_H
#define OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_JSON_DATA_HANDLER_H

#include <cmath>
#include <json.hpp>
#include <ros/ros.h>
#include <opentera_webrtc_ros_msgs/PeerData.h>
#include <opentera_webrtc_ros_msgs/Waypoint.h>
#include <opentera_webrtc_ros_msgs/WaypointArray.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>

namespace opentera 
{
    
    class RosJsonDataHandler
    {

    private:
        ros::Subscriber m_webrtcDataSubscriber;
        ros::Publisher m_cmdVelPublisher;
        ros::Publisher m_waypointsPub;
        float m_linear_multiplier;
        float m_angular_multiplier;

    protected:
        ros::NodeHandle m_nh;
        ros::NodeHandle m_p_nh;

        virtual void onWebRTCDataReceived(const ros::MessageEvent<opentera_webrtc_ros_msgs::PeerData const>& event);

    public:
        RosJsonDataHandler(ros::NodeHandle nh, ros::NodeHandle p_nh);
        virtual ~RosJsonDataHandler();

        void run();

    };
}

#endif