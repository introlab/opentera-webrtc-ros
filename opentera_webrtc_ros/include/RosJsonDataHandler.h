#ifndef OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_JSON_DATA_HANDLER_H
#define OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_JSON_DATA_HANDLER_H

#include <cmath>
#include <geometry_msgs/Twist.h>
#include <json.hpp>
#include <map_image_generator/ChangeMapView.h>
#include <opentera_webrtc_ros_msgs/LabelSimple.h>
#include <opentera_webrtc_ros_msgs/LabelSimpleEdit.h>
#include <opentera_webrtc_ros_msgs/PeerData.h>
#include <opentera_webrtc_ros_msgs/SetString.h>
#include <opentera_webrtc_ros_msgs/WaypointArray.h>
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>
#include <std_srvs/Empty.h>
#include <std_srvs/SetBool.h>

namespace opentera
{
    class RosJsonDataHandler
    {
    private:
        ros::Subscriber m_webrtcDataSubscriber;
        ros::Publisher m_stopPub;
        ros::Publisher m_startPub;
        ros::Publisher m_cmdVelPublisher;
        ros::Publisher m_waypointsPub;
        ros::Publisher m_navigateToLabelPub;
        ros::Publisher m_removeLabelPub;
        ros::Publisher m_addLabelPub;
        ros::Publisher m_editLabelPub;
        float m_linear_multiplier;
        float m_angular_multiplier;
        ros::ServiceClient m_dockingClient;
        ros::ServiceClient m_localizationModeClient;
        ros::ServiceClient m_mappingModeClient;
        ros::ServiceClient m_muteClient;
        ros::ServiceClient m_enableCameraClient;
        ros::ServiceClient m_changeMapViewClient;
        ros::ServiceClient m_setMovementModeClient;

    protected:
        ros::NodeHandle m_nh;
        ros::NodeHandle m_p_nh;

        virtual void onWebRTCDataReceived(
            const ros::MessageEvent<opentera_webrtc_ros_msgs::PeerData const>& event);

    public:
        RosJsonDataHandler(const ros::NodeHandle& nh, const ros::NodeHandle& p_nh);
        virtual ~RosJsonDataHandler();

        static void run();

    private:
        static opentera_webrtc_ros_msgs::Waypoint
        getWpFromData(const nlohmann::json& data);
    };
}

#endif
