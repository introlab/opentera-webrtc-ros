#ifndef OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_JSON_DATA_HANDLER_H
#define OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_JSON_DATA_HANDLER_H

#include <ros/ros.h>

#include <opentera_webrtc_ros_msgs/PeerData.h>

namespace opentera {

    class RosJsonDataHandler
    {

    private:
        ros::Subscriber m_webrtcDataSubscriber;
        ros::Publisher m_cmdVelPublisher;

    protected:
        ros::NodeHandle m_nh;

        virtual void onWebRTCDataReceived(const ros::MessageEvent<opentera_webrtc_ros_msgs::PeerData const>& event);

    public:
        RosJsonDataHandler(ros::NodeHandle nh);
        virtual ~RosJsonDataHandler();

        void run();

    };
}

#endif