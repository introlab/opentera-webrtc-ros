#ifndef FACE_FOLLOWER_H
#define FACE_FOLLOWER_H

#include "face_following/Parameters.h"

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <opentera_webrtc_ros_msgs/PeerImage.h>
#include <opencv4/opencv2/dnn.hpp>

namespace face_following
{
    class FaceFollower
    {
        const Parameters& m_parameters;
        ros::NodeHandle& m_nodeHandle;
        ros::Subscriber m_peerFrameSubscriber;
        ros::Publisher m_peerFramePublisher;
        image_transport::Subscriber m_itSubscriber;
        image_transport::Publisher m_itPublisher;
        int m_pubCounter;

        cv::dnn::Net m_network;

        std::list<cv::Rect> m_cutoutList;
        cv::Rect m_oldCutout;
        float m_xAspect;
        float m_yAspect;
        float m_aspectRatio;

    public:
        FaceFollower(Parameters& parameters, ros::NodeHandle& nodeHandle);
        virtual ~FaceFollower();

        void localFrameReceivedCallback(const sensor_msgs::ImageConstPtr& msg);
        void peerFrameReceivedCallback(const opentera_webrtc_ros_msgs::PeerImageConstPtr& msg);

    private:
        std::vector<cv::Rect> detectFaces(const cv::Mat& frame);
        cv::Mat cutoutFace(cv::Mat frame);
        cv::Rect getAverageRect(std::list<cv::Rect> rectangles);
        int getClosestNumberDividableBy(float a, float b);
        int getGCD(int num, int den);
        sensor_msgs::ImageConstPtr cvMatToImageConstPtr(cv::Mat frame);
    };
}
#endif
