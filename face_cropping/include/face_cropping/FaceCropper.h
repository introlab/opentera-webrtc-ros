#ifndef FACE_CROPPER_H
#define FACE_CROPPER_H

#include "face_cropping/Parameters.h"

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <opentera_webrtc_ros_msgs/PeerImage.h>
#include <opencv4/opencv2/dnn.hpp>
#include <opencv4/opencv2/objdetect.hpp>

namespace face_cropping
{
    class FaceCropper
    {
        const Parameters& m_parameters;
        ros::NodeHandle& m_nodeHandle;
        ros::Subscriber m_peerFrameSubscriber;
        ros::Publisher m_peerFramePublisher;
        image_transport::ImageTransport m_imageTransport;
        image_transport::Subscriber m_itSubscriber;
        image_transport::Publisher m_itPublisher;
        int m_pubCounter;

        cv::dnn::Net m_network;

        std::list<cv::Rect> m_cutoutList;
        cv::Rect m_oldCutout;
        float m_xAspect;
        float m_yAspect;
        float m_aspectRatio;
        std::vector<std::vector<std::tuple<int, cv::Rect>>> m_lastDetectedFaces;

    public:
        FaceCropper(Parameters& parameters, ros::NodeHandle& nodeHandle);
        virtual ~FaceCropper();

        void localFrameReceivedCallback(const sensor_msgs::ImageConstPtr& msg);
        void peerFrameReceivedCallback(const opentera_webrtc_ros_msgs::PeerImageConstPtr& msg);
        cv::CascadeClassifier faceCascade;

    private:
        std::vector<cv::Rect> detectFaces(cv::Mat& frame);
        cv::Mat cutoutFace(cv::Mat frame);
        cv::Rect getAverageRect(std::list<cv::Rect> rectangles);
        sensor_msgs::ImageConstPtr cvMatToImageConstPtr(cv::Mat frame);
        std::vector<cv::Rect> getValidFaces(std::vector<cv::Rect> detectedFaces, cv::Mat frame);
        void updateLastFacesDetected(std::vector<cv::Rect> detectedFaces, cv::Mat frame);
    };
}
#endif
