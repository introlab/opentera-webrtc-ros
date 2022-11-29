#ifndef FACE_CROPPER_H
#define FACE_CROPPER_H

#include "face_cropping/Parameters.h"

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/Image.h>
#include <opentera_webrtc_ros_msgs/PeerImage.h>
#include <opencv4/opencv2/objdetect.hpp>

namespace face_cropping
{
    using detectionVector = std::vector<std::tuple<int, std::vector<std::tuple<int, cv::Rect>>>>;

    class FaceCropper
    {
        Parameters m_parameters;
        ros::NodeHandle m_nodeHandle;
        ros::Subscriber m_enableCroppingSubscriber;
        ros::Subscriber m_peerFrameSubscriber;
        ros::Publisher m_peerFramePublisher;
        image_transport::ImageTransport m_imageTransport;
        image_transport::Subscriber m_itSubscriber;
        image_transport::Publisher m_itPublisher;
        int m_pubCounter;
        bool m_enabled;

        std::list<cv::Rect> m_cutoutList;
        cv::Rect m_oldCutout;
        float m_xAspect;
        float m_yAspect;
        float m_aspectRatio;
        detectionVector m_lastDetectedFaces;
        int m_noDetectionCounter;

    public:
        FaceCropper(Parameters& parameters, ros::NodeHandle& nodeHandle);
        virtual ~FaceCropper();

        void enableFaceCroppingCallback(const std_msgs::Bool& msg);
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
