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
#include <numeric>

namespace face_cropping
{
    struct DetectionFrame
    {
        int frame;
        cv::Rect detection;
    };

    struct FaceVector
    {
        int origin;
        std::vector<DetectionFrame> face;
    };

    const cv::Scalar GREEN = cv::Scalar(0, 255, 0);
    const cv::Scalar RED = cv::Scalar(255, 0, 0);

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
        int m_frameCounter;
        bool m_enabled;

        std::list<cv::Rect> m_cutoutList;
        cv::Rect m_oldCutout;
        cv::Mat m_resizedFrame;
        float m_xAspect;
        float m_yAspect;
        float m_aspectRatio;
        std::vector<FaceVector> m_lastDetectedFaces;
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
        std::vector<cv::Rect> getValidFaces(std::vector<cv::Rect> detectedFaces, cv::Mat frame);
        void updateLastFacesDetected(std::vector<cv::Rect> detectedFaces, cv::Mat frame);
        bool imageIsModifiable();
        cv::Rect getCutoutDimensions(cv::Rect face);
        cv::Rect validateCutoutMinimumChange(cv::Rect cutout);
        cv::Rect validateAspectRatio(cv::Rect cutout, cv::Rect face);
        cv::Rect correctCutoutBoundaries(cv::Rect cutout, cv::Mat frame);

        template<typename T>
        void eraseRemoveIf(std::vector<T>& vector, std::function<bool(T&)> condition);
    };
}
#endif
