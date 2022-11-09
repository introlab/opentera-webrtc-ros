#include "face_following/FaceFollower.h"
#include <iostream>

using namespace face_following;
using namespace std;

FaceFollower::FaceFollower(Parameters& parameters, ros::NodeHandle& nodeHandle)
    : m_parameters(parameters),
      m_nodeHandle(nodeHandle)
{
    std::string deployPath = m_parameters.dnnDeployPath();
    std::string modelPath = m_parameters.dnnModelPath();
    m_network = cv::dnn::readNetFromCaffe(deployPath, modelPath);

    m_oldCutout = cv::Rect(0, 0, 0, 0);
    m_xAspect = m_parameters.width();
    m_yAspect = m_parameters.height();
    m_aspectRatio = m_yAspect / m_xAspect;

    int d = getGCD(m_xAspect, m_yAspect);

    m_xAspect = m_xAspect / d;
    m_yAspect = m_yAspect / d;

    m_pubCounter = 0;
    image_transport::ImageTransport it(m_nodeHandle);
    if (m_parameters.isPeerImage())
    {
        m_peerFrameSubscriber =
            m_nodeHandle.subscribe("input_image", 10, &FaceFollower::peerFrameReceivedCallback, this);
        m_peerFramePublisher = m_nodeHandle.advertise<opentera_webrtc_ros_msgs::PeerImage>("output_image", 10, false);
    }
    else
    {
        m_itSubscriber =
            it.subscribe("input_image", 1, boost::bind(&FaceFollower::localFrameReceivedCallback, this, _1));
        m_itPublisher = it.advertise("output_image", 1);
    }
}

FaceFollower::~FaceFollower() = default;

void FaceFollower::localFrameReceivedCallback(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImagePtr cvPtr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
    cv::Mat frame = cutoutFace(cvPtr->image);
    sensor_msgs::ImageConstPtr processedMsg = cvMatToImageConstPtr(frame);
    m_itPublisher.publish(processedMsg);
}

void FaceFollower::peerFrameReceivedCallback(const opentera_webrtc_ros_msgs::PeerImageConstPtr& msg)
{
    cv_bridge::CvImagePtr cvPtr = cv_bridge::toCvCopy(msg->frame, sensor_msgs::image_encodings::BGR8);
    cv::Mat frame = cutoutFace(cvPtr->image);
    sensor_msgs::ImageConstPtr processedMsg = cvMatToImageConstPtr(frame);
    opentera_webrtc_ros_msgs::PeerImage newPeerImage;
    newPeerImage.frame = *processedMsg;
    newPeerImage.sender = msg->sender;
    m_peerFramePublisher.publish(newPeerImage);
}

sensor_msgs::ImageConstPtr FaceFollower::cvMatToImageConstPtr(cv::Mat frame)
{
    std_msgs::Header header;
    header.seq = m_pubCounter;
    m_pubCounter++;
    header.stamp = ros::Time::now();
    sensor_msgs::ImageConstPtr processedMsg =
        cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8, frame).toImageMsg();
    return processedMsg;
}

cv::Mat FaceFollower::cutoutFace(cv::Mat frame)
{
    std::vector<cv::Rect> faces = detectFaces(frame);

    // Stabilisation
    if (!faces.empty() && faces.size() == 1)
    {
        cv::Rect cutout = m_oldCutout;
        cv::Rect r = faces[0];
        if (r.height / static_cast<float>(r.width) >= m_aspectRatio)
        {
            int topMargin = m_parameters.topMargin() * r.height;
            int bottomMargin = m_parameters.bottomMargin() * r.height;
            cutout.height = getClosestNumberDividableBy(r.height + topMargin + bottomMargin, m_yAspect / m_xAspect);
            cutout.y = 0;
            if (r.y >= topMargin)
            {
                cutout.y = r.y - topMargin;
            }

            cutout.width = cutout.height / m_aspectRatio;
            float xMargin = static_cast<float>(cutout.width - r.width) / 2;
            cutout.x = 0;
            if (r.x >= xMargin)
            {
                cutout.x = r.x - xMargin;
            }
        }
        else
        {
            int leftMargin = m_parameters.leftMargin() * r.width;
            int rightMargin = m_parameters.rightMargin() * r.width;

            cutout.width = getClosestNumberDividableBy((r.width + leftMargin + rightMargin), m_xAspect / m_yAspect);
            cutout.x = 0;
            if (r.x >= leftMargin)
            {
                cutout.x = r.x - leftMargin;
            }

            cutout.height = cutout.width * m_aspectRatio;
            int yMargin = static_cast<float>(cutout.height - r.height) / 2;
            cutout.y = 0;
            if (r.y >= yMargin)
            {
                cutout.y = r.y - yMargin;
            }
        }

        int minWidthChange = m_oldCutout.width * m_parameters.minWidthChange();
        int minHeightChange = m_oldCutout.height * m_parameters.minHeightChange();
        int minXChange = m_oldCutout.width * m_parameters.minXChange();
        int minYChange = m_oldCutout.height * m_parameters.minYChange();

        if (cutout.height < m_oldCutout.height + minHeightChange &&
            cutout.height > m_oldCutout.height - minHeightChange && cutout.width < m_oldCutout.width + minWidthChange &&
            cutout.width > m_oldCutout.width - minWidthChange)
        {
            cutout.height = m_oldCutout.height;
            cutout.width = m_oldCutout.width;
        }
        if (cutout.y < m_oldCutout.y + minYChange && cutout.y > m_oldCutout.y - minYChange)
        {
            cutout.y = m_oldCutout.y;
        }
        if (cutout.x < m_oldCutout.x + minXChange && cutout.x > m_oldCutout.x - minXChange)
        {
            cutout.x = m_oldCutout.x;
        }

        m_cutoutList.push_back(cutout);
        if (m_cutoutList.size() > m_parameters.framesUsedForStabilizer())
        {
            m_cutoutList.pop_front();
            cutout = getAverageRect(m_cutoutList);
        }

        if (r.height / static_cast<float>(r.width) >= m_aspectRatio)
        {
            cutout.height = getClosestNumberDividableBy(cutout.height, m_yAspect / m_xAspect);
            cutout.width = cutout.height / m_aspectRatio;
        }
        else
        {
            cutout.width = getClosestNumberDividableBy(cutout.width, m_xAspect / m_yAspect);
            cutout.height = cutout.width * m_aspectRatio;
        }

        if (cutout.width < m_oldCutout.width + m_xAspect && cutout.width > m_oldCutout.width - m_xAspect)
        {
            cutout.width = m_oldCutout.width;
        }
        if (cutout.height < m_oldCutout.height + m_yAspect && cutout.height > m_oldCutout.height - m_yAspect)
        {
            cutout.height = m_oldCutout.height;
        }

        if (cutout.width <= 0)
        {
            cutout.width = 1;
        }
        if (cutout.width + cutout.x > frame.size().width)
        {
            cutout.x = frame.size().width - cutout.width;
        }
        if (cutout.height <= 0)
        {
            cutout.height = 1;
        }
        if (cutout.height + cutout.y > frame.size().height)
        {
            cutout.y = frame.size().height - cutout.height;
        }

        frame = frame(cutout);
        m_oldCutout = cutout;
    }
    else if (faces.size() == 0)
    {
        frame = frame(m_oldCutout);
    }

    cv::resize(frame, frame, cv::Size(m_parameters.width(), m_parameters.height()));
    return frame;
}

std::vector<cv::Rect> FaceFollower::detectFaces(const cv::Mat& frame)
{
    float confidenceThreshold = 0.5;
    int inputHeight = 300;
    int inputWidth = 300;
    double scale = 1.0;
    cv::Scalar meanValues = {104.0, 177.0, 123.0};

    if(m_parameters.useGpu())
    {
        m_network.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        m_network.setPreferableTarget(cv::dnn::DNN_BACKEND_CUDA);
    }
    cv::Mat blob =
        cv::dnn::blobFromImage(frame, scale, cv::Size(inputWidth, inputHeight), meanValues, false, false);
    m_network.setInput(blob, "data");
    cv::Mat detection = m_network.forward("detection_out");
    cv::Mat detectionMatrix(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    std::vector<cv::Rect> faces;

    for (int i = 0; i < detectionMatrix.rows; i++)
    {
        float confidence = detectionMatrix.at<float>(i, 2);

        if (confidence > confidenceThreshold)
        {
            int bottomLeftX = static_cast<int>(detectionMatrix.at<float>(i, 3) * frame.cols);
            int bottomLeftY = static_cast<int>(detectionMatrix.at<float>(i, 4) * frame.rows);
            int topRightX = static_cast<int>(detectionMatrix.at<float>(i, 5) * frame.cols);
            int topRightY = static_cast<int>(detectionMatrix.at<float>(i, 6) * frame.rows);

            faces.emplace_back(bottomLeftX, bottomLeftY, (topRightX - bottomLeftX), (topRightY - bottomLeftY));
        }
    }

    return faces;
}

int FaceFollower::getGCD(int a, int b)
{
    if (b == 0)
    {
        return a;
    }
    return getGCD(b, a % b);
}

int FaceFollower::getClosestNumberDividableBy(float a, float b)
{
    float n1 = a;
    while (ceilf(n1 / b) / floorf(n1 / b) != 1.0)
    {
        n1 += 1;
    }
    float n2 = a;
    while (ceilf(n2 / b) / floorf(n2 / b) != 1.0)
    {
        n2 -= 1;
    }
    if (abs(n1 - a) > abs(a - n2))
    {
        return n1;
    }
    return n2;
}

cv::Rect FaceFollower::getAverageRect(std::list<cv::Rect> list)
{
    cv::Rect r(0, 0, 0, 0);
    if (!list.empty())
    {
        int i = 0;
        for (cv::Rect& rect : list)
        {
            i++;
            r.x += rect.x;
            r.y += rect.y;
            r.width += rect.width;
            r.height += rect.height;
        }
        r.x /= i;
        r.y /= i;

        float newWidth = r.width / i;
        if (ceilf(newWidth) / floorf(newWidth) == 1.0)
        {
            r.width = newWidth;
        }
        else
        {
            r.width = static_cast<int>(round(newWidth));
        }

        float newHeight = r.height / i;
        if (ceilf(newHeight) / floorf(newHeight) == 1.0)
        {
            r.height = newHeight;
        }
        else if (abs(ceilf(newHeight) - newHeight) > abs(newHeight - floorf(newHeight)))
        {
            r.height = ceilf(newHeight);
        }
        else
        {
            r.height = floorf(newHeight);
        }
    }
    return r;
}