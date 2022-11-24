#include "face_cropping/FaceCropper.h"
#include "face_cropping/MathUtils.h"
#include <iostream>

using namespace face_cropping;
using namespace std;

FaceCropper::FaceCropper(Parameters& parameters, ros::NodeHandle& nodeHandle)
    : m_parameters(parameters),
      m_nodeHandle(nodeHandle),
      m_imageTransport(nodeHandle)
{
    m_oldCutout = cv::Rect(0, 0, 0, 0);
    m_xAspect = m_parameters.width();
    m_yAspect = m_parameters.height();
    m_aspectRatio = m_yAspect / m_xAspect;

    int d = getGCD(m_xAspect, m_yAspect);

    m_xAspect = m_xAspect / d;
    m_yAspect = m_yAspect / d;
    if (m_parameters.useLbp())
    {
        std::cout << "use lbp" << std::endl;
        faceCascade.load(m_parameters.lbpCascadePath());
    }
    else
    {
        std::cout << "use haar" << std::endl;
        faceCascade.load(m_parameters.haarCascadePath());
    }

    m_pubCounter = 0;
    if (m_parameters.isPeerImage())
    {
        m_peerFrameSubscriber =
            m_nodeHandle.subscribe("input_image", 10, &FaceCropper::peerFrameReceivedCallback, this);
        m_peerFramePublisher = m_nodeHandle.advertise<opentera_webrtc_ros_msgs::PeerImage>("output_image", 10, false);
    }
    else
    {
        m_itSubscriber = m_imageTransport.subscribe(
            "input_image",
            1,
            [this](const sensor_msgs::ImageConstPtr& msg) { localFrameReceivedCallback(msg); });
        m_itPublisher = m_imageTransport.advertise("output_image", 1);
    }
}

FaceCropper::~FaceCropper() = default;

void FaceCropper::localFrameReceivedCallback(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImagePtr cvPtr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
    cv::Mat frame = cutoutFace(cvPtr->image);
    sensor_msgs::ImageConstPtr processedMsg = cvMatToImageConstPtr(frame);
    m_itPublisher.publish(processedMsg);
}

void FaceCropper::peerFrameReceivedCallback(const opentera_webrtc_ros_msgs::PeerImageConstPtr& msg)
{
    cv_bridge::CvImagePtr cvPtr = cv_bridge::toCvCopy(msg->frame, sensor_msgs::image_encodings::BGR8);
    cv::Mat frame = cutoutFace(cvPtr->image);
    sensor_msgs::ImageConstPtr processedMsg = cvMatToImageConstPtr(frame);
    opentera_webrtc_ros_msgs::PeerImage newPeerImage;
    newPeerImage.frame = *processedMsg;
    newPeerImage.sender = msg->sender;
    m_peerFramePublisher.publish(newPeerImage);
}

sensor_msgs::ImageConstPtr FaceCropper::cvMatToImageConstPtr(cv::Mat frame)
{
    std_msgs::Header header;
    header.seq = m_pubCounter;
    m_pubCounter++;
    header.stamp = ros::Time::now();
    sensor_msgs::ImageConstPtr processedMsg =
        cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8, frame).toImageMsg();
    return processedMsg;
}

cv::Mat FaceCropper::cutoutFace(cv::Mat frame)
{
    if(m_pubCounter == 0 )
    {
        m_oldCutout.width = frame.size().width;
        m_oldCutout.height = frame.size().height;
    }

    std::vector<cv::Rect> faces;
    if(m_pubCounter % m_parameters.detectionFrames() == 0)
    {
        faces = detectFaces(frame);
    }

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

        // Ensures that the aspect ratio is preserved after calculating the new cutout
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

        if (cutout.width < m_oldCutout.width + m_xAspect && cutout.width > m_oldCutout.width - m_xAspect ||
            cutout.height < m_oldCutout.height + m_yAspect && cutout.height > m_oldCutout.height - m_yAspect)
        {
            cutout.width = m_oldCutout.width;
            cutout.height = m_oldCutout.height;
        }

        // Ensures that the cutout doesn't go beyond the boundaries of the original frame
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

    if (faces.size() <= 1 && !frame.empty())
    {
        cv::resize(frame, frame, cv::Size(m_parameters.width(), m_parameters.height()));
    }
    return frame;
}

std::vector<cv::Rect> FaceCropper::detectFaces(cv::Mat& frame)
{
    cv::Mat resizedFrame = frame.clone();
    int newWidth = m_parameters.detectionScale()*frame.size().width;
    int newHeight = m_parameters.detectionScale()*frame.size().height;

    cv::resize(frame, resizedFrame, cv::Size(newWidth, newHeight));
    
    std::vector<cv::Rect> detectedFaces;
    std::vector<cv::Rect> validFaces;

    cv::Mat gray;

    cv::cvtColor(resizedFrame, gray, cv::COLOR_RGBA2GRAY, 0);

    faceCascade.detectMultiScale(gray, detectedFaces, 1.1, 3, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(m_parameters.minFaceWidth(), m_parameters.minFaceHeight()));

    for(cv::Rect & r : detectedFaces)
    {
        r.x /= m_parameters.detectionScale();
        r.y /= m_parameters.detectionScale();
        r.width /= m_parameters.detectionScale();
        r.height /= m_parameters.detectionScale();
    }
    std::cout << "visages detecté: " << detectedFaces.size() << std::endl;

    validFaces = getValidFaces(detectedFaces, frame);
        
    return validFaces;
}

cv::Rect FaceCropper::getAverageRect(std::list<cv::Rect> list)
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
        if (fabsf(roundf(newWidth) - newWidth) > 0.00001f)
        {
            r.width = newWidth;
        }
        else
        {
            r.width = static_cast<int>(round(newWidth));
        }

        float newHeight = r.height / i;
        if (fabsf(roundf(newHeight) - newHeight) > 0.00001f)
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

std::vector<cv::Rect> FaceCropper::getValidFaces(std::vector<cv::Rect> detectedFaces, cv::Mat frame)
{
    updateLastFacesDetected(detectedFaces, frame);

    std::vector<cv::Rect> validFaces;

    float maxOverlapPercentage = 0.15;

    for(int i=0; i<m_lastDetectedFaces.size(); i++) //changer loop
    {
        std::vector<std::tuple<int, cv::Rect>> faceVector = m_lastDetectedFaces[i];

        std::vector<std::vector<std::tuple<int, cv::Rect>>> otherFaceVectors = m_lastDetectedFaces;
        otherFaceVectors.erase(otherFaceVectors.begin() + i);
        bool overlaps = false;
        cv::Rect face1 = get<1>(faceVector.back());
        for(std::vector<std::tuple<int, cv::Rect>> & otherFaceVector : otherFaceVectors)
        {
            cv::Rect face2 = get<1>(otherFaceVector.back());
            if((face1 & face2).area() > face1.area() * maxOverlapPercentage && faceVector.size() < otherFaceVector.size())
            {
                overlaps = true;
            }
        }

        if(faceVector.size() > m_parameters.faceStoringFrames() * m_parameters.validFaceMinTime() && !overlaps)
        {
            cv::rectangle(frame, get<1>(faceVector.back()), cv::Scalar(0, 255, 0), 2);
            validFaces.emplace_back(get<1>(faceVector.back()));
        } 
    }
    std::cout << "valid faces size: " << validFaces.size() << std::endl;
    std::cout << "m_lastDetectedFaces size: " << m_lastDetectedFaces.size() << std::endl;
    return validFaces;
}

void FaceCropper::updateLastFacesDetected(std::vector<cv::Rect> detectedFaces, cv::Mat frame)
{
    for(std::vector<std::vector<std::tuple<int, cv::Rect>>>::iterator it = m_lastDetectedFaces.begin(); it != m_lastDetectedFaces.end(); it++)
    {
        (*it).erase(std::remove_if(std::begin(*it), std::end(*it), [this](std::tuple<int, cv::Rect> & item)
        {
            return get<0>(item) < m_pubCounter - m_parameters.faceStoringFrames(); 
        }), std::end(*it));
    }
    m_lastDetectedFaces.erase(std::remove_if(std::begin(m_lastDetectedFaces), std::end(m_lastDetectedFaces), [this](std::vector<std::tuple<int, cv::Rect>> & item)
    {
        return item.size() == 0; 
    }), std::end(m_lastDetectedFaces));

    for(cv::Rect & detectedFace : detectedFaces)
    {
        bool isNewFace = true;
        if(!m_lastDetectedFaces.empty())
        {
            for(std::vector<std::vector<std::tuple<int, cv::Rect>>>::iterator it = m_lastDetectedFaces.begin(); it != m_lastDetectedFaces.end(); it++)
            {
                std::tuple<int, cv::Rect>& tuple = (*it).back();
                cv::Rect oldFace = get<1>(tuple);
                
                float stepX = detectedFace.x * m_parameters.maxPositionStep();
                float stepY = detectedFace.y * m_parameters.maxPositionStep();
                float stepW = detectedFace.width * m_parameters.maxSizeStep();
                float stepH = detectedFace.height * m_parameters.maxSizeStep();

                if(detectedFace.x < oldFace.x + stepX && detectedFace.x > oldFace.x - stepX &&
                    detectedFace.y < oldFace.y + stepY && detectedFace.y > oldFace.y - stepY &&
                    detectedFace.width < oldFace.width + stepW && detectedFace.width > oldFace.width - stepW &&
                    detectedFace.height < oldFace.height + stepH && detectedFace.height > oldFace.height - stepH)
                {
                    isNewFace = false;
                    ((*it).emplace_back(make_tuple(m_pubCounter, detectedFace)));
                    if((*it).size() > m_parameters.faceStoringFrames() * m_parameters.validFaceMinTime())
                    {
                        cv::rectangle(frame, detectedFace, cv::Scalar(0, 255, 0), 2);
                    }
                    else
                    {
                        cv::rectangle(frame, detectedFace, cv::Scalar(255, 0, 0), 2);
                    }
                    break;
                }
            }
        }
        if(isNewFace)
        {
            cv::rectangle(frame, detectedFace, cv::Scalar(255, 0, 0), 2);
            std::vector<std::tuple<int, cv::Rect>> newVect;
            newVect.emplace_back(make_tuple(m_pubCounter, detectedFace));
            m_lastDetectedFaces.emplace_back(newVect);
        }
    }
}
