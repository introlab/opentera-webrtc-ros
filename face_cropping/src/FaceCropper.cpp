#include "face_cropping/FaceCropper.h"
#include "face_cropping/MathUtils.h"

using namespace face_cropping;
using namespace std;

FaceCropper::FaceCropper(Parameters& parameters, ros::NodeHandle& nodeHandle)
    : m_parameters(parameters),
      m_nodeHandle(nodeHandle),
      m_imageTransport(nodeHandle)
{
    m_noDetectionCounter = 0;
    m_oldCutout = cv::Rect(0, 0, 0, 0);
    m_xAspect = m_parameters.width();
    m_yAspect = m_parameters.height();
    m_aspectRatio = m_yAspect / m_xAspect;

    int d = getGCD(m_xAspect, m_yAspect);

    m_xAspect = m_xAspect / d;
    m_yAspect = m_yAspect / d;
    if (m_parameters.useLbp())
    {
        faceCascade.load(m_parameters.lbpCascadePath());
    }
    else
    {
        faceCascade.load(m_parameters.haarCascadePath());
    }

    m_frameCounter = 0;
    m_enabled = true;
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
    m_enableCroppingSubscriber =
        m_nodeHandle.subscribe("enable_face_cropping", 10, &FaceCropper::enableFaceCroppingCallback, this);
}

FaceCropper::~FaceCropper() = default;

void FaceCropper::enableFaceCroppingCallback(const std_msgs::Bool& msg)
{
    m_enabled = msg.data;
}

void FaceCropper::localFrameReceivedCallback(const sensor_msgs::ImageConstPtr& msg)
{
    if (m_enabled)
    {
        cv_bridge::CvImagePtr cvPtr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
        cv::Mat frame = cutoutFace(cvPtr->image);
        sensor_msgs::ImageConstPtr processedMsg =
            cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::RGB8, frame).toImageMsg();
        m_itPublisher.publish(processedMsg);
    }
    else
    {
        m_itPublisher.publish(msg);
    }
}

void FaceCropper::peerFrameReceivedCallback(const opentera_webrtc_ros_msgs::PeerImageConstPtr& msg)
{
    if (m_enabled)
    {
        cv_bridge::CvImagePtr cvPtr = cv_bridge::toCvCopy(msg->frame, sensor_msgs::image_encodings::BGR8);
        cv::Mat frame = cutoutFace(cvPtr->image);
        sensor_msgs::ImageConstPtr processedMsg =
            cv_bridge::CvImage(msg->frame.header, sensor_msgs::image_encodings::RGB8, frame).toImageMsg();
        opentera_webrtc_ros_msgs::PeerImage newPeerImage;
        newPeerImage.frame = *processedMsg;
        newPeerImage.sender = msg->sender;
        m_peerFramePublisher.publish(newPeerImage);
    }
    else
    {
        m_peerFramePublisher.publish(msg);
    }
}

cv::Mat FaceCropper::cutoutFace(cv::Mat frame)
{
    std::vector<cv::Rect> faces;
    if (m_frameCounter % m_parameters.detectionFrames() == 0)
    {
        faces = detectFaces(frame);
    }

    if (!faces.empty() && faces.size() == 1)
    {
        cv::Rect cutout = m_oldCutout;
        cv::Rect face = faces[0];
        cutout = validateCutoutMinimumChange(getCutoutDimensions(face));

        m_cutoutList.push_back(cutout);
        if (m_cutoutList.size() > m_parameters.framesUsedForStabilizer())
        {
            m_cutoutList.pop_front();
            cutout = getAverageRect(m_cutoutList);
        }

        cutout = correctCutoutBoundaries(validateAspectRatio(cutout, face), frame);

        frame = frame(cutout);
        m_oldCutout = cutout;
        m_noDetectionCounter = 0;
    }
    else if (faces.size() == 0 && imageIsModifiable())
    {
        m_noDetectionCounter++;
        frame = frame(m_oldCutout);
    }
    else if (faces.size() > 1)
    {
        m_noDetectionCounter = 0;
    }

    if (faces.size() <= 1 && !frame.empty() && imageIsModifiable())
    {
        cv::resize(frame, frame, cv::Size(m_parameters.width(), m_parameters.height()));
    }
    m_frameCounter++;
    return frame;
}

std::vector<cv::Rect> FaceCropper::detectFaces(cv::Mat& frame)
{
    m_resizedFrame = frame;
    int newWidth = m_parameters.detectionScale() * frame.size().width;
    int newHeight = m_parameters.detectionScale() * frame.size().height;

    cv::resize(frame, m_resizedFrame, cv::Size(newWidth, newHeight));

    std::vector<cv::Rect> detectedFaces;
    std::vector<cv::Rect> validFaces;

    cv::Mat gray;

    cv::cvtColor(m_resizedFrame, gray, cv::COLOR_RGBA2GRAY, 0);

    faceCascade.detectMultiScale(
        gray,
        detectedFaces,
        1.1,
        3,
        0 | cv::CASCADE_SCALE_IMAGE,
        cv::Size(m_parameters.minFaceWidth(), m_parameters.minFaceHeight()));

    for (cv::Rect& r : detectedFaces)
    {
        r.x /= m_parameters.detectionScale();
        r.y /= m_parameters.detectionScale();
        r.width /= m_parameters.detectionScale();
        r.height /= m_parameters.detectionScale();
    }

    validFaces = getValidFaces(detectedFaces, frame);

    return validFaces;
}

cv::Rect FaceCropper::getCutoutDimensions(cv::Rect face)
{
    cv::Rect cutout;
    if (face.height / static_cast<float>(face.width) >= m_aspectRatio)
    {
        int topMargin = m_parameters.topMargin() * face.height;
        int bottomMargin = m_parameters.bottomMargin() * face.height;
        cutout.height = getClosestNumberDividableBy(face.height + topMargin + bottomMargin, m_yAspect / m_xAspect);
        cutout.y = 0;
        if (face.y >= topMargin)
        {
            cutout.y = face.y - topMargin;
        }

        cutout.width = cutout.height / m_aspectRatio;
        float xMargin = static_cast<float>(cutout.width - face.width) / 2;
        cutout.x = 0;
        if (face.x >= xMargin)
        {
            cutout.x = face.x - xMargin;
        }
    }
    else
    {
        int leftMargin = m_parameters.leftMargin() * face.width;
        int rightMargin = m_parameters.rightMargin() * face.width;

        cutout.width = getClosestNumberDividableBy((face.width + leftMargin + rightMargin), m_xAspect / m_yAspect);
        cutout.x = 0;
        if (face.x >= leftMargin)
        {
            cutout.x = face.x - leftMargin;
        }

        cutout.height = cutout.width * m_aspectRatio;
        int yMargin = static_cast<float>(cutout.height - face.height) / 2;
        cutout.y = 0;
        if (face.y >= yMargin)
        {
            cutout.y = face.y - yMargin;
        }
    }
    return cutout;
}

cv::Rect FaceCropper::validateCutoutMinimumChange(cv::Rect cutout)
{
    int minWidthChange = m_oldCutout.width * m_parameters.minWidthChange();
    int minHeightChange = m_oldCutout.height * m_parameters.minHeightChange();
    int minXChange = m_oldCutout.width * m_parameters.minXChange();
    int minYChange = m_oldCutout.height * m_parameters.minYChange();

    if (cutout.height < m_oldCutout.height + minHeightChange && cutout.height > m_oldCutout.height - minHeightChange &&
        cutout.width < m_oldCutout.width + minWidthChange && cutout.width > m_oldCutout.width - minWidthChange)
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
    return cutout;
}

cv::Rect FaceCropper::validateAspectRatio(cv::Rect cutout, cv::Rect face)
{
    if (face.height / static_cast<float>(face.width) >= m_aspectRatio)
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
    return cutout;
}

cv::Rect FaceCropper::correctCutoutBoundaries(cv::Rect cutout, cv::Mat frame)
{
    if (cutout.x > frame.size().width)
    {
        cutout.x = frame.size().width - cutout.width;
    }
    else if (cutout.x < 0)
    {
        cutout.x = 0;
    }

    if (cutout.width <= 0)
    {
        cutout.width = 1;
    }
    else if (cutout.width + cutout.x > frame.size().width)
    {
        cutout.x = frame.size().width - cutout.width;
    }

    if (cutout.y > frame.size().height)
    {
        cutout.y = frame.size().height - cutout.height;
    }
    else if (cutout.y < 0)
    {
        cutout.y = 0;
    }

    if (cutout.height <= 0)
    {
        cutout.height = 1;
    }
    else if (cutout.height + cutout.y > frame.size().height)
    {
        cutout.y = frame.size().height - cutout.height;
    }
    return cutout;
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
        if (static_cast<int>(trunc(newWidth)) > 0)
        {
            r.width = newWidth;
        }
        else
        {
            r.width = static_cast<int>(round(newWidth));
        }

        float newHeight = r.height / i;
        if (static_cast<int>(trunc(newHeight)) > 0)
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

    float maxOverlapPercentage = 0.2;

    for (int i = 0; i < m_lastDetectedFaces.size(); i++)
    {
        struct FaceVector faceVector = m_lastDetectedFaces[i];  // changer nom facevector

        std::vector<FaceVector> otherFaceVectors = m_lastDetectedFaces;
        otherFaceVectors.erase(otherFaceVectors.begin() + i);
        bool overlaps = false;
        cv::Rect face1 = faceVector.face.back().detection;

        // Validates that faces don't overlap with each other, if they do, the oldest and more constant one is valid
        for (auto& otherFaceVector : otherFaceVectors)
        {
            cv::Rect face2 = otherFaceVector.face.back().detection;
            if ((face1 & face2).area() > face1.area() * maxOverlapPercentage &&
                faceVector.face.size() <= otherFaceVector.face.size() && faceVector.origin > otherFaceVector.origin)
            {
                overlaps = true;
            }
        }

        // Ensures that a face as been detected for long enough to be valid
        if (faceVector.face.size() > m_parameters.faceStoringFrames() * m_parameters.validFaceMinTime() && !overlaps)
        {
            if (m_parameters.highlightDetections())
            {
                cv::rectangle(frame, face1, cv::Scalar(0, 255, 0), 2);
            }
            validFaces.emplace_back(face1);
        }
    }
    return validFaces;
}

void FaceCropper::updateLastFacesDetected(std::vector<cv::Rect> detectedFaces, cv::Mat frame)
{
    // Deletes face detections that have been in the detectionVector for more than the faceStoringFrames parameter
    for (auto& faceVector : m_lastDetectedFaces)
    {
        faceVector.face.erase(
            std::remove_if(
                std::begin(faceVector.face),
                std::end(faceVector.face),
                [this](auto& item) { return item.frame < m_frameCounter - m_parameters.faceStoringFrames(); }),
            std::end(faceVector.face));
    }

    // Deletes faces that no longer have detections
    m_lastDetectedFaces.erase(
        std::remove_if(
            std::begin(m_lastDetectedFaces),
            std::end(m_lastDetectedFaces),
            [this](auto& item) { return (item).face.size() == 0; }),
        std::end(m_lastDetectedFaces));

    // Checks each detections to see if they match a stored face
    for (cv::Rect& detectedFace : detectedFaces)
    {
        bool isNewFace = true;
        if (!m_lastDetectedFaces.empty())
        {
            for (auto& faceVector : m_lastDetectedFaces)
            {
                cv::Rect oldFace = faceVector.face.back().detection;

                float stepX = detectedFace.x * m_parameters.maxPositionStep();
                float stepY = detectedFace.y * m_parameters.maxPositionStep();
                float stepW = detectedFace.width * m_parameters.maxSizeStep();
                float stepH = detectedFace.height * m_parameters.maxSizeStep();

                if (detectedFace.x < oldFace.x + stepX && detectedFace.x > oldFace.x - stepX &&
                    detectedFace.y < oldFace.y + stepY && detectedFace.y > oldFace.y - stepY &&
                    detectedFace.width < oldFace.width + stepW && detectedFace.width > oldFace.width - stepW &&
                    detectedFace.height < oldFace.height + stepH && detectedFace.height > oldFace.height - stepH)
                {
                    // The detection matches a face, so it's added in the face's vector
                    isNewFace = false;

                    struct DetectionFrame newDetection = {m_frameCounter, detectedFace};
                    faceVector.face.emplace_back(newDetection);
                    if (faceVector.face.size() > m_parameters.faceStoringFrames() * m_parameters.validFaceMinTime() &&
                        m_parameters.highlightDetections())
                    {
                        cv::rectangle(frame, detectedFace, cv::Scalar(0, 255, 0), 2);
                    }
                    else if (m_parameters.highlightDetections())
                    {
                        cv::rectangle(frame, detectedFace, cv::Scalar(255, 0, 0), 2);
                    }
                    break;
                }
            }
        }
        if (isNewFace)
        {
            // The detection doensn't match any faces, so a face is created
            if (m_parameters.highlightDetections())
            {
                cv::rectangle(frame, detectedFace, cv::Scalar(255, 0, 0), 2);
            }

            std::vector<DetectionFrame> newVect;
            struct DetectionFrame newDetection = {m_frameCounter, detectedFace};
            newVect.emplace_back(newDetection);
            struct FaceVector newFace = {m_frameCounter, newVect};
            m_lastDetectedFaces.emplace_back(newFace);
        }
    }
}

bool FaceCropper::imageIsModifiable()
{
    return m_noDetectionCounter < m_parameters.refreshRate() * m_parameters.secondsWithoutDetection() &&
           m_oldCutout != cv::Rect(0, 0, 0, 0);
}
