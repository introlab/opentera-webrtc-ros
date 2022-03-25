#include "map_image_generator/drawers/LabelImageDrawer.h"

#include <opencv2/imgproc.hpp>

using namespace map_image_generator;

LabelImageDrawer::LabelImageDrawer(
    const Parameters& parameters,
    ros::NodeHandle& nodeHandle,
    tf::TransformListener& tfListener)
    : ImageDrawer(parameters, nodeHandle, tfListener),
      m_labelArraySubscriber{m_nodeHandle.subscribe("stored_labels", 1, &LabelImageDrawer::labelArrayCallback, this)}
{
}

LabelImageDrawer::~LabelImageDrawer() = default;

void LabelImageDrawer::labelArrayCallback(const opentera_webrtc_ros_msgs::LabelArray::ConstPtr& labelArray)
{
    m_lastLabelArray = labelArray;
}


void LabelImageDrawer::draw(cv::Mat& image)
{
    if (!m_lastLabelArray)
    {
        return;
    }

    for (const auto& label : m_lastLabelArray->labels)
    {
        auto tf = getTransformInRef(label.pose.header.frame_id);
        if (tf)
        {
            drawLabel(label, image, *tf);
        }
    }
}

void LabelImageDrawer::drawLabel(const opentera_webrtc_ros_msgs::Label& label, cv::Mat& image, tf::Transform& transform)
{
    const cv::Scalar& color = m_parameters.labelColor();
    int size = m_parameters.labelSize();

    tf::Pose labelPose;
    tf::poseMsgToTF(label.pose.pose, labelPose);
    labelPose = transform * labelPose;
    adjustTransformForRobotRef(labelPose);
    double yaw = tf::getYaw(labelPose.getRotation());

    int startX, startY;
    convertTransformToMapCoordinates(labelPose, startX, startY);

    int endX = static_cast<int>(startX + size * cos(yaw));
    int endY = static_cast<int>(startY + size * sin(yaw));

    cv::drawMarker(
        image,
        cv::Point(startX, startY),
        color,
        cv::MARKER_DIAMOND,
        ceilDivision(size, 4.0),
        ceilDivision(size, 12.0),
        cv::FILLED);
    cv::putText(
        image,
        label.name,
        cv::Point(startX, startY),
        cv::FONT_HERSHEY_DUPLEX,
        0.5,
        m_parameters.textColor(),
        1);
}
