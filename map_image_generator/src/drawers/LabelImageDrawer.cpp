#include "map_image_generator/drawers/LabelImageDrawer.h"

#include <opencv2/imgproc.hpp>

#include <tf2/utils.h>

using namespace map_image_generator;

LabelImageDrawer::LabelImageDrawer(const Parameters& parameters, rclcpp::Node& node, tf2_ros::Buffer& tfBuffer)
    : ImageDrawer(parameters, node, tfBuffer),
      m_labelArraySubscriber{m_node.create_subscription<opentera_webrtc_ros_msgs::msg::LabelArray>(
          "stored_labels",
          1,
          bind_this<opentera_webrtc_ros_msgs::msg::LabelArray>(this, &LabelImageDrawer::labelArrayCallback))}
{
}

LabelImageDrawer::~LabelImageDrawer() = default;

void LabelImageDrawer::labelArrayCallback(const opentera_webrtc_ros_msgs::msg::LabelArray::ConstSharedPtr& labelArray)
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

void LabelImageDrawer::drawLabel(
    const opentera_webrtc_ros_msgs::msg::Label& label,
    cv::Mat& image,
    tf2::Transform& transform)
{
    const cv::Scalar& color = m_parameters.labelColor();
    int size = m_parameters.labelSize();

    tf2::Transform labelPose;
    tf2::fromMsg(label.pose.pose, labelPose);
    labelPose = transform * labelPose;
    adjustTransformForRobotRef(labelPose);
    double yaw = tf2::getYaw(labelPose.getRotation());

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
