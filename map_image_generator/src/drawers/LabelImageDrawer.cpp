#include "map_image_generator/drawers/LabelImageDrawer.h"

#include <opencv2/imgproc.hpp>

using namespace map_image_generator;

LabelImageDrawer::LabelImageDrawer(const Parameters& parameters,
                                   ros::NodeHandle& nodeHandle,
                                   tf::TransformListener& tfListener)
    : ImageDrawer(parameters, nodeHandle, tfListener)
{
}

LabelImageDrawer::~LabelImageDrawer() = default;

void LabelImageDrawer::draw(cv::Mat& image) {}

void LabelImageDrawer::drawLabel(const geometry_msgs::PoseStamped& label,
                                 const std::string& text, cv::Mat& image,
                                 tf::Transform& transform)
{
    const cv::Scalar& color = m_parameters.labelColor();
    int size = m_parameters.labelSize();

    tf::Pose labelPose;
    tf::poseMsgToTF(label.pose, labelPose);
    labelPose = transform * labelPose;
    adjustTransformForRobotRef(labelPose);
    double yaw = tf::getYaw(labelPose.getRotation());

    int startX, startY;
    convertTransformToMapCoordinates(labelPose, startX, startY);

    int endX = static_cast<int>(startX + size * cos(yaw));
    int endY = static_cast<int>(startY + size * sin(yaw));

    cv::circle(image, cv::Point(startX, startY), static_cast<int>(ceil(size / 5.0)),
               color, cv::FILLED);
    cv::putText(image, text, cv::Point(startX, startY), cv::FONT_HERSHEY_DUPLEX, 0.5,
                m_parameters.textColor(), 1);
}
