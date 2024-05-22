#include "map_image_generator/drawers/RobotImageDrawer.h"

#include <cmath>
#include <tf2/utils.h>

using namespace map_image_generator;
using namespace std;

RobotImageDrawer::RobotImageDrawer(const Parameters& parameters, rclcpp::Node& node, tf2_ros::Buffer& tfBuffer)
    : ImageDrawer(parameters, node, tfBuffer)
{
}

RobotImageDrawer::~RobotImageDrawer() = default;

void RobotImageDrawer::draw(cv::Mat& image)
{
    auto tf = getTransformInRef(m_parameters.robotFrameId());
    if (tf)
    {
        drawRobot(image, *tf);
    }
}

void RobotImageDrawer::drawRobot(cv::Mat& image, tf2::Transform& robotTransform)
{
    const cv::Scalar& color = m_parameters.robotColor();
    int size = m_parameters.robotSize();

    adjustTransformForRobotRef(robotTransform);
    double yaw = tf2::getYaw(robotTransform.getRotation());

    int startX, startY;
    convertTransformToMapCoordinates(robotTransform, startX, startY);

    int endX = static_cast<int>(startX + size * cos(yaw));
    int endY = static_cast<int>(startY + size * sin(yaw));

    cv::circle(image, cv::Point(startX, startY), ceilDivision(size, 5.0), color, cv::FILLED, cv::LINE_8);
    cv::arrowedLine(
        image,
        cv::Point(startX, startY),
        cv::Point(endX, endY),
        color,
        ceilDivision(size, 10.0),
        cv::LINE_8,
        0,
        0.3);
}
