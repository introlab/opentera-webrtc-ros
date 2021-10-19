#include "map_image_generator/drawers/RobotImageDrawer.h"

#include <tf/tf.h>

#include <cmath>

using namespace map_image_generator;
using namespace std;

RobotImageDrawer::RobotImageDrawer(const Parameters& parameters,
    ros::NodeHandle& nodeHandle,
    tf::TransformListener& tfListener) :
    ImageDrawer(parameters, nodeHandle, tfListener)
{
}

RobotImageDrawer::~RobotImageDrawer()
{
}

void RobotImageDrawer::draw(cv::Mat& image)
{
    tf::StampedTransform robotTransform;
    try
    {
        m_tfListener.lookupTransform(m_parameters.mapFrameId(), m_parameters.robotFrameId(),
            ros::Time(0), robotTransform);
        drawRobot(image, robotTransform);
    }
    catch (tf::TransformException ex)
    {
        ROS_ERROR("%s",ex.what());
    }
}

void RobotImageDrawer::drawRobot(cv::Mat& image, tf::StampedTransform& robotTransform)
{
    const cv::Scalar& color = m_parameters.robotColor();
    int size = m_parameters.robotSize();

    double yaw = tf::getYaw(robotTransform.getRotation());

    int startX, startY;
    convertTransformToMapCoordinates(robotTransform, startX, startY);

    int endX = static_cast<int>(startX + size * cos(yaw));
    int endY = static_cast<int>(startY + size * sin(yaw));

    cv::circle(image, 
        cv::Point(startX, startY),
        static_cast<int>(ceil(size / 5.0)), 
        color,
        cv::FILLED,
        cv::LINE_8);
    cv::arrowedLine(image,
        cv::Point(startX, startY),
        cv::Point(endX, endY),
        color,
        static_cast<int>(ceil(size / 10.0)), 
        cv::LINE_8,
        0, 
        0.3);
}