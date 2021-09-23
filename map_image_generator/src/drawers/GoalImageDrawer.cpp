#include "map_image_generator/drawers/GoalImageDrawer.h"

#include <tf/tf.h>

#include <cmath>

using namespace map_image_generator;
using namespace std;

GoalImageDrawer::GoalImageDrawer(const Parameters& parameters,
    ros::NodeHandle& nodeHandle,
    tf::TransformListener& tfListener) :
    ImageDrawer(parameters, nodeHandle, tfListener)
{
    m_goalSubscriber = m_nodeHandle.subscribe("input_goal", 1,
        &GoalImageDrawer::goalCallback, this);
}

GoalImageDrawer::~GoalImageDrawer()
{
}

void GoalImageDrawer::draw(cv::Mat& image)
{
    if (!m_lastGoal) { return; }

    tf::StampedTransform transform;
    try
    {
        m_tfListener.lookupTransform(m_parameters.mapFrameId(), m_lastGoal->header.frame_id,
            ros::Time(0), transform);
        drawGoal(image, transform);
    }
    catch (tf::TransformException ex)
    {
        ROS_ERROR("%s",ex.what());
    }
}

void GoalImageDrawer::goalCallback(const geometry_msgs::PoseStamped::ConstPtr& goal)
{
    m_lastGoal = goal;
}

void GoalImageDrawer::drawGoal(cv::Mat& image, tf::StampedTransform& transform)
{
    const cv::Scalar& color = m_parameters.goalColor();
    int size = m_parameters.goalSize();

    tf::Pose goalPose;
    tf::poseMsgToTF(m_lastGoal->pose, goalPose);
    goalPose = transform * goalPose;
    double yaw = tf::getYaw(goalPose.getRotation());

    int startX, startY;
    convertTransformToMapCoordinates(goalPose, startX, startY);

    int endX = static_cast<int>(startX + size * cos(yaw));
    int endY = static_cast<int>(startY + size * sin(yaw));

    cv::circle(image, 
        cv::Point(startX, startY),
        static_cast<int>(ceil(size / 5.0)), 
        color,
        cv::FILLED);
    cv::arrowedLine(image,
        cv::Point(startX, startY),
        cv::Point(endX, endY),
        color,
        static_cast<int>(ceil(size / 10.0)), 
        cv::LINE_8,
        0, 
        0.3);
}