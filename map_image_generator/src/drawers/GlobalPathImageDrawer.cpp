#include "map_image_generator/drawers/GlobalPathImageDrawer.h"

#include <tf/tf.h>

using namespace map_image_generator;

GlobalPathImageDrawer::GlobalPathImageDrawer(const Parameters& parameters,
    ros::NodeHandle& nodeHandle,
    tf::TransformListener& tfListener) :
    ImageDrawer(parameters, nodeHandle, tfListener)
{
    m_globalPathSubscriber = m_nodeHandle.subscribe("global_path", 1,
        &GlobalPathImageDrawer::globalPathCallback, this);
}

GlobalPathImageDrawer::~GlobalPathImageDrawer()
{
}

void GlobalPathImageDrawer::draw(cv::Mat& image)
{
    if (!m_lastGlobalPath) { return; }

    tf::StampedTransform transform;
    try
    {
        m_tfListener.lookupTransform(m_parameters.mapFrameId(), m_lastGlobalPath->header.frame_id,
            ros::Time(0), transform);
        drawGlobalPath(image, transform);
    }
    catch (tf::TransformException ex)
    {
        ROS_ERROR("%s",ex.what());
    }
}

void GlobalPathImageDrawer::globalPathCallback(const nav_msgs::Path::ConstPtr& globalPath)
{
    m_lastGlobalPath = globalPath;
}

void GlobalPathImageDrawer::drawGlobalPath(cv::Mat& image, tf::StampedTransform& transform)
{
    const cv::Scalar& color = m_parameters.globalPathColor();
    int thickness = m_parameters.globalPathThickness();

    for(int i = 0; i + 1 < m_lastGlobalPath->poses.size(); i++)
    {
        tf::Pose startPose;
        tf::poseMsgToTF(m_lastGlobalPath->poses[i].pose, startPose);
        tf::Pose endPose;
        tf::poseMsgToTF(m_lastGlobalPath->poses[i + 1].pose, endPose);

        startPose = transform * startPose;
        endPose = transform * endPose;

        int startX, startY, endX, endY;
        convertTransformToMapCoordinates(startPose, startX, startY);
        convertTransformToMapCoordinates(endPose, endX, endY);

        cv::line(image,
            cv::Point(startX, startY),
            cv::Point(endX, endY),
            color,
            thickness,
            cv::LINE_AA);
    }
}
