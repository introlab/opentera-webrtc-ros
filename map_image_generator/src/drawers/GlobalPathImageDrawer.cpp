#include "map_image_generator/drawers/GlobalPathImageDrawer.h"

#include <tf/tf.h>

using namespace map_image_generator;

GlobalPathImageDrawer::GlobalPathImageDrawer(
    const Parameters& parameters,
    ros::NodeHandle& nodeHandle,
    tf::TransformListener& tfListener)
    : ImageDrawer(parameters, nodeHandle, tfListener),
      m_globalPathSubscriber{nodeHandle.subscribe("global_path", 1, &GlobalPathImageDrawer::globalPathCallback, this)},
      m_clearGlobalPathService{
          m_nodeHandle.advertiseService("clear_global_path", &GlobalPathImageDrawer::clearGlobalPath, this)}
{
}

GlobalPathImageDrawer::~GlobalPathImageDrawer() = default;

void GlobalPathImageDrawer::draw(cv::Mat& image)
{
    if (!m_lastGlobalPath)
    {
        return;
    }

    auto tf = getTransformInRef(m_lastGlobalPath->header.frame_id);
    if (tf)
    {
        drawGlobalPath(image, *tf);
    }
}

void GlobalPathImageDrawer::globalPathCallback(const nav_msgs::Path::Ptr& globalPath)
{
    m_lastGlobalPath = globalPath;
}

void GlobalPathImageDrawer::drawGlobalPath(cv::Mat& image, tf::Transform& transform)
{
    const cv::Scalar& color = m_parameters.globalPathColor();
    int thickness = m_parameters.globalPathThickness();

    for (int i = 0; i + 1 < m_lastGlobalPath->poses.size(); i++)
    {
        tf::Pose startPose;
        tf::poseMsgToTF(m_lastGlobalPath->poses[i].pose, startPose);
        tf::Pose endPose;
        tf::poseMsgToTF(m_lastGlobalPath->poses[i + 1].pose, endPose);

        startPose = transform * startPose;
        endPose = transform * endPose;

        adjustTransformForRobotRef(startPose);
        adjustTransformForRobotRef(endPose);

        int startX, startY, endX, endY;
        convertTransformToMapCoordinates(startPose, startX, startY);
        convertTransformToMapCoordinates(endPose, endX, endY);

        cv::line(image, cv::Point(startX, startY), cv::Point(endX, endY), color, thickness, cv::LINE_AA);
    }
}

bool GlobalPathImageDrawer::clearGlobalPath(std_srvs::SetBool::Request& req, std_srvs::SetBool::Response& res)
{
    if (req.data)
    {
        if (!m_lastGlobalPath)
            return true;
        m_lastGlobalPath->poses.clear();
        res.success = true;
        return true;
    }
    return false;
}
