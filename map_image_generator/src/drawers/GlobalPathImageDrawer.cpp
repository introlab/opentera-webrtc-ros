#include "map_image_generator/drawers/GlobalPathImageDrawer.h"

using namespace map_image_generator;

GlobalPathImageDrawer::GlobalPathImageDrawer(
    const Parameters& parameters,
    rclcpp::Node& node,
    tf2_ros::Buffer& tfBuffer)
    : ImageDrawer(parameters, node, tfBuffer),
      m_globalPathSubscriber{m_node.create_subscription<nav_msgs::msg::Path>(
          "global_path",
          1,
          bind_this<nav_msgs::msg::Path>(this, &GlobalPathImageDrawer::globalPathCallback))},
      m_clearGlobalPathService{m_node.create_service<std_srvs::srv::SetBool>(
          "clear_global_path",
          bind_this<std_srvs::srv::SetBool>(this, &GlobalPathImageDrawer::clearGlobalPath))}
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

void GlobalPathImageDrawer::globalPathCallback(const nav_msgs::msg::Path::ConstSharedPtr& globalPath)
{
    m_lastGlobalPath = globalPath ? std::make_unique<nav_msgs::msg::Path>(*globalPath) : std::move(m_lastGlobalPath);
}

void GlobalPathImageDrawer::drawGlobalPath(cv::Mat& image, tf2::Transform& transform)
{
    const cv::Scalar& color = m_parameters.globalPathColor();
    int thickness = m_parameters.globalPathThickness();

    for (std::size_t i = 0; i + 1 < m_lastGlobalPath->poses.size(); i++)
    {
        tf2::Transform startPose;
        tf2::fromMsg(m_lastGlobalPath->poses[i].pose, startPose);
        tf2::Transform endPose;
        tf2::fromMsg(m_lastGlobalPath->poses[i + 1].pose, endPose);

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

void GlobalPathImageDrawer::clearGlobalPath(
    const std_srvs::srv::SetBool::Request::ConstSharedPtr& req,
    const std_srvs::srv::SetBool::Response::SharedPtr& res)
{
    if (req->data)
    {
        if (!m_lastGlobalPath)
        {
            res->success = true;
            return;
        }
        m_lastGlobalPath->poses.clear();
        res->success = true;
        return;
    }
    res->success = false;
    return;
}
