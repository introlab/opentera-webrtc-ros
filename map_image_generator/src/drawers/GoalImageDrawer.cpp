#include "map_image_generator/drawers/GoalImageDrawer.h"

#include <cmath>
#include <rclcpp/rclcpp.hpp>
#include <tf2/utils.h>

using namespace map_image_generator;
using namespace std;

GoalImageDrawer::GoalImageDrawer(const Parameters& parameters, rclcpp::Node& node, tf2_ros::Buffer& tfBuffer)
    : ImageDrawer(parameters, node, tfBuffer),
      m_add_goal_sub{m_node.create_subscription<geometry_msgs::msg::PoseStamped>(
          "map_image_drawer/add_goal",
          10,
          bind_this<geometry_msgs::msg::PoseStamped>(this, &GoalImageDrawer::addGoalCallback))},
      m_remove_goal_sub{m_node.create_subscription<geometry_msgs::msg::PoseStamped>(
          "map_image_drawer/remove_goal",
          10,
          bind_this<geometry_msgs::msg::PoseStamped>(this, &GoalImageDrawer::removeGoalCallback))},
      m_clearGoalsService{m_node.create_service<std_srvs::srv::SetBool>(
          "map_image_drawer/clear_goals",
          bind_this<std_srvs::srv::SetBool>(this, &GoalImageDrawer::clearGoals))}
{
}

GoalImageDrawer::~GoalImageDrawer() = default;

void GoalImageDrawer::addGoalCallback(const geometry_msgs::msg::PoseStamped::ConstSharedPtr& goal)
{
    m_activeGoals.emplace_back(*goal);
}

void GoalImageDrawer::removeGoalCallback(const geometry_msgs::msg::PoseStamped::ConstSharedPtr& goal)
{
    if (!m_activeGoals.empty() &&
        std::round(m_activeGoals.front().pose.position.z) == std::round(goal->pose.position.z))
    {
        m_activeGoals.pop_front();
    }
    else
    {
        RCLCPP_WARN(m_node.get_logger(), "%s", "Tried to remove a goal that was not the first one");
    }
}

void GoalImageDrawer::clearGoals(
    const std_srvs::srv::SetBool::Request::ConstSharedPtr& req,
    const std_srvs::srv::SetBool::Response::SharedPtr& res)
{
    if (req->data)
    {
        m_activeGoals.clear();
        res->success = true;
    }
}

void GoalImageDrawer::draw(cv::Mat& image)
{
    for (const auto& goal : m_activeGoals)
    {
        auto tf = getTransformInRef(goal.header.frame_id);
        if (tf)
        {
            drawGoal(goal, image, *tf);
        }
    }
}

void GoalImageDrawer::drawGoal(const geometry_msgs::msg::PoseStamped& goal, cv::Mat& image, tf2::Transform& transform)
{
    const cv::Scalar& color = m_parameters.goalColor();
    int size = m_parameters.goalSize();
    size_t index = std::lround(goal.pose.position.z);

    tf2::Transform goalPose;
    tf2::fromMsg(goal.pose, goalPose);
    goalPose = transform * goalPose;
    adjustTransformForRobotRef(goalPose);
    double yaw = tf2::getYaw(goalPose.getRotation());

    int startX, startY;
    convertTransformToMapCoordinates(goalPose, startX, startY);

    int endX = static_cast<int>(startX + size * cos(yaw));
    int endY = static_cast<int>(startY + size * sin(yaw));

    cv::circle(image, cv::Point(startX, startY), ceilDivision(size, 5.0), color, cv::FILLED);
    cv::arrowedLine(
        image,
        cv::Point(startX, startY),
        cv::Point(endX, endY),
        color,
        ceilDivision(size, 10.0),
        cv::LINE_8,
        0,
        0.3);
    cv::putText(
        image,
        std::to_string(index),
        cv::Point(startX, startY),
        cv::FONT_HERSHEY_DUPLEX,
        0.5,
        m_parameters.textColor(),
        1);
}
