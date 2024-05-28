#ifndef GOAL_IMAGE_DRAWER_H
#define GOAL_IMAGE_DRAWER_H

#include "map_image_generator/drawers/ImageDrawer.h"

#include <deque>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/set_bool.hpp>

namespace map_image_generator
{
    class GoalImageDrawer : public ImageDrawer
    {
        std::deque<geometry_msgs::msg::PoseStamped> m_activeGoals;
        rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr m_add_goal_sub;
        rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr m_remove_goal_sub;
        rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr m_clearGoalsService;

    public:
        GoalImageDrawer(const Parameters& parameters, rclcpp::Node& node, tf2_ros::Buffer& tfBuffer);
        ~GoalImageDrawer() override;

        void addGoalCallback(const geometry_msgs::msg::PoseStamped::ConstSharedPtr& goal);
        void removeGoalCallback(const geometry_msgs::msg::PoseStamped::ConstSharedPtr& goal);

        void draw(cv::Mat& image) override;

    private:
        void drawGoal(const geometry_msgs::msg::PoseStamped& goal, cv::Mat& image, tf2::Transform& transform);
        void clearGoals(
            const std_srvs::srv::SetBool::Request::ConstSharedPtr& req,
            const std_srvs::srv::SetBool::Response::SharedPtr& res);
    };
}
#endif
