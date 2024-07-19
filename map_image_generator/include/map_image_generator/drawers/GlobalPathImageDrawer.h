#ifndef GLOBAL_PATH_IMAGE_DRAWER_H
#define GLOBAL_PATH_IMAGE_DRAWER_H

#include "map_image_generator/drawers/ImageDrawer.h"

#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/set_bool.hpp>

namespace map_image_generator
{
    class GlobalPathImageDrawer : public ImageDrawer
    {
        rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr m_globalPathSubscriber;
        nav_msgs::msg::Path::UniquePtr m_lastGlobalPath;
        rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr m_clearGlobalPathService;

    public:
        GlobalPathImageDrawer(const Parameters& parameters, rclcpp::Node& node, tf2_ros::Buffer& tfBuffer);
        ~GlobalPathImageDrawer() override;

        void draw(cv::Mat& image) override;

    private:
        void globalPathCallback(const nav_msgs::msg::Path::ConstSharedPtr& globalPath);
        void drawGlobalPath(cv::Mat& image, tf2::Transform& transform);
        void clearGlobalPath(
            const std_srvs::srv::SetBool::Request::ConstSharedPtr& req,
            const std_srvs::srv::SetBool::Response::SharedPtr& res);
    };
}
#endif
