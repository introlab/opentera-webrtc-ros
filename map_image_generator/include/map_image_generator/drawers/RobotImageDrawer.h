#ifndef ROBOT_IMAGE_DRAWER_H
#define ROBOT_IMAGE_DRAWER_H

#include "map_image_generator/drawers/ImageDrawer.h"

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>

namespace map_image_generator
{
    class RobotImageDrawer : public ImageDrawer
    {
    public:
        RobotImageDrawer(const Parameters& parameters, rclcpp::Node& nodeHandle, tf2_ros::Buffer& tfBuffer);
        ~RobotImageDrawer() override;

        void draw(cv::Mat& image) override;

    private:
        void drawRobot(cv::Mat& image, tf2::Transform& robotTransform);
    };
}
#endif
