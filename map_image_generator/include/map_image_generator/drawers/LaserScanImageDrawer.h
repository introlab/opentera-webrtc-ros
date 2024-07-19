#ifndef LASER_SCAN_IMAGE_DRAWER_H
#define LASER_SCAN_IMAGE_DRAWER_H

#include "map_image_generator/drawers/ImageDrawer.h"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

namespace map_image_generator
{
    class LaserScanImageDrawer : public ImageDrawer
    {
        rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr m_laserScanSubscriber;
        sensor_msgs::msg::LaserScan::ConstSharedPtr m_lastLaserScan;

    public:
        LaserScanImageDrawer(const Parameters& parameters, rclcpp::Node& node, tf2_ros::Buffer& tfBuffer);
        ~LaserScanImageDrawer() override;

        void draw(cv::Mat& image) override;

    private:
        void laserScanCallback(const sensor_msgs::msg::LaserScan::ConstSharedPtr& laserScan);

        void drawLaserScan(cv::Mat& image, tf2::Transform& transform);
        void drawRange(cv::Mat& image, tf2::Transform& transform, float range, float angle);
    };
}
#endif
