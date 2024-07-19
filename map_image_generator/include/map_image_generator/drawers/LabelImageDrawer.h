#ifndef LABEL_IMAGE_DRAWER_H
#define LABEL_IMAGE_DRAWER_H

#include "map_image_generator/drawers/ImageDrawer.h"

#include <opentera_webrtc_ros_msgs/msg/label.hpp>
#include <opentera_webrtc_ros_msgs/msg/label_array.hpp>
#include <rclcpp/rclcpp.hpp>

namespace map_image_generator
{
    class LabelImageDrawer : public ImageDrawer
    {
        rclcpp::Subscription<opentera_webrtc_ros_msgs::msg::LabelArray>::SharedPtr m_labelArraySubscriber;
        opentera_webrtc_ros_msgs::msg::LabelArray::ConstSharedPtr m_lastLabelArray;

    public:
        LabelImageDrawer(const Parameters& parameters, rclcpp::Node& node, tf2_ros::Buffer& tfBuffer);
        ~LabelImageDrawer() override;

        void draw(cv::Mat& image) override;

    private:
        void drawLabel(const opentera_webrtc_ros_msgs::msg::Label& label, cv::Mat& image, tf2::Transform& transform);

        void labelArrayCallback(const opentera_webrtc_ros_msgs::msg::LabelArray::ConstSharedPtr& labelArray);
    };
}
#endif
