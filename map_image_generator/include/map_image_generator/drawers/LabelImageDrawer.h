#ifndef LABEL_IMAGE_DRAWER_H
#define LABEL_IMAGE_DRAWER_H

#include "map_image_generator/drawers/ImageDrawer.h"

#include <opentera_webrtc_ros_msgs/Label.h>
#include <opentera_webrtc_ros_msgs/LabelArray.h>
#include <ros/ros.h>

namespace map_image_generator
{
    class LabelImageDrawer : public ImageDrawer
    {
        ros::Subscriber m_labelArraySubscriber;
        opentera_webrtc_ros_msgs::LabelArray::ConstPtr m_lastLabelArray;

    public:
        LabelImageDrawer(const Parameters& parameters, ros::NodeHandle& nodeHandle, tf::TransformListener& tfListener);
        ~LabelImageDrawer() override;

        void draw(cv::Mat& image) override;

    private:
        void drawLabel(const opentera_webrtc_ros_msgs::Label& label, cv::Mat& image, tf::Transform& transform);

        void labelArrayCallback(const opentera_webrtc_ros_msgs::LabelArray::ConstPtr& labelArray);
    };
}
#endif
