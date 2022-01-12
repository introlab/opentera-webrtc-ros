#ifndef ROBOT_IMAGE_DRAWER_H
#define ROBOT_IMAGE_DRAWER_H

#include "map_image_generator/drawers/ImageDrawer.h"

#include <geometry_msgs/PoseStamped.h>
#include <ros/ros.h>

namespace map_image_generator
{
    class RobotImageDrawer : public ImageDrawer
    {
    public:
        RobotImageDrawer(const Parameters& parameters, ros::NodeHandle& nodeHandle,
                         tf::TransformListener& tfListener);
        virtual ~RobotImageDrawer();

        virtual void draw(cv::Mat& image);

    private:
        void drawRobot(cv::Mat& image, tf::Transform& robotTransform);
    };
}
#endif
