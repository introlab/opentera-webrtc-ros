#ifndef GLOBAL_PATH_IMAGE_DRAWER_H
#define GLOBAL_PATH_IMAGE_DRAWER_H

#include "map_image_generator/drawers/ImageDrawer.h"

#include <ros/ros.h>
#include <nav_msgs/Path.h>

namespace map_image_generator
{
    class GlobalPathImageDrawer : public ImageDrawer
    {
        ros::Subscriber m_globalPathSubscriber;
        nav_msgs::Path::ConstPtr m_lastGlobalPath;

    public:
        GlobalPathImageDrawer(const Parameters& parameters, ros::NodeHandle& nodeHandle, tf::TransformListener& tfListener);
        virtual ~GlobalPathImageDrawer();
        
        virtual void draw(cv::Mat& image);

    private:
        void globalPathCallback(const nav_msgs::Path::ConstPtr& globalPath);

        void drawGlobalPath(cv::Mat& image, tf::StampedTransform& transform);
    };
}
#endif