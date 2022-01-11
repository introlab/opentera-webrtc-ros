#ifndef GLOBAL_PATH_IMAGE_DRAWER_H
#define GLOBAL_PATH_IMAGE_DRAWER_H

#include "map_image_generator/drawers/ImageDrawer.h"

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <std_srvs/SetBool.h>

namespace map_image_generator
{
    class GlobalPathImageDrawer : public ImageDrawer
    {
        ros::Subscriber m_globalPathSubscriber;
        nav_msgs::Path::Ptr m_lastGlobalPath;
        ros::ServiceServer m_clearGlobalPathService;

    public:
        GlobalPathImageDrawer(const Parameters& parameters, ros::NodeHandle& nodeHandle, tf::TransformListener& tfListener);
        virtual ~GlobalPathImageDrawer();

        virtual void draw(cv::Mat& image);

    private:
        void globalPathCallback(const nav_msgs::Path::Ptr& globalPath);
        void drawGlobalPath(cv::Mat& image, tf::Transform& transform);
        bool clearGlobalPath(std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &res);
    };
}
#endif
