#ifndef GOAL_IMAGE_DRAWER_H
#define GOAL_IMAGE_DRAWER_H

#include "map_image_generator/drawers/ImageDrawer.h"

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>

namespace map_image_generator
{
    class GoalImageDrawer : public ImageDrawer
    {
        geometry_msgs::PoseStamped::Ptr m_activeGoal;

    public:
        GoalImageDrawer(const Parameters& parameters, ros::NodeHandle& nodeHandle, tf::TransformListener& tfListener,
                        geometry_msgs::PoseStamped::Ptr activeGoal);
        virtual ~GoalImageDrawer();
        
        virtual void draw(cv::Mat& image);

    private:
        void drawGoal(cv::Mat& image, tf::StampedTransform& transform);
    };
}
#endif