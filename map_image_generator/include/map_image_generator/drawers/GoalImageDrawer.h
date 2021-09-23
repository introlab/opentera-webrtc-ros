#ifndef GOAL_IMAGE_DRAWER_H
#define GOAL_IMAGE_DRAWER_H

#include "map_image_generator/drawers/ImageDrawer.h"

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>

namespace map_image_generator
{
    class GoalImageDrawer : public ImageDrawer
    {
        ros::Subscriber m_goalSubscriber;
        geometry_msgs::PoseStamped::ConstPtr m_lastGoal;

    public:
        GoalImageDrawer(const Parameters& parameters, ros::NodeHandle& nodeHandle, tf::TransformListener& tfListener);
        virtual ~GoalImageDrawer();
        
        virtual void draw(cv::Mat& image);

    private:
        void goalCallback(const geometry_msgs::PoseStamped::ConstPtr& goal);

        void drawGoal(cv::Mat& image, tf::StampedTransform& transform);
    };
}
#endif