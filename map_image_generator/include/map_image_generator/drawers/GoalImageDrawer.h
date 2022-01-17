#ifndef GOAL_IMAGE_DRAWER_H
#define GOAL_IMAGE_DRAWER_H

#include "map_image_generator/drawers/ImageDrawer.h"

#include <deque>
#include <geometry_msgs/PoseStamped.h>
#include <ros/ros.h>
#include <std_srvs/SetBool.h>

namespace map_image_generator
{
    class GoalImageDrawer : public ImageDrawer
    {
        std::deque<geometry_msgs::PoseStamped> m_activeGoals;
        ros::Subscriber m_add_goal_sub;
        ros::Subscriber m_remove_goal_sub;
        ros::ServiceServer m_clearGoalsService;

    public:
        GoalImageDrawer(const Parameters& parameters, ros::NodeHandle& nodeHandle,
                        tf::TransformListener& tfListener);
        ~GoalImageDrawer() override;

        void addGoalCallback(const geometry_msgs::PoseStamped::ConstPtr& goal);
        void removeGoalCallback(const geometry_msgs::PoseStamped::ConstPtr& goal);

        void draw(cv::Mat& image, double& scaleFactor) override;

    private:
        void drawGoal(const geometry_msgs::PoseStamped& goal, cv::Mat& image,
                      tf::Transform& transform, double& scaleFactor);
        bool clearGoals(std_srvs::SetBool::Request& req,
                        std_srvs::SetBool::Response& res);
    };
}
#endif
