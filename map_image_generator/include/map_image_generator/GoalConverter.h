#ifndef GOAL_TRANSLATOR_H
#define GOAL_TRANSLATOR_H

#include "map_image_generator/ImageGoalToMapGoal.h"
#include "map_image_generator/Parameters.h"
#include "map_image_generator/utils.h"

#include <geometry_msgs/PointStamped.h>
#include <ros/ros.h>
#include <sstream>
#include <std_msgs/String.h>
#include <stdexcept>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>

namespace map_image_generator
{
    class GoalConverter
    {
        const Parameters& m_parameters;
        ros::NodeHandle& m_nodeHandle;

        ros::ServiceServer image_goal_to_map_goal_service;

        geometry_msgs::PoseStamped::Ptr m_activeGoal;

    public:
        GoalConverter(const Parameters& parameters, ros::NodeHandle& nodeHandle,
                      geometry_msgs::PoseStamped::Ptr activeGoal);
        virtual ~GoalConverter();

    private:
        bool mapImageGoalCallback(ImageGoalToMapGoal::Request& req,
                                  ImageGoalToMapGoal::Response& res);
    };
}
#endif
