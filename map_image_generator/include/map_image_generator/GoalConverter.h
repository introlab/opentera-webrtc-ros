#ifndef GOAL_TRANSLATOR_H
#define GOAL_TRANSLATOR_H

#include "map_image_generator/Parameters.h"

#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <geometry_msgs/PointStamped.h>
#include <std_msgs/String.h>
#include <sstream>
#include <stdexcept>

#include "map_image_generator/utils.h"
#include "map_image_generator/ImageGoalToMapGoal.h"

namespace map_image_generator
{
    class GoalConverter
    {
        const Parameters& m_parameters;
        ros::NodeHandle& m_nodeHandle;

        ros::Publisher m_goalPublisher;
        ros::Publisher m_moveBaseSimpleGoalPub;

        ros::ServiceServer image_goal_to_map_goal_service;

    public:
        GoalConverter(const Parameters &parameters, ros::NodeHandle &nodeHandle);
        virtual ~GoalConverter();

    private:
        bool mapImageGoalCallback(ImageGoalToMapGoal::Request &req,
                                  ImageGoalToMapGoal::Response &res);
    };
}
#endif
