#ifndef GOAL_TRANSLATOR_H
#define GOAL_TRANSLATOR_H

#include "map_image_generator/Parameters.h"
#include "map_image_generator/utils.h"
#include "opentera_webrtc_ros_msgs/srv/image_goal_to_map_goal.hpp"

#include <geometry_msgs/msg/point_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sstream>
#include <std_msgs/msg/string.hpp>
#include <stdexcept>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace map_image_generator
{
    class GoalConverter
    {
        const Parameters& m_parameters;
        rclcpp::Node& m_node;
        tf2_ros::Buffer& m_tfBuffer;

        rclcpp::Service<opentera_webrtc_ros_msgs::srv::ImageGoalToMapGoal>::SharedPtr image_goal_to_map_goal_service;

    public:
        GoalConverter(const Parameters& parameters, rclcpp::Node& node, tf2_ros::Buffer& tfBuffer);
        virtual ~GoalConverter();

    private:
        void mapImageGoalCallback(
            const opentera_webrtc_ros_msgs::srv::ImageGoalToMapGoal::Request::ConstSharedPtr& req,
            const opentera_webrtc_ros_msgs::srv::ImageGoalToMapGoal::Response::SharedPtr& res);
    };
}
#endif
