#include "map_image_generator/GoalConverter.h"

using namespace map_image_generator;

GoalConverter::GoalConverter(const Parameters& parameters, rclcpp::Node& node, tf2_ros::Buffer& tfBuffer)
    : m_parameters(parameters),
      m_node(node),
      m_tfBuffer(tfBuffer),
      image_goal_to_map_goal_service{m_node.create_service<opentera_webrtc_ros_msgs::srv::ImageGoalToMapGoal>(
          "image_goal_to_map_goal",
          bind_this<opentera_webrtc_ros_msgs::srv::ImageGoalToMapGoal>(this, &GoalConverter::mapImageGoalCallback))}
{
}

GoalConverter::~GoalConverter() = default;

void GoalConverter::mapImageGoalCallback(
    const opentera_webrtc_ros_msgs::srv::ImageGoalToMapGoal::Request::ConstSharedPtr& req,
    const opentera_webrtc_ros_msgs::srv::ImageGoalToMapGoal::Response::SharedPtr& res)
{
    if (m_parameters.centeredRobot())
    {
        res->map_goal = convertMapImageToRobot(m_parameters, req->image_goal);
        try
        {
            m_tfBuffer.transform(res->map_goal, res->map_goal, m_parameters.mapFrameId());
        }
        catch (const tf2::TransformException& ex)
        {
            RCLCPP_ERROR(m_node.get_logger(), "%s", ex.what());
            res->success = false;
            return;
        }
    }
    else
    {
        res->map_goal = convertMapImageToMap(m_parameters, req->image_goal);
    }
    res->success = true;
    return;
}
