#include "map_image_generator/GoalConverter.h"

using namespace map_image_generator;

GoalConverter::GoalConverter(const Parameters &parameters, ros::NodeHandle &nodeHandle) : 
    m_parameters(parameters),
    m_nodeHandle(nodeHandle)
{
    image_goal_to_map_goal_service = m_nodeHandle.advertiseService("image_goal_to_map_goal",
                                                                   &GoalConverter::mapImageGoalCallback,
                                                                   this);
    m_goalPublisher = m_nodeHandle.advertise<geometry_msgs::PoseStamped>("output_goal", 50);
}

GoalConverter::~GoalConverter()
{
}

bool GoalConverter::mapImageGoalCallback(ImageGoalToMapGoal::Request &req,
                                         ImageGoalToMapGoal::Response &res)
{
    res.map_goal = convertMapImageToMap(m_parameters, req.image_goal);
    // TODO: right now, the drawer draws the goal it receives on the output_goal topic, which is published by the same
    // node. This should be internal instead of using a topic.
    m_goalPublisher.publish(res.map_goal);
    return true;
}
