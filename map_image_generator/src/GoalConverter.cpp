#include "map_image_generator/GoalConverter.h"

using namespace map_image_generator;

GoalConverter::GoalConverter(const Parameters &parameters, ros::NodeHandle &nodeHandle, geometry_msgs::PoseStamped::Ptr activeGoal) : 
    m_parameters(parameters),
    m_nodeHandle(nodeHandle),
    m_activeGoal(activeGoal)
{
    image_goal_to_map_goal_service = m_nodeHandle.advertiseService("image_goal_to_map_goal",
                                                                   &GoalConverter::mapImageGoalCallback,
                                                                   this);
}

GoalConverter::~GoalConverter()
{
}

bool GoalConverter::mapImageGoalCallback(ImageGoalToMapGoal::Request &req,
                                         ImageGoalToMapGoal::Response &res)
{
    res.map_goal = convertMapImageToMap(m_parameters, req.image_goal);
    *m_activeGoal = res.map_goal;
    return true;
}
