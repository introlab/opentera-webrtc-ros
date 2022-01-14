#include "map_image_generator/GoalConverter.h"

using namespace map_image_generator;

GoalConverter::GoalConverter(const Parameters& parameters, ros::NodeHandle& nodeHandle,
                             tf::TransformListener& tfListener)
    : m_parameters(parameters), m_nodeHandle(nodeHandle), m_tfListener(tfListener),
      image_goal_to_map_goal_service{m_nodeHandle.advertiseService(
          "image_goal_to_map_goal", &GoalConverter::mapImageGoalCallback, this)}
{
}

GoalConverter::~GoalConverter() = default;

bool GoalConverter::mapImageGoalCallback(ImageGoalToMapGoal::Request& req,
                                         ImageGoalToMapGoal::Response& res)
{
    if (m_parameters.centeredRobot())
    {
        res.map_goal = convertMapImageToRobot(m_parameters, req.image_goal);
        try
        {
            m_tfListener.transformPose(m_parameters.mapFrameId(), res.map_goal,
                                       res.map_goal);
        }
        catch (tf::TransformException& ex)
        {
            ROS_ERROR("%s", ex.what());
            return false;
        }
    }
    else
    {
        res.map_goal = convertMapImageToMap(m_parameters, req.image_goal);
    }
    return true;
}
