#include "map_image_generator/drawers/GoalImageDrawer.h"

#include <cmath>
#include <ros/subscriber.h>
#include <tf/tf.h>

using namespace map_image_generator;
using namespace std;

GoalImageDrawer::GoalImageDrawer(const Parameters& parameters,
                                 ros::NodeHandle& nodeHandle,
                                 tf::TransformListener& tfListener)
    : ImageDrawer(parameters, nodeHandle, tfListener),
      m_add_goal_sub{nodeHandle.subscribe("map_image_drawer/add_goal", 10,
                                          &GoalImageDrawer::addGoalCallback, this)},
      m_remove_goal_sub{nodeHandle.subscribe("map_image_drawer/remove_goal", 10,
                                             &GoalImageDrawer::removeGoalCallback, this)},
      m_clearGoalsService{nodeHandle.advertiseService("map_image_drawer/clear_goals",
                                                      &GoalImageDrawer::clearGoals, this)}
{
}

GoalImageDrawer::~GoalImageDrawer() = default;

void GoalImageDrawer::addGoalCallback(const geometry_msgs::PoseStamped::ConstPtr& goal)
{
    m_activeGoals.emplace_back(*goal);
}

void GoalImageDrawer::removeGoalCallback(const geometry_msgs::PoseStamped::ConstPtr& goal)
{
    if (!m_activeGoals.empty()
        && std::round(m_activeGoals.front().pose.position.z)
               == std::round(goal->pose.position.z))
    {
        m_activeGoals.pop_front();
    }
    else
    {
        ROS_WARN("%s", "Tried to remove a goal that was not the first one");
    }
}

bool GoalImageDrawer::clearGoals(std_srvs::SetBool::Request& req,
                                 std_srvs::SetBool::Response& res)
{
    if (req.data)
    {
        m_activeGoals.clear();
        res.success = true;
        return true;
    }
    return false;
}

void GoalImageDrawer::draw(cv::Mat& image)
{
    for (const auto& goal : m_activeGoals)
    {
        auto tf = getTransformInRef(goal.header.frame_id);
        if (tf)
        {
            drawGoal(goal, image, *tf);
        }
    }
}

void GoalImageDrawer::drawGoal(const geometry_msgs::PoseStamped& goal, cv::Mat& image,
                               tf::Transform& transform)
{
    const cv::Scalar& color = m_parameters.goalColor();
    int size = m_parameters.goalSize();
    size_t index = std::lround(goal.pose.position.z);

    tf::Pose goalPose;
    tf::poseMsgToTF(goal.pose, goalPose);
    goalPose = transform * goalPose;
    adjustTransformForRobotRef(goalPose);
    double yaw = tf::getYaw(goalPose.getRotation());

    int startX, startY;
    convertTransformToMapCoordinates(goalPose, startX, startY);

    int endX = static_cast<int>(startX + size * cos(yaw));
    int endY = static_cast<int>(startY + size * sin(yaw));

    cv::circle(image, cv::Point(startX, startY), static_cast<int>(ceil(size / 5.0)),
               color, cv::FILLED);
    cv::arrowedLine(image, cv::Point(startX, startY), cv::Point(endX, endY), color,
                    static_cast<int>(ceil(size / 10.0)), cv::LINE_8, 0, 0.3);
    cv::putText(image, std::to_string(index), cv::Point(startX, startY),
                cv::FONT_HERSHEY_DUPLEX, 0.5, m_parameters.textColor(), 1);
}
