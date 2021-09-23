#include "map_image_generator/GoalConverter.h"
#include "map_image_generator/utils.h"

using namespace map_image_generator;

GoalConverter::GoalConverter(const Parameters& parameters,
    ros::NodeHandle& nodeHandle) :
    m_parameters(parameters),
    m_nodeHandle(nodeHandle)
{
    m_mapImageGoalSubscriber = m_nodeHandle.subscribe("map_image_goal", 1000,
        &GoalConverter::mapImageGoalCallback, this);
    m_goalPublisher = m_nodeHandle.advertise<geometry_msgs::PoseStamped>("output_goal", 50);

    m_moveBaseSimpleImageGoalSub = m_nodeHandle.subscribe(
        "electron/goto",
        1000,
        &GoalConverter::moveBaseSimpleImageGoalCB,
        this);
    m_moveBaseSimpleGoalPub = m_nodeHandle.advertise<geometry_msgs::PoseStamped>("move_base_simple/goal", 50);
}

GoalConverter::~GoalConverter()
{
}

void GoalConverter::mapImageGoalCallback(const geometry_msgs::PoseStamped::ConstPtr& mapImageGoal)
{
    m_goalPublisher.publish(convertMapImageToMap(m_parameters, *mapImageGoal));
}

void GoalConverter::moveBaseSimpleImageGoalCB(const std_msgs::String::ConstPtr& moveBaseSimpleImageGoal)
{
    geometry_msgs::PoseStamped mapImageGoal;

    mapImageGoal.header.frame_id = m_parameters.mapFrameId();
    mapImageGoal.header.stamp = ros::Time::now();

    double pose[3];
    try {
        std::string input = moveBaseSimpleImageGoal->data;

        double found;
        int i = 0;
        std::string temp;

        size_t pos = 0;
        bool first_garbage = true;
        while ((pos = input.find(":")) != std::string::npos) {
            temp = input.substr(0, pos);
            ROS_INFO("parsing %s", temp.c_str());

            if (!first_garbage) {
                std::stringstream(temp) >> found;
                pose[i] = found;
                ROS_INFO("pose[%d] is %f", i, found);
                i++;
            }
            if (first_garbage) {
                first_garbage = false;
            }

            input.erase(0, pos + std::string(":").length());
        }
        ROS_INFO("parsing %s", input.c_str());
        std::stringstream(input) >> found;
        pose[i] = found;
        ROS_INFO("pose[%d] is %f", i, found);
        i++;

        if (i != 3) {
            throw std::invalid_argument("Missing value");
        }
    }
    catch(...) {
        ROS_ERROR("Invalid move_base_simple/image_goal received");
        return;
    }

    mapImageGoal.pose.position.x = pose[0];
    mapImageGoal.pose.position.y = pose[1];
    mapImageGoal.pose.position.z = 0.0;

    mapImageGoal.pose.orientation = tf::createQuaternionMsgFromYaw(-3.1416*pose[2]/180.0);

    m_moveBaseSimpleGoalPub.publish(convertMapImageToMap(m_parameters, mapImageGoal));
}
