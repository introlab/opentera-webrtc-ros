#include "face_following/Parameters.h"
#include "face_following/FaceFollower.h"

#include <ros/ros.h>

using namespace face_following;

int main(int argc, char** argv)
{
    ros::init(argc, argv, "face_following");

    ros::NodeHandle nodeHandle;
    ros::NodeHandle nodeHandleParam{"~"};

    Parameters parameters{nodeHandleParam};
    FaceFollower facefollower{parameters, nodeHandle};

    ros::Rate loop_rate{parameters.refreshRate()};
    if (ros::ok())
    {
        loop_rate.sleep();
        ros::spinOnce();
    }

    while (ros::ok())
    {
        loop_rate.sleep();
        ros::spinOnce();
    }

    return 0;
}
