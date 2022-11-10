#include "face_cropping/Parameters.h"
#include "face_cropping/FaceCropper.h"

#include <ros/ros.h>

using namespace face_cropping;

int main(int argc, char** argv)
{
    ros::init(argc, argv, "face_cropping");

    ros::NodeHandle nodeHandle;
    ros::NodeHandle nodeHandleParam{"~"};

    Parameters parameters{nodeHandleParam};
    FaceCropper faceCropper{parameters, nodeHandle};

    ros::Rate loop_rate{parameters.refreshRate()};

    while (ros::ok())
    {
        loop_rate.sleep();
        ros::spinOnce();
    }

    return 0;
}
