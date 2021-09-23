#include "map_image_generator/Parameters.h"
#include "map_image_generator/MapImageGenerator.h"
#include "map_image_generator/GoalConverter.h"
#include "map_image_generator/MapLabelsConverter.h"

#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <image_transport/image_transport.h>

using namespace map_image_generator;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "map_image_generator");

    ros::NodeHandle nodeHandle;
    ros::NodeHandle nodeHandleParam("~");

    Parameters parameters(nodeHandleParam);
    tf::TransformListener tfListener;
    MapImageGenerator mapImageGenerator(parameters, nodeHandle, tfListener);
    GoalConverter goalConverter(parameters, nodeHandle);
    MapLabelsConverter mapLabelsConverter(parameters, nodeHandle);

    image_transport::ImageTransport imageTransport(nodeHandle);
    image_transport::Publisher mapImagePublisher = imageTransport.advertise("map_image", 1);
    sensor_msgs::Image mapImage;

    ROS_INFO("MapImage initialized, starting image generation after first cycle...");

    ros::Rate loop_rate(parameters.refreshRate());
    if (ros::ok())
    {
        loop_rate.sleep();
        ros::spinOnce();
    }

    ROS_INFO("Skipped first cycle...");

    while(ros::ok())
    {
        mapImageGenerator.generate(mapImage);
        mapImagePublisher.publish(mapImage);

        loop_rate.sleep();
        ros::spinOnce();
    }

    return 0;
}
