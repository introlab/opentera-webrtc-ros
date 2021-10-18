#ifndef UTILS_H
#define UTILS_H

#include "map_image_generator/Parameters.h"

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>

namespace map_image_generator
{
    geometry_msgs::PoseStamped convertMapToMapImage(const Parameters& parameters,
        const geometry_msgs::PoseStamped& mapPose);
    geometry_msgs::Pose convertMapToMapImage(const Parameters& parameters,
        const geometry_msgs::Pose& mapPose);

    geometry_msgs::PoseStamped convertMapImageToMap(const Parameters& parameters,
        const geometry_msgs::PoseStamped& mapImagePose);
    geometry_msgs::Pose convertMapImageToMap(const Parameters& parameters,
        const geometry_msgs::Pose& mapImagePose);

    void flipYawOnY(geometry_msgs::Pose& pose);
    double flipYawOnY(double yaw);
}

#endif