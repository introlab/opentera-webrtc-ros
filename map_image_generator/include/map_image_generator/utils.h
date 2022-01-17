#ifndef UTILS_H
#define UTILS_H

#include "map_image_generator/Parameters.h"

#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/tf.h>

namespace map_image_generator
{
    geometry_msgs::PoseStamped
    convertMapToMapImage(const Parameters& parameters,
                         const geometry_msgs::PoseStamped& mapPose);
    geometry_msgs::Pose convertMapToMapImage(const Parameters& parameters,
                                             const geometry_msgs::Pose& mapPose);

    geometry_msgs::Pose
    convertRobotCenteredMapCoordinatesToPose(const Parameters& parameters, int x, int y,
                                             double yaw);

    geometry_msgs::PoseStamped
    convertMapImageToRobot(const Parameters& parameters,
                           const geometry_msgs::PoseStamped& mapImagePose);
    geometry_msgs::Pose convertMapImageToRobot(const Parameters& parameters,
                                               const geometry_msgs::Pose& mapImagePose);

    geometry_msgs::PoseStamped
    convertMapImageToMap(const Parameters& parameters,
                         const geometry_msgs::PoseStamped& mapImagePose);
    geometry_msgs::Pose convertMapImageToMap(const Parameters& parameters,
                                             const geometry_msgs::Pose& mapImagePose);

    void offsetYawByMinus90Degrees(geometry_msgs::Pose& pose);

    void flipYawOnY(geometry_msgs::Pose& pose);
    void flipYawOnY(tf::Transform& transform);
    double flipYawOnY(double yaw);

    template <typename T>
    inline T sign(T val)
    {
        return static_cast<T>(T{0} < val) - static_cast<T>(val < T{0});
    }

    template <typename T>
    inline T restrictToPositive(T val)
    {
        return std::max(T{0}, val);
    }

    template <typename T>
    inline T deg2rad(T deg)
    {
        return deg * M_PI / 180;
    }

    template <typename T>
    inline T rad2deg(T rad)
    {
        return rad * 180 / M_PI;
    }
}

#endif
