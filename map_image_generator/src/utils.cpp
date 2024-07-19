#include "map_image_generator/utils.h"

#include <cmath>
#include <tf2/utils.h>

using namespace std;

namespace map_image_generator
{
    tf2::Quaternion createQuaternionFromYaw(double yaw)
    {
        tf2::Quaternion quat;
        quat.setRPY(0, 0, yaw);
        return quat;
    }

    geometry_msgs::msg::Quaternion createQuaternionMsgFromYaw(double yaw)
    {
        return tf2::toMsg(createQuaternionFromYaw(yaw));
    }

    geometry_msgs::msg::PoseStamped
        convertMapToMapImage(const Parameters& parameters, const geometry_msgs::msg::PoseStamped& mapPose)
    {
        geometry_msgs::msg::PoseStamped mapImagePose;
        mapImagePose.header.stamp = mapPose.header.stamp;
        mapImagePose.header.frame_id = "map_image";

        mapImagePose.pose = convertMapToMapImage(parameters, mapPose.pose);
        return mapImagePose;
    }

    geometry_msgs::msg::Pose convertMapToMapImage(const Parameters& parameters, const geometry_msgs::msg::Pose& mapPose)
    {
        geometry_msgs::msg::Pose mapImagePose;

        double mapImageWidth = parameters.resolution() * parameters.width();
        mapImagePose.position.x = mapImageWidth - (mapPose.position.x * parameters.resolution() + parameters.xOrigin());
        mapImagePose.position.y = mapPose.position.y * parameters.resolution() + parameters.yOrigin();
        mapImagePose.position.z = 0;

        mapImagePose.orientation = mapPose.orientation;
        flipYawOnY(mapImagePose);
        return mapImagePose;
    }

    geometry_msgs::msg::Pose
        convertRobotCenteredMapCoordinatesToPose(const Parameters& parameters, int x, int y, double yaw)
    {
        geometry_msgs::msg::Pose pose;
        int rows = parameters.height() * parameters.resolution();
        int cols = parameters.width() * parameters.resolution();
        double centerY = rows / 2.0;
        double centerX = cols / 2.0;
        pose.position.x =
            -(y - centerY - parameters.robotVerticalOffset()) / static_cast<double>(parameters.resolution());
        pose.position.y = -(x - centerX) / static_cast<double>(parameters.resolution());
        pose.position.z = 0;
        pose.orientation = createQuaternionMsgFromYaw(yaw);
        offsetYawByMinus90Degrees(pose);
        flipYawOnY(pose);
        return pose;
    }

    geometry_msgs::msg::PoseStamped
        convertMapImageToRobot(const Parameters& parameters, const geometry_msgs::msg::PoseStamped& mapImagePose)
    {
        geometry_msgs::msg::PoseStamped robotPose;
        robotPose.header.stamp = mapImagePose.header.stamp;
        robotPose.header.frame_id = parameters.robotFrameId();

        robotPose.pose = convertMapImageToRobot(parameters, mapImagePose.pose);
        return robotPose;
    }

    geometry_msgs::msg::Pose
        convertMapImageToRobot(const Parameters& parameters, const geometry_msgs::msg::Pose& mapImagePose)
    {
        return convertRobotCenteredMapCoordinatesToPose(
            parameters,
            static_cast<int>(mapImagePose.position.x),
            static_cast<int>(mapImagePose.position.y),
            tf2::getYaw(mapImagePose.orientation));
    }

    geometry_msgs::msg::PoseStamped
        convertMapImageToMap(const Parameters& parameters, const geometry_msgs::msg::PoseStamped& mapImagePose)
    {
        geometry_msgs::msg::PoseStamped mapPose;
        mapPose.header.stamp = mapImagePose.header.stamp;
        mapPose.header.frame_id = parameters.mapFrameId();

        mapPose.pose = convertMapImageToMap(parameters, mapImagePose.pose);
        return mapPose;
    }

    geometry_msgs::msg::Pose
        convertMapImageToMap(const Parameters& parameters, const geometry_msgs::msg::Pose& mapImagePose)
    {
        geometry_msgs::msg::Pose mapPose;

        double flippedXOnY = parameters.resolution() * parameters.width() - mapImagePose.position.x;
        mapPose.position.x = (flippedXOnY - parameters.xOrigin()) / parameters.resolution() / parameters.scaleFactor();
        mapPose.position.y =
            (mapImagePose.position.y - parameters.yOrigin()) / parameters.resolution() / parameters.scaleFactor();
        mapPose.position.z = 0;

        mapPose.orientation = mapImagePose.orientation;
        flipYawOnY(mapPose);
        return mapPose;
    }

    void offsetYawByMinus90Degrees(geometry_msgs::msg::Pose& pose)
    {
        double yaw = tf2::getYaw(pose.orientation);
        yaw -= M_PI_2;
        pose.orientation = createQuaternionMsgFromYaw(yaw);
    }

    cv::Scalar interpolateColors(const cv::Scalar& color1, const cv::Scalar& color2, double ratio)
    {
        cv::Scalar color;
        color[0] = (color2[0] - color1[0]) * ratio + color1[0];
        color[1] = (color2[1] - color1[1]) * ratio + color1[1];
        color[2] = (color2[2] - color1[2]) * ratio + color1[2];
        color[3] = (color2[3] - color1[3]) * ratio + color1[3];
        return color;
    }

    void flipYawOnY(geometry_msgs::msg::Pose& pose)
    {
        double yaw = tf2::getYaw(pose.orientation);

        pose.orientation = createQuaternionMsgFromYaw(flipYawOnY(yaw));
    }

    void flipYawOnY(tf2::Transform& transform)
    {
        double yaw = tf2::getYaw(transform.getRotation());

        transform.setRotation(createQuaternionFromYaw(flipYawOnY(yaw)));
    }

    double flipYawOnY(double yaw)
    {
        yaw = fmod(yaw, 2 * M_PI);
        if (0 <= yaw && yaw <= M_PI)
        {
            return M_PI - yaw;
        }
        else if (M_PI <= yaw && yaw <= 2 * M_PI)
        {
            return 3 * M_PI - yaw;
        }
        else if (-M_PI <= yaw && yaw <= 0)
        {
            return -M_PI - yaw;
        }
        else if (-2 * M_PI <= yaw && yaw <= -M_PI)
        {
            return -3 * M_PI - yaw;
        }
        return yaw;
    }
}
