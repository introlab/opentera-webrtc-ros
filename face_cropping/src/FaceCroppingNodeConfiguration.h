#ifndef FACE_CROPPING_NODE_CONFIGURATION_H
#define FACE_CROPPING_NODE_CONFIGURATION_H

#include <rclcpp/rclcpp.hpp>

#include <optional>

struct FaceCroppingNodeConfiguration
{
    std::string faceDetectionModel;
    bool useGpuIfAvailable;

    float minFaceWidth;
    float minFaceHeight;
    int outputWidth;
    int outputHeight;

    bool adjustBrightness;

    static std::optional<FaceCroppingNodeConfiguration> fromRosParameters(rclcpp::Node& nodeHandle);
};

#endif
