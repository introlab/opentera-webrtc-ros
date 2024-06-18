#include "FaceCroppingNodeConfiguration.h"

std::optional<FaceCroppingNodeConfiguration> FaceCroppingNodeConfiguration::fromRosParameters(rclcpp::Node& nodeHandle)
{
    FaceCroppingNodeConfiguration configuration;

    nodeHandle.declare_parameter("face_detection_model", rclcpp::PARAMETER_STRING);
    nodeHandle.declare_parameter("min_face_width", rclcpp::PARAMETER_DOUBLE);
    nodeHandle.declare_parameter("min_face_height", rclcpp::PARAMETER_DOUBLE);
    nodeHandle.declare_parameter("output_width", rclcpp::PARAMETER_INTEGER);
    nodeHandle.declare_parameter("output_height", rclcpp::PARAMETER_INTEGER);

    if (!nodeHandle.get_parameter("face_detection_model", configuration.faceDetectionModel))
    {
        RCLCPP_ERROR(nodeHandle.get_logger(), "The parameter face_detection_model is required.");
        return std::nullopt;
    }
    configuration.useGpuIfAvailable = nodeHandle.declare_parameter("use_gpu_if_available", false);


    if (!nodeHandle.get_parameter("min_face_width", configuration.minFaceWidth))
    {
        RCLCPP_ERROR(nodeHandle.get_logger(), "The parameter min_face_width is required.");
        return std::nullopt;
    }
    if (!nodeHandle.get_parameter("min_face_height", configuration.minFaceHeight))
    {
        RCLCPP_ERROR(nodeHandle.get_logger(), "The parameter min_face_height is required.");
        return std::nullopt;
    }
    if (!nodeHandle.get_parameter("output_width", configuration.outputWidth))
    {
        RCLCPP_ERROR(nodeHandle.get_logger(), "The parameter output_width is required.");
        return std::nullopt;
    }
    if (!nodeHandle.get_parameter("output_height", configuration.outputHeight))
    {
        RCLCPP_ERROR(nodeHandle.get_logger(), "The parameter output_height is required.");
        return std::nullopt;
    }
    configuration.adjustBrightness = nodeHandle.declare_parameter("adjust_brightness", true);

    return configuration;
}
