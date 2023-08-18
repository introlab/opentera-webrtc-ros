#include "FaceCroppingNodeConfiguration.h"

std::optional<FaceCroppingNodeConfiguration>
    FaceCroppingNodeConfiguration::fromRosParameters(const ros::NodeHandle& privateNodeHandle)
{
    FaceCroppingNodeConfiguration configuration;

    if (!privateNodeHandle.getParam("face_detection_model", configuration.faceDetectionModel))
    {
        ROS_ERROR("The parameter face_detection_model is required.");
        return std::nullopt;
    }
    privateNodeHandle.param("use_gpu_if_available", configuration.useGpuIfAvailable, false);


    if (!privateNodeHandle.getParam("min_face_width", configuration.minFaceWidth))
    {
        ROS_ERROR("The parameter min_face_width is required.");
        return std::nullopt;
    }
    if (!privateNodeHandle.getParam("min_face_height", configuration.minFaceHeight))
    {
        ROS_ERROR("The parameter min_face_height is required.");
        return std::nullopt;
    }
    if (!privateNodeHandle.getParam("output_width", configuration.outputWidth))
    {
        ROS_ERROR("The parameter output_width is required.");
        return std::nullopt;
    }
    if (!privateNodeHandle.getParam("output_height", configuration.outputHeight))
    {
        ROS_ERROR("The parameter output_height is required.");
        return std::nullopt;
    }
    privateNodeHandle.param("adjust_brightness", configuration.adjustBrightness, true);

    return configuration;
}
