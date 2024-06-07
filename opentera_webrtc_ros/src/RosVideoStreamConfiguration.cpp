#include <rclcpp/rclcpp.hpp>
#include <opentera_webrtc_ros/RosVideoStreamConfiguration.h>
#include <opentera_webrtc_ros/RosStreamBridge.h>
#include <opentera_webrtc_ros/RosNodeParameters.h>

using namespace opentera;
using namespace std;
using namespace rclcpp;

/**
 * @brief Build a video stream configuration from the ROS parameter server
 * Parameters are retrieved from the signaling namespace under the node private namespace
 *
 * @return The video stream configuration
 */
VideoStreamConfiguration RosVideoStreamConfiguration::fromRosParam(RosNodeParameters& nodeParameters)
{
    unordered_set<VideoStreamCodec> forcedCodecs;
    bool forceGStreamerHardwareAcceleration, useGStreamerSoftwareEncoderDecoder;

    nodeParameters.loadVideoCodecParams(
        forcedCodecs,
        forceGStreamerHardwareAcceleration,
        useGStreamerSoftwareEncoderDecoder);

    return VideoStreamConfiguration::create(
        forcedCodecs,
        forceGStreamerHardwareAcceleration,
        useGStreamerSoftwareEncoderDecoder);
}
