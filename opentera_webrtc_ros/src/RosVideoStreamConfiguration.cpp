#include <ros/node_handle.h>
#include <RosVideoStreamConfiguration.h>
#include <RosStreamBridge.h>
#include <RosNodeParameters.h>

using namespace opentera;
using namespace std;
using namespace ros;

/**
 * @brief Build a video stream configuration from the ROS parameter server
 * Parameters are retrieved from the signaling namespace under the node private namespace
 *
 * @return The video stream configuration
 */
VideoStreamConfiguration RosVideoStreamConfiguration::fromRosParam()
{
    unordered_set<VideoStreamCodec> forcedCodecs;
    bool forceGStreamerHardwareAcceleration, useGStreamerSoftwareEncoderDecoder;

    RosNodeParameters::loadVideoCodecParams(forcedCodecs, forceGStreamerHardwareAcceleration, useGStreamerSoftwareEncoderDecoder);

    return VideoStreamConfiguration::create(forcedCodecs, forceGStreamerHardwareAcceleration, useGStreamerSoftwareEncoderDecoder);
}
