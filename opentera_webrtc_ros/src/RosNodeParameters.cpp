#include <rclcpp/rclcpp.hpp>
#include <opentera_webrtc_ros/RosNodeParameters.h>

#include <stdexcept>

using namespace opentera;

bool RosNodeParameters::isStandAlone(rclcpp::Node& node)
{
    return node.declare_parameter("is_stand_alone", true);
}

/**
 * @brief Load signaling parameters from ROS parameter server
 *
 * @param clientName Client's name
 * @param room Room's name
 */
void RosNodeParameters::loadSignalingParams(rclcpp::Node& node, std::string& clientName, std::string& room)
{
    std::map<std::string, std::string> dict;

    clientName = node.declare_parameter("signaling.client_name", "streamer");
    room = node.declare_parameter("signaling.room_name", "chat");
}

/**
 * @brief Load signaling parameters from ROS parameter server
 *
 * @param serverUrl Signaling server Url
 * @param clientName Client's name
 * @param room Room's name
 * @param password Room's password
 */
void RosNodeParameters::loadSignalingParams(
    rclcpp::Node& node,
    std::string& serverUrl,
    std::string& clientName,
    std::string& room,
    std::string& password)
{
    serverUrl = node.declare_parameter("signaling.server_url", "http://localhost:8080");
    clientName = node.declare_parameter("signaling.client_name", "streamer");
    room = node.declare_parameter("signaling.room_name", "chat");
    password = node.declare_parameter("signaling.room_password", "abc");
}

/**
 * @brief Load signaling parameters (verify_ssl) from ROS parameter server
 *
 * @param verifySSL Verify SSL peer
 */
void RosNodeParameters::loadSignalingParamsVerifySSL(rclcpp::Node& node, bool& verifySSL)
{
    verifySSL = node.declare_parameter("signaling.verify_ssl", true);
}

/**
 * @brief Load video stream parameters from ROS parameter server
 *
 * @param canSendVideoStream whether the node can send video stream to the signaling server
 * @param canReceiveVideoStream whether the node can received video stream from the signaling server
 * @param denoise whether the images require denoising
 * @param screencast whether the images are a screen capture
 */
void RosNodeParameters::loadVideoStreamParams(
    rclcpp::Node& node,
    bool& canSendVideoStream,
    bool& canReceiveVideoStream,
    bool& denoise,
    bool& screencast)
{
    canSendVideoStream = node.declare_parameter("video_stream.can_send_video_stream", false);
    canReceiveVideoStream = node.declare_parameter("video_stream.can_receive_video_stream", false);
    denoise = node.declare_parameter("video_stream.needs_denoising", false);
    screencast = node.declare_parameter("video_stream.is_screen_cast", false);
}

/**
 * @brief Load video codec parameters from ROS parameter server
 *
 * @param forcedCodecs The codecs that must be used, an empty set means all codecs.
 * @param forceGStreamerHardwareAcceleration whether to force GStreamer hardware acceleration
 * @param useGStreamerSoftwareEncoderDecoder whether to use GStreamer software encoders/decoders
 */
void RosNodeParameters::loadVideoCodecParams(
    rclcpp::Node& node,
    std::unordered_set<VideoStreamCodec>& forcedCodecs,
    bool& forceGStreamerHardwareAcceleration,
    bool& useGStreamerSoftwareEncoderDecoder)
{
    std::vector<std::string> forcedCodecStrings =
        node.declare_parameter("video_codecs.forced_codecs", std::vector<std::string>());

    transform(
        forcedCodecStrings.begin(),
        forcedCodecStrings.end(),
        inserter(forcedCodecs, forcedCodecs.begin()),
        [](const std::string& codecString)
        {
            auto codec = stringToVideoStreamCodec(codecString);
            if (codec.has_value())
            {
                return *codec;
            }
            else
            {
                throw std::runtime_error("Invalid codec: " + codecString);
            }
        });

    forceGStreamerHardwareAcceleration =
        node.declare_parameter("video_codecs.force_gstreamer_hardware_acceleration", false);
    useGStreamerSoftwareEncoderDecoder =
        node.declare_parameter("video_codecs.use_gstreamer_software_encoder_decoder", false);
}

/**
 * @brief Load audio stream parameters from ROS parameter server
 *
 * @param canSendAudioStream whether the node can send audio stream to the signaling server
 * @param canReceiveAudioStream whether the node can received audio stream from the signaling server
 */
void RosNodeParameters::loadAudioStreamParams(
    rclcpp::Node& node,
    bool& canSendAudioStream,
    bool& canReceiveAudioStream,
    unsigned int& soundCardTotalDelayMs,
    bool& echoCancellation,
    bool& autoGainControl,
    bool& noiseSuppression,
    bool& highPassFilter,
    bool& stereoSwapping,
    bool& transientSuppression)
{
    canSendAudioStream = node.declare_parameter("audio_stream.can_send_audio_stream", false);
    canReceiveAudioStream = node.declare_parameter("audio_stream.can_receive_audio_stream", false);
    soundCardTotalDelayMs = node.declare_parameter("audio_stream.sound_card_total_delay_ms", 40);
    echoCancellation = node.declare_parameter("audio_stream.echo_cancellation", true);
    autoGainControl = node.declare_parameter("audio_stream.auto_gain_control", true);
    noiseSuppression = node.declare_parameter("audio_stream.noise_suppression", true);
    highPassFilter = node.declare_parameter("audio_stream.high_pass_filter", false);
    stereoSwapping = node.declare_parameter("audio_stream.stereo_swapping", false);
    transientSuppression = node.declare_parameter("audio_stream.transient_suppression", true);
}
