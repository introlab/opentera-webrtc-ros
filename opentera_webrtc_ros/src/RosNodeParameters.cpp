#include <rclcpp/rclcpp.hpp>
#include <opentera_webrtc_ros/RosNodeParameters.h>

#include <stdexcept>

using namespace opentera;

namespace
{
    std::vector<std::string> remove_empty_strings(const std::vector<std::string>& vec)
    {
        std::vector<std::string> out;
        std::copy_if(
            vec.begin(),
            vec.end(),
            std::back_inserter(out),
            [](const std::string& codecString) { return !codecString.empty(); });

        return out;
    }
}

RosNodeParameters::RosNodeParameters(rclcpp::Node& node) : m_node{node}
{
    m_node.declare_parameter("is_stand_alone", true);

    // Signaling params
    m_node.declare_parameter("signaling.client_name", "streamer");
    m_node.declare_parameter("signaling.room_name", "chat");
    m_node.declare_parameter("signaling.server_url", "http://localhost:8080");
    m_node.declare_parameter("signaling.room_password", "abc");
    m_node.declare_parameter("signaling.verify_ssl", true);

    // Video stream params
    m_node.declare_parameter("video_stream.can_send_video_stream", false);
    m_node.declare_parameter("video_stream.can_receive_video_stream", false);
    m_node.declare_parameter("video_stream.needs_denoising", false);
    m_node.declare_parameter("video_stream.is_screen_cast", false);

    // Video codec params
    m_node.declare_parameter("video_codecs.forced_codecs", std::vector<std::string>());
    m_node.declare_parameter("video_codecs.force_gstreamer_hardware_acceleration", false);
    m_node.declare_parameter("video_codecs.use_gstreamer_software_encoder_decoder", false);

    // Audio stream params
    m_node.declare_parameter("audio_stream.can_send_audio_stream", false);
    m_node.declare_parameter("audio_stream.can_receive_audio_stream", false);
    m_node.declare_parameter("audio_stream.sound_card_total_delay_ms", 40);
    m_node.declare_parameter("audio_stream.echo_cancellation", true);
    m_node.declare_parameter("audio_stream.auto_gain_control", true);
    m_node.declare_parameter("audio_stream.noise_suppression", true);
    m_node.declare_parameter("audio_stream.high_pass_filter", false);
    m_node.declare_parameter("audio_stream.stereo_swapping", false);
    m_node.declare_parameter("audio_stream.transient_suppression", true);
}

bool RosNodeParameters::isStandAlone() const
{
    return m_node.get_parameter("is_stand_alone").as_bool();
}

/**
 * @brief Load signaling parameters from ROS parameter server
 *
 * @param clientName Client's name
 * @param room Room's name
 */
void RosNodeParameters::loadSignalingParams(std::string& clientName, std::string& room) const
{
    clientName = m_node.get_parameter("signaling.client_name").as_string();
    room = m_node.get_parameter("signaling.room_name").as_string();
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
    std::string& serverUrl,
    std::string& clientName,
    std::string& room,
    std::string& password) const
{
    serverUrl = m_node.get_parameter("signaling.server_url").as_string();
    clientName = m_node.get_parameter("signaling.client_name").as_string();
    room = m_node.get_parameter("signaling.room_name").as_string();
    password = m_node.get_parameter("signaling.room_password").as_string();
}

/**
 * @brief Load signaling parameters (verify_ssl) from ROS parameter server
 *
 * @param verifySSL Verify SSL peer
 */
void RosNodeParameters::loadSignalingParamsVerifySSL(bool& verifySSL) const
{
    verifySSL = m_node.get_parameter("signaling.verify_ssl").as_bool();
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
    bool& canSendVideoStream,
    bool& canReceiveVideoStream,
    bool& denoise,
    bool& screencast) const
{
    canSendVideoStream = m_node.get_parameter("video_stream.can_send_video_stream").as_bool();
    canReceiveVideoStream = m_node.get_parameter("video_stream.can_receive_video_stream").as_bool();
    denoise = m_node.get_parameter("video_stream.needs_denoising").as_bool();
    screencast = m_node.get_parameter("video_stream.is_screen_cast").as_bool();
}

/**
 * @brief Load video codec parameters from ROS parameter server
 *
 * @param forcedCodecs The codecs that must be used, an empty set means all codecs.
 * @param forceGStreamerHardwareAcceleration whether to force GStreamer hardware acceleration
 * @param useGStreamerSoftwareEncoderDecoder whether to use GStreamer software encoders/decoders
 */
void RosNodeParameters::loadVideoCodecParams(
    std::unordered_set<VideoStreamCodec>& forcedCodecs,
    bool& forceGStreamerHardwareAcceleration,
    bool& useGStreamerSoftwareEncoderDecoder) const
{
    std::vector<std::string> forcedCodecStrings =
        remove_empty_strings(m_node.get_parameter("video_codecs.forced_codecs").as_string_array());

    std::transform(
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
        m_node.get_parameter("video_codecs.force_gstreamer_hardware_acceleration").as_bool();
    useGStreamerSoftwareEncoderDecoder =
        m_node.get_parameter("video_codecs.use_gstreamer_software_encoder_decoder").as_bool();
}

/**
 * @brief Load audio stream parameters from ROS parameter server
 *
 * @param canSendAudioStream whether the node can send audio stream to the signaling server
 * @param canReceiveAudioStream whether the node can received audio stream from the signaling server
 */
void RosNodeParameters::loadAudioStreamParams(
    bool& canSendAudioStream,
    bool& canReceiveAudioStream,
    unsigned int& soundCardTotalDelayMs,
    bool& echoCancellation,
    bool& autoGainControl,
    bool& noiseSuppression,
    bool& highPassFilter,
    bool& stereoSwapping,
    bool& transientSuppression) const
{
    canSendAudioStream = m_node.get_parameter("audio_stream.can_send_audio_stream").as_bool();
    canReceiveAudioStream = m_node.get_parameter("audio_stream.can_receive_audio_stream").as_bool();
    soundCardTotalDelayMs = m_node.get_parameter("audio_stream.sound_card_total_delay_ms").as_int();
    echoCancellation = m_node.get_parameter("audio_stream.echo_cancellation").as_bool();
    autoGainControl = m_node.get_parameter("audio_stream.auto_gain_control").as_bool();
    noiseSuppression = m_node.get_parameter("audio_stream.noise_suppression").as_bool();
    highPassFilter = m_node.get_parameter("audio_stream.high_pass_filter").as_bool();
    stereoSwapping = m_node.get_parameter("audio_stream.stereo_swapping").as_bool();
    transientSuppression = m_node.get_parameter("audio_stream.transient_suppression").as_bool();
}
