#include <rclcpp/rclcpp.hpp>
#include <opentera_webrtc_ros/RosStreamBridge.h>
#include <opentera_webrtc_ros/RosSignalingServerConfiguration.h>
#include <opentera_webrtc_ros/RosVideoStreamConfiguration.h>
#include <cv_bridge/cv_bridge.h>
#include <opentera_webrtc_ros_msgs/msg/peer_image.hpp>
#include <opentera_webrtc_ros_msgs/msg/peer_audio.hpp>
#include <opentera_webrtc_ros_msgs/msg/peer_status.hpp>
#include <audio_utils_msgs/msg/audio_frame.hpp>
#include <opentera_webrtc_ros/RosNodeParameters.h>
#include <vector>

#include <opentera_webrtc_ros/RosWebRTCBridge.h>

using namespace opentera;

/**
 * @brief construct a topic streamer node
 */
RosStreamBridge::RosStreamBridge() : RosWebRTCBridge("stream_bridge"), m_videoSource(nullptr), m_audioSource(nullptr)
{
    if (m_nodeParameters.isStandAlone())
    {
        init(
            RosSignalingServerConfiguration::fromRosParam(m_nodeParameters),
            RosVideoStreamConfiguration::fromRosParam(m_nodeParameters));
        connect();
    }
}

/**
 * @brief Initialize the stream client and his callback
 *
 * @param signalingServerConfiguration Signaling server configuration
 */
void RosStreamBridge::init(
    const opentera::SignalingServerConfiguration& signalingServerConfiguration,
    const opentera::VideoStreamConfiguration& videoStreamConfiguration)
{
    bool needsDenoising, isScreencast;
    unsigned int soundCardTotalDelayMs;
    bool echoCancellation;
    bool autoGainControl;
    bool noiseSuppression;
    bool highPassFilter;
    bool stereoSwapping;
    bool transientSuppression;

    // Load ROS parameters
    m_nodeParameters.loadAudioStreamParams(
        m_canSendAudioStream,
        m_canReceiveAudioStream,
        soundCardTotalDelayMs,
        echoCancellation,
        autoGainControl,
        noiseSuppression,
        highPassFilter,
        stereoSwapping,
        transientSuppression);


    m_nodeParameters.loadVideoStreamParams(m_canSendVideoStream, m_canReceiveVideoStream, needsDenoising, isScreencast);

    // WebRTC video stream interfaces
    m_videoSource = std::make_shared<RosVideoSource>(needsDenoising, isScreencast);
    m_audioSource = std::make_shared<RosAudioSource>(
        *this,
        soundCardTotalDelayMs,
        echoCancellation,
        autoGainControl,
        noiseSuppression,
        highPassFilter,
        stereoSwapping,
        transientSuppression);


    bool verifySSL;
    m_nodeParameters.loadSignalingParamsVerifySSL(verifySSL);

    std::string iceServersUrl =
        RosSignalingServerConfiguration::getIceServerUrl(m_nodeParameters, signalingServerConfiguration.url());
    RCLCPP_INFO(this->get_logger(), "RosStreamBridge Fetching ice servers from : %s", iceServersUrl.c_str());
    std::vector<IceServer> iceServers;
    if (!IceServer::fetchFromServer(iceServersUrl, signalingServerConfiguration.password(), iceServers, verifySSL))
    {
        RCLCPP_ERROR(this->get_logger(), "RosStreamBridge Error fetching ice servers from %s", iceServersUrl.c_str());
        iceServers.clear();
    }

    m_signalingClient = std::make_unique<StreamClient>(
        signalingServerConfiguration,
        WebrtcConfiguration::create(iceServers),
        videoStreamConfiguration,
        (m_canReceiveVideoStream || m_canSendVideoStream ? m_videoSource : nullptr),
        (m_canReceiveAudioStream || m_canSendAudioStream ? m_audioSource : nullptr));


    m_signalingClient->setTlsVerificationEnabled(verifySSL);

    if (m_canReceiveAudioStream || m_canReceiveVideoStream)
    {
        // Stream event
        m_signalingClient->setOnAddRemoteStream(
            [this](const Client& client)
            {
                publishPeerStatus(client, opentera_webrtc_ros_msgs::msg::PeerStatus::STATUS_REMOTE_STREAM_ADDED);
                RCLCPP_INFO_STREAM(
                    this->get_logger(),
                    " --> RosStreamBridge Signaling on add remote stream: " << "id: " << client.id()
                                                                            << ", name: " << client.name());
            });
        m_signalingClient->setOnRemoveRemoteStream(
            [this](const Client& client)
            {
                publishPeerStatus(client, opentera_webrtc_ros_msgs::msg::PeerStatus::STATUS_REMOTE_STREAM_REMOVED);
                RCLCPP_INFO_STREAM(
                    this->get_logger(),
                    " --> RosStreamBridge Signaling on remove remote stream: " << "id: " << client.id()
                                                                               << ", name: " << client.name());
            });

        if (m_canReceiveAudioStream)
        {
            m_audioPublisher = this->create_publisher<opentera_webrtc_ros_msgs::msg::PeerAudio>("webrtc_audio", 100);
            m_mixedAudioPublisher = this->create_publisher<audio_utils_msgs::msg::AudioFrame>("audio_mixed", 100);

            m_signalingClient->setOnAudioFrameReceived(
                [this](auto&& PH1, auto&& PH2, auto&& PH3, auto&& PH4, auto&& PH5, auto&& PH6)
                {
                    onAudioFrameReceived(
                        std::forward<decltype(PH1)>(PH1),
                        std::forward<decltype(PH2)>(PH2),
                        std::forward<decltype(PH3)>(PH3),
                        std::forward<decltype(PH4)>(PH4),
                        std::forward<decltype(PH5)>(PH5),
                        std::forward<decltype(PH6)>(PH6));
                });

            m_signalingClient->setOnMixedAudioFrameReceived(
                [this](auto&& PH1, auto&& PH2, auto&& PH3, auto&& PH4, auto&& PH5)
                {
                    onMixedAudioFrameReceived(
                        std::forward<decltype(PH1)>(PH1),
                        std::forward<decltype(PH2)>(PH2),
                        std::forward<decltype(PH3)>(PH3),
                        std::forward<decltype(PH4)>(PH4),
                        std::forward<decltype(PH5)>(PH5));
                });
        }

        if (m_canReceiveVideoStream)
        {
            m_imagePublisher = this->create_publisher<opentera_webrtc_ros_msgs::msg::PeerImage>("webrtc_image", 10);
            // Video and audio frame
            m_signalingClient->setOnVideoFrameReceived(
                [this](auto&& PH1, auto&& PH2, auto&& PH3)
                {
                    onVideoFrameReceived(
                        std::forward<decltype(PH1)>(PH1),
                        std::forward<decltype(PH2)>(PH2),
                        std::forward<decltype(PH3)>(PH3));
                });
        }
    }
    m_callAllSubscriber = this->create_subscription<std_msgs::msg::Empty>(
        "call_all",
        10,
        bind_this<std_msgs::msg::Empty>(this, &RosStreamBridge::callAllCallBack));
    m_micVolumeSubscriber = this->create_subscription<std_msgs::msg::Float32>(
        "mic_volume",
        10,
        bind_this<std_msgs::msg::Float32>(this, &RosStreamBridge::micVolumeCallback));
    m_enableCameraSubscriber = this->create_subscription<std_msgs::msg::Bool>(
        "enable_camera",
        10,
        bind_this<std_msgs::msg::Bool>(this, &RosStreamBridge::enableCameraCallback));
    m_volumeSubscriber = this->create_subscription<std_msgs::msg::Float32>(
        "volume",
        10,
        bind_this<std_msgs::msg::Float32>(this, &RosStreamBridge::volumeCallback));
}

void RosStreamBridge::onJoinSessionEvents(const std::vector<opentera_webrtc_ros_msgs::msg::JoinSessionEvent>& events)
{
    // Already in a session ?
    // Should disconnect
    disconnect();

    RCLCPP_INFO_STREAM(this->get_logger(), " onJoinSessionEvents " << events[0].session_url);

    // TODO: Handle each item of the vector
    init(
        RosSignalingServerConfiguration::fromUrl(m_nodeParameters, events[0].session_url),
        RosVideoStreamConfiguration::fromRosParam(m_nodeParameters));
    connect();
}

void RosStreamBridge::onStopSessionEvents(const std::vector<opentera_webrtc_ros_msgs::msg::StopSessionEvent>& events)
{
    (void)events;
    disconnect();
}

void RosStreamBridge::onSignalingConnectionOpened()
{
    RosWebRTCBridge::onSignalingConnectionOpened();

    if (m_canSendAudioStream)
    {
        // Audio
        m_audioSubscriber = this->create_subscription<audio_utils_msgs::msg::AudioFrame>(
            "audio_in",
            10,
            bind_this<audio_utils_msgs::msg::AudioFrame>(this, &RosStreamBridge::audioCallback));
    }

    if (m_canSendVideoStream)
    {
        // Video
        m_imageSubscriber = this->create_subscription<sensor_msgs::msg::Image>(
            "ros_image",
            10,
            bind_this<sensor_msgs::msg::Image>(this, &RosStreamBridge::imageCallback));
    }
}

void RosStreamBridge::onSignalingConnectionClosed()
{
    RosWebRTCBridge::onSignalingConnectionClosed();
    RCLCPP_ERROR(this->get_logger(), " --> RosStreamBridge Signaling connection closed, shutting down...");
    rclcpp::shutdown();
}

void RosStreamBridge::onSignalingConnectionError(const std::string& msg)
{
    RosWebRTCBridge::onSignalingConnectionError(msg);
    RCLCPP_ERROR_STREAM(
        this->get_logger(),
        " --> RosStreamBridge Signaling connection error " << msg.c_str() << ", shutting down...");
    rclcpp::shutdown();
}

/**
 * @brief publish an image using the node image publisher
 *
 * @param client Client who sent the frame
 * @param bgrImg BGR8 encoded image
 * @param timestampUs image timestamp in microseconds
 */
void RosStreamBridge::onVideoFrameReceived(const Client& client, const cv::Mat& bgrImg, uint64_t timestampUs)
{
    std_msgs::msg::Header imgHeader;
    imgHeader.stamp = from_microseconds(timestampUs);

    sensor_msgs::msg::Image::SharedPtr img = cv_bridge::CvImage(imgHeader, "bgr8", bgrImg).toImageMsg();

    publishPeerFrame(*m_imagePublisher, client, *img);
}

/**
 * @brief Format a PeerAudio message for publishing
 *
 * @param client Client who sent the frame
 * @param audioData
 * @param bitsPerSample format of the sample
 * @param sampleRate Sampling frequency
 * @param numberOfChannels
 * @param numberOfFrames
 */
void RosStreamBridge::onAudioFrameReceived(
    const Client& client,
    const void* audioData,
    int bitsPerSample,
    int sampleRate,
    size_t numberOfChannels,
    size_t numberOfFrames)
{
    audio_utils_msgs::msg::AudioFrame frame =
        createAudioFrame(audioData, bitsPerSample, sampleRate, numberOfChannels, numberOfFrames);
    publishPeerFrame(*m_audioPublisher, client, frame);
}

/**
 * @brief Format a AudioFrame message for publishing
 * @param audioData
 * @param bitsPerSample format of the sample
 * @param sampleRate Sampling frequency
 * @param numberOfChannels
 * @param numberOfFrames
 */
void RosStreamBridge::onMixedAudioFrameReceived(
    const void* audioData,
    int bitsPerSample,
    int sampleRate,
    size_t numberOfChannels,
    size_t numberOfFrames)
{
    m_mixedAudioPublisher->publish(
        createAudioFrame(audioData, bitsPerSample, sampleRate, numberOfChannels, numberOfFrames));
}

audio_utils_msgs::msg::AudioFrame RosStreamBridge::createAudioFrame(
    const void* audioData,
    int bitsPerSample,
    int sampleRate,
    size_t numberOfChannels,
    size_t numberOfFrames)
{
    audio_utils_msgs::msg::AudioFrame frame;
    frame.format = "signed_" + std::to_string(bitsPerSample);
    frame.channel_count = numberOfChannels;
    frame.sampling_frequency = sampleRate;
    frame.frame_sample_count = numberOfFrames;

    const auto* buffer = reinterpret_cast<const uint8_t*>(audioData);
    size_t bufferSize = numberOfChannels * numberOfFrames * bitsPerSample / 8;
    frame.data = std::vector<uint8_t>(buffer, buffer + bufferSize);

    return frame;
}

void RosStreamBridge::audioCallback(const audio_utils_msgs::msg::AudioFrame::ConstSharedPtr& msg)
{
    if (m_audioSource)
    {
        m_audioSource->sendFrame(msg);
    }
}

void RosStreamBridge::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
{
    if (m_videoSource)
    {
        m_videoSource->sendFrame(msg);
    }
}

void RosStreamBridge::callAllCallBack(const std_msgs::msg::Empty::ConstSharedPtr& msg)
{
    (void)msg;
    m_signalingClient->callAll();
}

void RosStreamBridge::micVolumeCallback(const std_msgs::msg::Float32::ConstSharedPtr& msg)
{
    if (msg->data != 0)
    {
        m_signalingClient->setLocalAudioMuted(false);
    }
    else
    {
        m_signalingClient->setLocalAudioMuted(true);
    }
}

void RosStreamBridge::enableCameraCallback(const std_msgs::msg::Bool::ConstSharedPtr& msg)
{
    m_signalingClient->setLocalVideoMuted(!msg->data);
}

void RosStreamBridge::volumeCallback(const std_msgs::msg::Float32::ConstSharedPtr& msg)
{
    if (msg->data != 0)
    {
        m_signalingClient->setRemoteAudioMuted(false);
    }
    else
    {
        m_signalingClient->setRemoteAudioMuted(true);
    }
}

RosStreamBridge::~RosStreamBridge() = default;

/**
 * @brief runs a ROS topic streamer Node
 *
 * @param argc ROS argument count
 * @param argv ROS argument values
 * @return nothing
 */
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<RosStreamBridge>();

    RCLCPP_INFO(node->get_logger(), " --> starting...");
    node->run();
    RCLCPP_INFO(node->get_logger(), " --> done...");
}
