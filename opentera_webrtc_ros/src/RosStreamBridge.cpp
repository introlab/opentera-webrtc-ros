#include <ros/ros.h>
#include <RosStreamBridge.h>
#include <RosSignalingServerconfiguration.h>
#include <cv_bridge/cv_bridge.h>
#include <opentera_webrtc_ros_msgs/PeerImage.h>
#include <opentera_webrtc_ros_msgs/PeerAudio.h>
#include <opentera_webrtc_ros_msgs/PeerStatus.h>
#include <audio_utils/AudioFrame.h>
#include <RosNodeParameters.h>
#include <vector>

#include <RosWebRTCBridge.h>

using namespace opentera;
using namespace ros;
using namespace std;
using namespace opentera_webrtc_ros_msgs;
using namespace audio_utils;

/**
 * @brief construct a topic streamer node
 */
RosStreamBridge::RosStreamBridge(const ros::NodeHandle& nh)
    : RosWebRTCBridge(nh),
      m_videoSource(nullptr),
      m_audioSource(nullptr)
{
    if (RosNodeParameters::isStandAlone())
    {
        init(RosSignalingServerConfiguration::fromRosParam());
        connect();
    }
}

/**
 * @brief Initialize the stream client and his callback
 *
 * @param signalingServerConfiguration Signaling server configuration
 */
void RosStreamBridge::init(const opentera::SignalingServerConfiguration& signalingServerConfiguration)
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
    RosNodeParameters::loadAudioStreamParams(
        m_canSendAudioStream,
        m_canReceiveAudioStream,
        soundCardTotalDelayMs,
        echoCancellation,
        autoGainControl,
        noiseSuppression,
        highPassFilter,
        stereoSwapping,
        transientSuppression);


    RosNodeParameters::loadVideoStreamParams(
        m_canSendVideoStream,
        m_canReceiveVideoStream,
        needsDenoising,
        isScreencast);

    // WebRTC video stream interfaces
    m_videoSource = make_shared<RosVideoSource>(needsDenoising, isScreencast);
    m_audioSource = make_shared<RosAudioSource>(
        soundCardTotalDelayMs,
        echoCancellation,
        autoGainControl,
        noiseSuppression,
        highPassFilter,
        stereoSwapping,
        transientSuppression);


    bool verifySSL;
    RosNodeParameters::loadSignalingParamsVerifySSL(verifySSL);

    string iceServersUrl = RosSignalingServerConfiguration::getIceServerUrl(signalingServerConfiguration.url());
    ROS_INFO("RosStreamBridge Fetching ice servers from : %s", iceServersUrl.c_str());
    vector<IceServer> iceServers;
    if (!IceServer::fetchFromServer(iceServersUrl, signalingServerConfiguration.password(), iceServers, verifySSL))
    {
        ROS_ERROR("RosStreamBridge Error fetching ice servers from %s", iceServersUrl.c_str());
        iceServers.clear();
    }

    m_signalingClient = make_unique<StreamClient>(
        signalingServerConfiguration,
        WebrtcConfiguration::create(iceServers),
        (m_canReceiveVideoStream || m_canSendVideoStream ? m_videoSource : nullptr),
        (m_canReceiveAudioStream || m_canSendAudioStream ? m_audioSource : nullptr));


    m_signalingClient->setTlsVerificationEnabled(verifySSL);

    if (m_canReceiveAudioStream || m_canReceiveVideoStream)
    {
        // Stream event
        m_signalingClient->setOnAddRemoteStream(
            [this](const Client& client)
            {
                publishPeerStatus(client, PeerStatus::STATUS_REMOTE_STREAM_ADDED);
                ROS_INFO_STREAM(
                    nodeName << " --> "
                             << "RosStreamBridge Signaling on add remote stream: "
                             << "id: " << client.id() << ", name: " << client.name());
            });
        m_signalingClient->setOnRemoveRemoteStream(
            [this](const Client& client)
            {
                publishPeerStatus(client, PeerStatus::STATUS_REMOTE_STREAM_REMOVED);
                ROS_INFO_STREAM(
                    nodeName << " --> "
                             << "RosStreamBridge Signaling on remove remote stream: "
                             << "id: " << client.id() << ", name: " << client.name());
            });

        if (m_canReceiveAudioStream)
        {
            m_audioPublisher = m_nh.advertise<PeerAudio>("webrtc_audio", 100, false);
            m_mixedAudioPublisher = m_nh.advertise<AudioFrame>("audio_mixed", 100, false);

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
            m_imagePublisher = m_nh.advertise<PeerImage>("webrtc_image", 10, false);
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
    m_micVolumeSubscriber = m_nh.subscribe("mic_volume", 10, &RosStreamBridge::micVolumeCallback, this);
    m_enableCameraSubscriber = m_nh.subscribe("enable_camera", 10, &RosStreamBridge::enableCameraCallback, this);
    m_volumeSubscriber = m_nh.subscribe("volume", 10, &RosStreamBridge::volumeCallback, this);
}

void RosStreamBridge::onJoinSessionEvents(const std::vector<opentera_webrtc_ros_msgs::JoinSessionEvent>& events)
{
    // Already in a session ?
    // Should disconnect
    disconnect();

    ROS_INFO_STREAM(nodeName << " onJoinSessionEvents " << events[0].session_url);

    // TODO: Handle each item of the vector
    init(RosSignalingServerConfiguration::fromUrl(events[0].session_url));
    connect();
}

void RosStreamBridge::onStopSessionEvents(const std::vector<opentera_webrtc_ros_msgs::StopSessionEvent>& events)
{
    disconnect();
}

void RosStreamBridge::onSignalingConnectionOpened()
{
    RosWebRTCBridge::onSignalingConnectionOpened();

    if (m_canSendAudioStream)
    {
        // Audio
        m_audioSubscriber = m_nh.subscribe("audio_in", 10, &RosStreamBridge::audioCallback, this);
    }

    if (m_canSendVideoStream)
    {
        // Video
        m_imageSubscriber = m_nh.subscribe("ros_image", 5, &RosStreamBridge::imageCallback, this);
    }
}

void RosStreamBridge::onSignalingConnectionClosed()
{
    RosWebRTCBridge::onSignalingConnectionClosed();
    ROS_ERROR_STREAM(
        nodeName << " --> "
                 << "RosStreamBridge Signaling connection closed, shutting down...");
    ros::requestShutdown();
}

void RosStreamBridge::onSignalingConnectionError(const std::string& msg)
{
    RosWebRTCBridge::onSignalingConnectionError(msg);
    ROS_ERROR_STREAM(
        nodeName << " --> "
                 << "RosStreamBridge Signaling connection error " << msg.c_str() << ", shutting down...");
    ros::requestShutdown();
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
    std_msgs::Header imgHeader;
    imgHeader.stamp.fromNSec(1000 * timestampUs);

    sensor_msgs::ImagePtr img = cv_bridge::CvImage(imgHeader, "bgr8", bgrImg).toImageMsg();

    publishPeerFrame<PeerImage>(m_imagePublisher, client, *img);
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
    audio_utils::AudioFrame frame =
        createAudioFrame(audioData, bitsPerSample, sampleRate, numberOfChannels, numberOfFrames);
    publishPeerFrame<PeerAudio>(m_audioPublisher, client, frame);
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
    m_mixedAudioPublisher.publish(
        createAudioFrame(audioData, bitsPerSample, sampleRate, numberOfChannels, numberOfFrames));
}

audio_utils::AudioFrame RosStreamBridge::createAudioFrame(
    const void* audioData,
    int bitsPerSample,
    int sampleRate,
    size_t numberOfChannels,
    size_t numberOfFrames)
{
    audio_utils::AudioFrame frame;
    frame.format = "signed_" + to_string(bitsPerSample);
    frame.channel_count = numberOfChannels;
    frame.sampling_frequency = sampleRate;
    frame.frame_sample_count = numberOfFrames;

    const auto* buffer = reinterpret_cast<const uint8_t*>(audioData);
    size_t bufferSize = numberOfChannels * numberOfFrames * bitsPerSample / 8;
    frame.data = vector<uint8_t>(buffer, buffer + bufferSize);

    return frame;
}

void RosStreamBridge::audioCallback(const audio_utils::AudioFrameConstPtr& msg)
{
    if (m_audioSource)
    {
        m_audioSource->sendFrame(msg);
    }
}

void RosStreamBridge::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    if (m_videoSource)
    {
        m_videoSource->sendFrame(msg);
    }
}

void RosStreamBridge::micVolumeCallback(const std_msgs::Float32& msg)
{
    if (msg.data != 0)
    {
        m_signalingClient->setLocalAudioMuted(false);
    }
    else
    {
        m_signalingClient->setLocalAudioMuted(true);
    }
}

void RosStreamBridge::enableCameraCallback(const std_msgs::Bool& msg)
{
    m_signalingClient->setLocalVideoMuted(!msg.data);
}

void RosStreamBridge::volumeCallback(const std_msgs::Float32& msg)
{
    if (msg.data != 0)
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
    init(argc, argv, "stream_bridge");
    ros::NodeHandle nh;

    ROS_INFO_STREAM(
        ros::this_node::getName() << " --> "
                                  << "starting...");
    RosStreamBridge node(nh);
    node.run();
    ROS_INFO_STREAM(
        ros::this_node::getName() << " --> "
                                  << "done...");
}
