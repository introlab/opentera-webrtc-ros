#include <ros/ros.h>
#include <RosStreamBridge.h>
#include <RosSignalingServerconfiguration.h>
#include <cv_bridge/cv_bridge.h>
#include <opentera_webrtc_ros_msgs/PeerImage.h>
#include <opentera_webrtc_ros_msgs/PeerAudio.h>
#include <opentera_webrtc_ros_msgs/PeerStatus.h>
#include <audio_utils/AudioFrame.h>
#include <RosNodeParameters.h>

#include <RosWebRTCBridge.h>

using namespace opentera;
using namespace ros;
using namespace std;
using namespace opentera_webrtc_ros_msgs;

/**
 * @brief construct a topic streamer node
 */
RosStreamBridge::RosStreamBridge(const ros::NodeHandle& nh): RosWebRTCBridge(nh), m_videoSource(nullptr), m_audioSource(nullptr)
{
    if (RosNodeParameters::isStandAlone()) {
        init(RosSignalingServerConfiguration::fromRosParam());
        connect();
    }
}

/**
 * @brief Initialize the stream client and his callback
 * 
 * @param signalingServerConfiguration Signaling server configuration
 */
void RosStreamBridge::init(const opentera::SignalingServerConfiguration &signalingServerConfiguration) 
{
    bool needsDenoising, isScreencast;

    // Load ROS parameters
    RosNodeParameters::loadStreamParams(m_canSendStream, m_canReceiveStream, needsDenoising, isScreencast);

    // WebRTC video stream interfaces
    m_videoSource = make_shared<RosVideoSource>(needsDenoising, isScreencast);
    m_audioSource = make_shared<RosAudioSource>();
    
    m_signalingClient = make_unique<StreamClient>(
            signalingServerConfiguration,
            WebrtcConfiguration::create(),
            m_videoSource);

    m_signalingClient->setTlsVerificationEnabled(false);

    if (m_canReceiveStream) {

        m_imagePublisher = m_nh.advertise<PeerImage>("webrtc_image", 10, false);
        m_audioPublisher = m_nh.advertise<PeerAudio>("webrtc_audio", 100, false);
        m_peerStatusPublisher = m_nh.advertise<PeerStatus>("webrtc_peer_status", 10, false);

        // Stream event
        m_signalingClient->setOnAddRemoteStream([&, this](const Client& client) {
            this->publishPeerStatus(client, PeerStatus::STATUS_REMOTE_STREAM_ADDED);
            ROS_INFO_STREAM(nodeName << " --> "
                            << "Signaling on add remote stream: " << "id: " << client.id() << ", name: " << client.name());
        });
        m_signalingClient->setOnRemoveRemoteStream([&, this](const Client& client) {
            this->publishPeerStatus(client, PeerStatus::STATUS_REMOTE_STREAM_REMOVED);
            ROS_INFO_STREAM(nodeName << " --> "
                            << "Signaling on remove remote stream: " << "id: " << client.id() << ", name: " << client.name());
        });

        // Video and audio frame
        m_signalingClient->setOnVideoFrameReceived(std::bind(&RosStreamBridge::onVideoFrameReceived, this,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3));
        m_signalingClient->setOnAudioFrameReceived(std::bind(&RosStreamBridge::onAudioFrameReceived, this,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3,
            std::placeholders::_4,
            std::placeholders::_5,
            std::placeholders::_6));
    } 
}

void RosStreamBridge::publishPeerStatus(const Client &client, int status)
{
    PeerStatus msg;
    msg.sender.id = client.id();
    msg.sender.name = client.name();
    msg.status = status;
    m_peerStatusPublisher.publish(msg);
}

void RosStreamBridge::onJoinSessionEvents(const std::vector<opentera_webrtc_ros_msgs::JoinSessionEvent> &events) 
{
    // TODO: Handle each item of the vector
    init(RosSignalingServerConfiguration::fromUrl(events[0].session_url));
    connect();
}

void RosStreamBridge::onStopSessionEvents(const std::vector<opentera_webrtc_ros_msgs::StopSessionEvent> &events) 
{
    disconnect();
}


void RosStreamBridge::onSignalingConnectionOpened() 
{
    RosWebRTCBridge::onSignalingConnectionOpened();

    if (m_canSendStream) {
        m_imageSubscriber = m_nh.subscribe(
            "ros_image",
            1,
            &RosVideoSource::imageCallback,
            m_videoSource.get());
    }
}

void RosStreamBridge::onSignalingConnectionClosed() 
{
    RosWebRTCBridge::onSignalingConnectionClosed();
    ROS_WARN_STREAM(nodeName << " --> " << "shutting down...");
    ros::requestShutdown();
}

void RosStreamBridge::onSignalingConnectionError(const std::string& msg) 
{
    RosWebRTCBridge::onSignalingConnectionError(msg);
    ROS_ERROR_STREAM(nodeName << " --> " << "shutting down...");
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
void RosStreamBridge::onAudioFrameReceived(const Client& client,
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

    const uint8_t* buffer = reinterpret_cast<const uint8_t*>(audioData);
    size_t bufferSize = numberOfChannels * numberOfFrames * bitsPerSample / 8;
    frame.data = vector<uint8_t>(buffer, buffer + bufferSize);

    publishPeerFrame<PeerAudio>(m_audioPublisher, client, frame);       
}

RosStreamBridge::~RosStreamBridge()
{

}

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

    RosStreamBridge node(nh);
    node.run();
}
