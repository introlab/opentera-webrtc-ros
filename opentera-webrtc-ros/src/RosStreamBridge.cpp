#include <ros/ros.h>
#include <RosStreamBridge.h>
#include <RosSignalingServerconfiguration.h>
#include <cv_bridge/cv_bridge.h>
#include <opentera_webrtc_ros_msgs/PeerImage.h>
#include <opentera_webrtc_ros_msgs/PeerAudio.h>
#include <audio_utils/AudioFrame.h>
#include <RosNodeParameters.h>

using namespace opentera;
using namespace ros;
using namespace std;
using namespace opentera_webrtc_ros_msgs;

/**
 * @brief construct a topic streamer node
 */
RosStreamBridge::RosStreamBridge() {
    bool needsDenoising;
    bool isScreencast;
    bool canSendStream;
    bool canReceiveStream;

    nodeName = ros::this_node::getName();

    // Load ROS parameters
    RosNodeParameters::loadStreamParams(canSendStream, canReceiveStream, needsDenoising, isScreencast);

    // WebRTC video stream interfaces
    m_videoSource = make_shared<RosVideoSource>(needsDenoising, isScreencast);
    m_audioSource = make_shared<RosAudioSource>();
    
    m_signalingClient = make_unique<StreamClient>(
            RosSignalingServerConfiguration::fromRosParam(),
            WebrtcConfiguration::create(),
            m_videoSource);

    m_imagePublisher = m_nh.advertise<PeerImage>("webrtc_image", 1, false);
    m_audioPublisher = m_nh.advertise<PeerAudio>("webrtc_audio", 1, false);



    // Shutdown ROS when signaling client disconnect
    m_signalingClient->setOnSignalingConnectionClosed([&]{
        ROS_WARN_STREAM(nodeName << " --> " << "Signaling connection closed, shutting down...");
        requestShutdown();
    });

    // Shutdown ROS on signaling client error
    m_signalingClient->setOnSignalingConnectionError([&](auto msg){
        ROS_ERROR_STREAM(nodeName << " --> " << "Signaling connection error " << msg.c_str() << ", shutting down...");
        requestShutdown();
    });

    m_signalingClient->setOnRoomClientsChanged([&](const vector<RoomClient>& roomClients){
        ROS_INFO_STREAM(nodeName << " --> " << "Signaling on room clients changed: ");
        for (const auto& client : roomClients) {
            ROS_INFO_STREAM("\tid: " << client.id() << ", name: " << client.name() << ", isConnected: " << client.isConnected());
        }
    });

    // Connection's event
    m_signalingClient->setOnClientConnected([&](const Client& client){
        ROS_INFO_STREAM(nodeName << " --> " 
                        << "Signaling on client connected: " << "id: " << client.id() << ", name: " << client.name());
    });
    m_signalingClient->setOnClientDisconnected([&](const Client& client){
        ROS_INFO_STREAM(nodeName << " --> " 
                        << "Signaling on client disconnected: " << "id: " << client.id() << ", name: " << client.name());
    });


    if (canSendStream) {

        // Subscribe to image topic when signaling client connects
        m_signalingClient->setOnSignalingConnectionOpened([&]{
            ROS_INFO_STREAM(nodeName << " --> " << "Signaling connection opened, streaming topic...");
            m_imageSubsriber = m_nh.subscribe(
                    "ros_image",
                    1,
                    &RosVideoSource::imageCallback,
                    m_videoSource.get());
        });
    }

    if (canReceiveStream) {

        // Stream event
        m_signalingClient->setOnAddRemoteStream([&](const Client& client) {
            ROS_INFO_STREAM(nodeName << " --> "
                            << "Signaling on add remote stream: " << "id: " << client.id() << ", name: " << client.name());
        });
        m_signalingClient->setOnRemoveRemoteStream([&](const Client& client) {
            ROS_INFO_STREAM(nodeName << " --> "
                            << "Signaling on remove remote stream: " << "id: " << client.id() << ", name: " << client.name());
        });

        // Video and audio frame
        m_signalingClient->setOnVideoFrameReceived([&](const Client& client, const cv::Mat& bgrImg, uint64_t timestampUS) {
            onVideoFrameReceived(client, bgrImg, timestampUS);
        });
        m_signalingClient->setOnAudioFrameReceived([&](const Client& client,
            const void* audioData,
            int bitsPerSample,
            int sampleRate,
            size_t numberOfChannels,
            size_t numberOfFrames)
        {
            onAudioFrameReceived(client, audioData, bitsPerSample, sampleRate, numberOfChannels, numberOfFrames);
        });
    }    
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

    uint8_t* charBuf = (uint8_t*)const_cast<void*>(audioData);
    frame.data = vector<uint8_t>(charBuf, charBuf + sizeof(charBuf));

    publishPeerFrame<PeerAudio>(m_audioPublisher, client, frame);       
}

/**
 * @brief Close signaling client connection when this object is destroyed
 */
RosStreamBridge::~RosStreamBridge()
{
    ROS_INFO("ROS is shutting down, closing signaling client connection.");
    m_signalingClient->closeSync();
    ROS_INFO("Signaling client disconnected, goodbye.");
}

/**
 * @brief Connect to server and process images forever
 */
void RosStreamBridge::run()
{
    ROS_INFO("Connecting to signaling server at.");
    m_signalingClient->connect();
    spin();
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

    RosStreamBridge node;
    node.run();
}
