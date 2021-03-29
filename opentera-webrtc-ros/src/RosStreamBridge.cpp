#include <ros/ros.h>
#include <RosStreamBridge.h>
#include <RosSignalingServerconfiguration.h>
#include <cv_bridge/cv_bridge.h>
#include <opentera_webrtc_ros/PeerImage.h>
#include <opentera_webrtc_ros/PeerAudio.h>
#include <audio_utils/AudioFrame.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace opentera;
using namespace ros;
using namespace std;
using namespace opentera_webrtc_ros;

/**
 * @brief construct a topic streamer node
 */
RosStreamBridge::RosStreamBridge() {
    bool needsDenoising;
    bool isScreencast;

    // Load ROS parameters
    loadStreamParams(needsDenoising, isScreencast);

    // WebRTC video stream interfaces
    m_videoSource = make_shared<RosVideoSource>(needsDenoising, isScreencast);
    m_audioSource = make_shared<RosAudioSource>();
    
    // m_videoSink = make_shared<VideoSink>([&](const cv::Mat& bgrImg, uint64_t timestampUs){
    //     onFrameReceived(bgrImg, timestampUs);
    // });

    // Signaling client connection
    /*
            StreamClient(SignalingServerConfiguration signalingServerConfiguration,
                WebrtcConfiguration webrtcConfiguration,
                std::shared_ptr<VideoSource> videoSource,
                std::shared_ptr<AudioSource> audioSource);
    
     std::shared_ptr<opentera::RosVideoSource>&, std::shared_ptr<opentera::VideoSink>&)’
    
    */
    m_signalingClient = make_unique<StreamClient>(
            RosSignalingServerConfiguration::fromRosParam("streamer"),
            WebrtcConfiguration::create(),
            m_videoSource);

    //m_imagePublisher = m_nh.advertise<sensor_msgs::Image>("webrtc_image", 1, false);
    m_imagePublisher = m_nh.advertise<PeerImage>("webrtc_image", 1, false);
    m_audioPublisher = m_nh.advertise<PeerAudio>("webrtc_audio", 1, false);

    //TODO configure callbacks for audio & video



    // Subscribe to image topic when signaling client connects
    m_signalingClient->setOnSignalingConnectionOpened([&]{
        ROS_INFO("Signaling connection opened, streaming topic...");
        m_imageSubsriber = m_nh.subscribe(
                "ros_image",
                1,
                &RosVideoSource::imageCallback,
                m_videoSource.get());
    });

    // Shutdown ROS when signaling client disconnect
    m_signalingClient->setOnSignalingConnectionClosed([]{
        ROS_WARN("Signaling connection closed, shutting down...");
        requestShutdown();
    });

    // Shutdown ROS on signaling client error
    m_signalingClient->setOnSignalingConnectionError([](auto msg){
        ROS_ERROR("Signaling connection error %s, shutting down...", msg.c_str());
        requestShutdown();
    });

    m_signalingClient->setOnRoomClientsChanged([](const vector<RoomClient>& roomClients){
        ROS_INFO("Signaling on room clients changed: ");
        for (const auto& client : roomClients) {
            ROS_INFO_STREAM("\tid: " << client.id() << ", name: " << client.name() << ", isConnected: " << client.isConnected());
        }
    });

    // Connection's event
    m_signalingClient->setOnClientConnected([](const Client& client){
        ROS_INFO_STREAM("Signaling on client connected: " << "id: " << client.id() << ", name: " << client.name());
    });
    m_signalingClient->setOnClientDisconnected([](const Client& client){
        ROS_INFO_STREAM("Signaling on client disconnected: " << "id: " << client.id() << ", name: " << client.name());
    });

    // Stream event
    m_signalingClient->setOnAddRemoteStream([](const Client& client) {
        ROS_INFO_STREAM("Signaling on add remote stream: " << "id: " << client.id() << ", name: " << client.name());
    });
    m_signalingClient->setOnRemoveRemoteStream([](const Client& client) {
        ROS_INFO_STREAM("Signaling on remove remote stream: " << "id: " << client.id() << ", name: " << client.name());
    });

    // Video and audio frame
    m_signalingClient->setOnVideoFrameReceived([&](const Client& client, const cv::Mat& bgrImg, uint64_t timestampUS){
        // TMP
        // cv::imshow(client.id(), bgrImg);
        // cv::waitKey(1);

        std_msgs::Header imgHeader;
        imgHeader.stamp.fromNSec(1000 * timestampUS);

        sensor_msgs::ImagePtr img = cv_bridge::CvImage(imgHeader, "bgr8", bgrImg).toImageMsg();

        publishPeerFrame<PeerImage>(m_imagePublisher, client, *img);
    });
    m_signalingClient->setOnAudioFrameReceived([&](const Client& client,
        const void* audioData,
        int bitsPerSample,
        int sampleRate,
        size_t numberOfChannels,
        size_t numberOfFrames)
    {
        audio_utils::AudioFrame frame;
        frame.format = "signed_16";
        frame.channel_count = numberOfChannels;
        frame.sampling_frequency = sampleRate;
        frame.frame_sample_count = numberOfFrames;

        uint8_t* charBuf = (uint8_t*)const_cast<void*>(audioData);
        frame.data = vector<uint8_t>(charBuf, charBuf + sizeof(charBuf));

        publishPeerFrame<PeerAudio>(m_audioPublisher, client, frame);
    });
}

/**
 * @brief publish an image using the node image publisher
 *
 * @param bgrImg BGR8 encoded image
 * @param timestampUs image timestamp in microseconds
 */
void RosStreamBridge::onFrameReceived(const cv::Mat& bgrImg, uint64_t timestampUs)
{
    std_msgs::Header imgHeader;
    imgHeader.stamp.fromNSec(1000 * timestampUs);

    sensor_msgs::ImagePtr imgMsg = cv_bridge::CvImage(imgHeader, "bgr8", bgrImg).toImageMsg();
    m_imagePublisher.publish(imgMsg);
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
 * @brief Load video stream parameters from ROS parameter server
 *
 * @param denoise whether the images require denoising
 * @param screencast whether the images are a screen capture
 */
void RosStreamBridge::loadStreamParams(bool &denoise, bool &screencast)
{
    NodeHandle pnh("~stream");

    pnh.param("is_screen_cast", screencast, false);
    pnh.param("needs_denoising", denoise, false);
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
