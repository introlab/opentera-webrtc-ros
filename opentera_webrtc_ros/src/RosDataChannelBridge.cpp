#include <ros/ros.h>
#include <RosDataChannelBridge.h>
#include <RosSignalingServerconfiguration.h>
#include <opentera_webrtc_ros_msgs/PeerData.h>
#include <vector>

using namespace opentera;
using namespace ros;
using namespace std_msgs;
using namespace std;
using namespace opentera_webrtc_ros_msgs;

/**
 * @brief Construct a data channel bridge
 */
RosDataChannelBridge::RosDataChannelBridge(const ros::NodeHandle& nh): RosWebRTCBridge(nh)
{
    if (RosNodeParameters::isStandAlone())
    {
        initSignalingClient(RosSignalingServerConfiguration::fromRosParam());
        initAdvertiseTopics();
        initDataChannelCallback();
        connect();
    }
}

/**
 * @brief Close signaling client connection when this object is destroyed
 */
RosDataChannelBridge::~RosDataChannelBridge()
{
    if (RosNodeParameters::isStandAlone())
    {
        m_signalingClient = nullptr;
        stopAdvertiseTopics();
        stopDataChannelCallback();
        disconnect();
    }
}

/**
 * @brief Initialize the data channel client
 *
 * @param signalingServerConfiguration Signaling server configuration
 */
void RosDataChannelBridge::initSignalingClient(const SignalingServerConfiguration &signalingServerConfiguration)
{

    size_t pos1 = 0;
    pos1 = signalingServerConfiguration.url().find_last_of("/");
    string iceServersUrl = signalingServerConfiguration.url().substr(0, pos1) + "/iceservers";
    ROS_INFO("Fetching ice servers from : %s", iceServersUrl.c_str());
    vector<IceServer> iceServers;
    if (!IceServer::fetchFromServer(iceServersUrl,
        signalingServerConfiguration.password(), iceServers))
    {
        ROS_ERROR("Error fetching ice servers from %s", iceServersUrl.c_str());
        iceServers.clear();
    }

    // Create signaling client
    m_signalingClient = make_unique<DataChannelClient>(
            signalingServerConfiguration,
            WebrtcConfiguration::create(iceServers),
            DataChannelConfiguration::create());

    m_signalingClient->setTlsVerificationEnabled(false);
}

/**
 * @brief Initialize the subscriber and publisher
 */
void RosDataChannelBridge::initAdvertiseTopics()
{
    m_dataPublisher = m_nh.advertise<PeerData>("webrtc_data_incoming", 10);
    m_dataSubscriber = m_nh.subscribe("webrtc_data_outgoing", 10, &RosDataChannelBridge::onRosData, this);
}

/**
 * @brief Initialize the subscriber and publisher
 */
void RosDataChannelBridge::stopAdvertiseTopics()
{
    m_dataPublisher.shutdown();
    m_dataSubscriber.shutdown();
}

/**
 * @brief Initialize the data channel client callback
 */
void RosDataChannelBridge::initDataChannelCallback()
{
    // Setup data channel callback
    m_signalingClient->setOnDataChannelMessageString([&](const Client& client, const string& data)
    {
        PeerData msg;
        //TODO PeerData should have a json string sent by client.data()
        msg.data = data;

        msg.sender.id = client.id();
        msg.sender.name = client.name();
        m_dataPublisher.publish(msg);
    });
}

/**
 * @brief Stop the data channel client callback
 */
void RosDataChannelBridge::stopDataChannelCallback()
{
    // Setup data channel callback
    m_signalingClient->setOnDataChannelMessageString(nullptr);
}

void RosDataChannelBridge::onJoinSessionEvents(const std::vector<opentera_webrtc_ros_msgs::JoinSessionEvent> &events)
{
    initSignalingClient(RosSignalingServerConfiguration::fromUrl(events[0].session_url));
    initAdvertiseTopics();
    initDataChannelCallback();
    connect();
}

void RosDataChannelBridge::onStopSessionEvents(const std::vector<opentera_webrtc_ros_msgs::StopSessionEvent> &events)
{
    stopAdvertiseTopics();
    stopDataChannelCallback();
    disconnect();
}

void RosDataChannelBridge::onSignalingConnectionClosed()
{
    RosWebRTCBridge::onSignalingConnectionClosed();
    ROS_WARN_STREAM(nodeName << " --> " << "shutting down...");
    ros::requestShutdown();
}

void RosDataChannelBridge::onSignalingConnectionError(const std::string& msg)
{
    RosWebRTCBridge::onSignalingConnectionError(msg);
    ROS_ERROR_STREAM(nodeName << " --> " << msg << " shutting down...");
    ros::requestShutdown();
}

/**
 * @brief Send the data received trough ROS to the WebRTC data channel
 *
 * Is used as a callback for the ROS subscriber.
 * Data is sent to all connected peers if the data channel is connected
 *
 * @param msg the received ROS message
 */
void RosDataChannelBridge::onRosData(const StringConstPtr& msg)
{
    if (m_signalingClient)
    {
        m_signalingClient->sendToAll(msg->data);
    }
}

/**
 * @brief runs a ROS data channel bridge
 *
 * @param argc ROS argument count
 * @param argv ROS argument values
 * @return nothing
 */
int main(int argc, char** argv)
{
    init(argc, argv, "data_channel_bridge");
    ros::NodeHandle nh;
    ROS_INFO_STREAM(ros::this_node::getName() << " --> " << "starting...");
    RosDataChannelBridge node(nh);
    node.run();
    ROS_INFO_STREAM(ros::this_node::getName()<< " --> " << "done...");
}
