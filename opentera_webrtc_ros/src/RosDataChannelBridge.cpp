#include <ros/ros.h>
#include <RosDataChannelBridge.h>
#include <RosSignalingServerconfiguration.h>
#include <opentera_webrtc_ros_msgs/PeerData.h>

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
}

/**
 * @brief Initialize the data channel client
 * 
 * @param signalingServerConfiguration Signaling server configuration
 */
void RosDataChannelBridge::initSignalingClient(const SignalingServerConfiguration &signalingServerConfiguration) 
{
    // Create signaling client
    m_signalingClient = make_unique<DataChannelClient>(
            signalingServerConfiguration,
            WebrtcConfiguration::create(),
            DataChannelConfiguration::create());

    m_signalingClient->setTlsVerificationEnabled(false);
}

/**
 * @brief Initialize the subscriber and publisher
 */
void RosDataChannelBridge::initAdvertiseTopics() 
{
    m_dataPublisher = m_nh.advertise<PeerData>("webrtc_data", 10);
    m_robotStatusSubscriber = m_nh.subscribe("robot_status", 10, &RosDataChannelBridge::onRosData, this);
    m_waypointReachedSubscriber = m_nh.subscribe("waypoint_reached", 10, &RosDataChannelBridge::onRosData, this);
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

void RosDataChannelBridge::onJoinSessionEvents(const std::vector<opentera_webrtc_ros_msgs::JoinSessionEvent> &events)
{
    initSignalingClient(RosSignalingServerConfiguration::fromUrl(events[0].session_url));
    initAdvertiseTopics();
    initDataChannelCallback();
    connect();
}

void RosDataChannelBridge::onStopSessionEvents(const std::vector<opentera_webrtc_ros_msgs::StopSessionEvent> &events) 
{
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
    ROS_ERROR_STREAM(nodeName << " --> " << "shutting down...");
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
    m_signalingClient->sendToAll(msg->data);
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

    RosDataChannelBridge node(nh);
    node.run();
}
