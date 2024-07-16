#include <rclcpp/rclcpp.hpp>
#include <opentera_webrtc_ros/RosDataChannelBridge.h>
#include <opentera_webrtc_ros/RosSignalingServerConfiguration.h>
#include <opentera_webrtc_ros_msgs/msg/peer_data.hpp>
#include <vector>

using namespace opentera;

/**
 * @brief Construct a data channel bridge
 */
RosDataChannelBridge::RosDataChannelBridge() : RosWebRTCBridge("data_channel_bridge")
{
    if (m_nodeParameters.isStandAlone())
    {
        initSignalingClient(RosSignalingServerConfiguration::fromRosParam(m_nodeParameters));
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
    if (m_nodeParameters.isStandAlone())
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
void RosDataChannelBridge::initSignalingClient(const SignalingServerConfiguration& signalingServerConfiguration)
{
    bool verifySSL;
    m_nodeParameters.loadSignalingParamsVerifySSL(verifySSL);

    std::string iceServersUrl =
        RosSignalingServerConfiguration::getIceServerUrl(m_nodeParameters, signalingServerConfiguration.url());
    RCLCPP_INFO(this->get_logger(), "Fetching ice servers from : %s", iceServersUrl.c_str());
    std::vector<IceServer> iceServers;
    if (!IceServer::fetchFromServer(iceServersUrl, signalingServerConfiguration.password(), iceServers, verifySSL))
    {
        RCLCPP_ERROR(this->get_logger(), "Error fetching ice servers from %s", iceServersUrl.c_str());
        iceServers.clear();
    }

    // Create signaling client
    m_signalingClient = std::make_unique<DataChannelClient>(
        signalingServerConfiguration,
        WebrtcConfiguration::create(iceServers),
        DataChannelConfiguration::create());

    m_signalingClient->setTlsVerificationEnabled(verifySSL);
}

/**
 * @brief Initialize the subscriber and publisher
 */
void RosDataChannelBridge::initAdvertiseTopics()
{
    m_dataPublisher = this->create_publisher<opentera_webrtc_ros_msgs::msg::PeerData>("webrtc_data_incoming", 10);
    m_dataSubscriber = this->create_subscription<std_msgs::msg::String>(
        "webrtc_data_outgoing",
        10,
        bind_this<std_msgs::msg::String>(this, &RosDataChannelBridge::onRosData));
    m_callAllSubscriber = this->create_subscription<std_msgs::msg::Empty>(
        "call_all",
        10,
        bind_this<std_msgs::msg::Empty>(this, &RosDataChannelBridge::callAllCallBack));
}

/**
 * @brief Clear the subscriber and publisher
 */
void RosDataChannelBridge::stopAdvertiseTopics()
{
    m_dataPublisher.reset();
    m_dataSubscriber.reset();
}

/**
 * @brief Initialize the data channel client callback
 */
void RosDataChannelBridge::initDataChannelCallback()
{
    // Setup data channel callback
    m_signalingClient->setOnDataChannelMessageString(
        [&](const Client& client, const std::string& data)
        {
            opentera_webrtc_ros_msgs::msg::PeerData msg;
            // TODO PeerData should have a json string sent by client.data()
            msg.data = data;

            msg.sender.id = client.id();
            msg.sender.name = client.name();
            m_dataPublisher->publish(msg);
        });
}

/**
 * @brief Stop the data channel client callback
 */
void RosDataChannelBridge::stopDataChannelCallback()
{
    // Setup data channel callback
    if (m_signalingClient)
    {
        m_signalingClient->setOnDataChannelMessageString(nullptr);
    }
}

void RosDataChannelBridge::callAllCallBack(const std_msgs::msg::Empty::ConstSharedPtr& msg)
{
    (void)msg;
    m_signalingClient->callAll();
}

void RosDataChannelBridge::onJoinSessionEvents(
    const std::vector<opentera_webrtc_ros_msgs::msg::JoinSessionEvent>& events)
{
    initSignalingClient(RosSignalingServerConfiguration::fromUrl(m_nodeParameters, events[0].session_url));
    initAdvertiseTopics();
    initDataChannelCallback();
    connect();
}

void RosDataChannelBridge::onStopSessionEvents(
    const std::vector<opentera_webrtc_ros_msgs::msg::StopSessionEvent>& events)
{
    (void)events;
    stopAdvertiseTopics();
    stopDataChannelCallback();
    disconnect();
}

void RosDataChannelBridge::onSignalingConnectionClosed()
{
    RosWebRTCBridge::onSignalingConnectionClosed();
    RCLCPP_WARN(this->get_logger(), " --> shutting down...");
    rclcpp::shutdown();
}

void RosDataChannelBridge::onSignalingConnectionError(const std::string& msg)
{
    RosWebRTCBridge::onSignalingConnectionError(msg);
    RCLCPP_ERROR_STREAM(this->get_logger(), " --> " << msg << " shutting down...");
    rclcpp::shutdown();
}

/**
 * @brief Send the data received trough ROS to the WebRTC data channel
 *
 * Is used as a callback for the ROS subscriber.
 * Data is sent to all connected peers if the data channel is connected
 *
 * @param msg the received ROS message
 */
void RosDataChannelBridge::onRosData(const std_msgs::msg::String::ConstSharedPtr& msg)
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
    rclcpp::init(argc, argv);

    auto node = std::make_shared<RosDataChannelBridge>();

    RCLCPP_INFO(node->get_logger(), " --> starting...");
    node->run();
    RCLCPP_INFO(node->get_logger(), " --> done...");
}
