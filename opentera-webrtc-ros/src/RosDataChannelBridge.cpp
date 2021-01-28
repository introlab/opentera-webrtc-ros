#include <ros/ros.h>
#include <RosDataChannelBridge.h>
#include <RosSignalingServerconfiguration.h>
#include <opentera_webrtc_ros/PeerData.h>

using namespace opentera;
using namespace ros;
using namespace std_msgs;
using namespace std;
using namespace opentera_webrtc_ros;

/**
 * @brief Construct a data channel bridge
 */
RosDataChannelBridge::RosDataChannelBridge()
{
    NodeHandle nh;

    // Create signaling client
    m_signalingClient = make_unique<DataChannelClient>(
            RosSignalingServerConfiguration::fromRosParam("data_bridge"),
            WebrtcConfiguration::create(),
            DataChannelConfiguration::create());

    // Advertise topics
    m_dataPublisher = nh.advertise<PeerData>("webrtc_data", 10);
    m_dataSubscriber = nh.subscribe("ros_data", 10, &RosDataChannelBridge::onRosData, this);

    // Setup data channel callback
    m_signalingClient->setOnDataChannelMessageString([&](const Client& client, const string& data) {
        PeerData msg;
        //TODO PeerData should have a json string sent by client.data()
        msg.data = data;
        
        msg.sender.id = client.id();
        msg.sender.name = client.name();
        m_dataPublisher.publish(msg);
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
}

/**
 * @brief Close signaling client connection when this object is destroyed
 */
RosDataChannelBridge::~RosDataChannelBridge()
{
    ROS_INFO("ROS is shutting down, closing signaling client connection.");
    m_signalingClient->closeSync();
    ROS_INFO("Signaling client disconnected, goodbye.");
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
 * @brief Connect to the server and forward data forever
 */
void RosDataChannelBridge::run()
{
    m_signalingClient->connect();
    spin();
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

    RosDataChannelBridge node;
    node.run();
}
