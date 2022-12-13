#ifndef OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_WEBRTC_BRIDGE_H
#define OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_WEBRTC_BRIDGE_H

#include <ros/ros.h>
#include <signal.h>

#include <OpenteraWebrtcNativeClient/Configurations/SignalingServerConfiguration.h>
#include <OpenteraWebrtcNativeClient/Utils/Client.h>
#include <OpenteraWebrtcNativeClient/SignalingClient.h>

#include <opentera_webrtc_ros_msgs/OpenTeraEvent.h>
#include <opentera_webrtc_ros_msgs/DatabaseEvent.h>
#include <opentera_webrtc_ros_msgs/DeviceEvent.h>
#include <opentera_webrtc_ros_msgs/JoinSessionEvent.h>
#include <opentera_webrtc_ros_msgs/JoinSessionReplyEvent.h>
#include <opentera_webrtc_ros_msgs/LeaveSessionEvent.h>
#include <opentera_webrtc_ros_msgs/LogEvent.h>
#include <opentera_webrtc_ros_msgs/ParticipantEvent.h>
#include <opentera_webrtc_ros_msgs/StopSessionEvent.h>
#include <opentera_webrtc_ros_msgs/UserEvent.h>
#include <opentera_webrtc_ros_msgs/PeerStatus.h>

#include <RosNodeParameters.h>
#include <RosSignalingServerconfiguration.h>

namespace opentera
{

    /**
     * @brief Interface a ROS node to bridge WebRTC to ROS topics
     */
    template<typename T>
    class RosWebRTCBridge
    {
        static_assert(std::is_base_of<SignalingClient, T>::value, "T must inherit from opentera::SignalingClient");

    private:
        ros::Subscriber m_eventSubscriber;
        bool m_hasMultipleDevices;

    protected:
        std::string nodeName;
        ros::NodeHandle m_nh;
        ros::Publisher m_peerStatusPublisher;
        std::unique_ptr<T> m_signalingClient;

        virtual void connect();
        virtual void disconnect();

        virtual void onEvent(const opentera_webrtc_ros_msgs::OpenTeraEventConstPtr& msg);
        virtual void onDataBaseEvents(const std::vector<opentera_webrtc_ros_msgs::DatabaseEvent>& events);
        virtual void onDeviceEvents(const std::vector<opentera_webrtc_ros_msgs::DeviceEvent>& events);
        virtual void onJoinSessionEvents(const std::vector<opentera_webrtc_ros_msgs::JoinSessionEvent>& events);
        virtual void
            onJoinSessionReplyEvents(const std::vector<opentera_webrtc_ros_msgs::JoinSessionReplyEvent>& events);
        virtual void onLeaveSessionEvents(const std::vector<opentera_webrtc_ros_msgs::LeaveSessionEvent>& events);
        virtual void onLogEvents(const std::vector<opentera_webrtc_ros_msgs::LogEvent>& events);
        virtual void onParticipantEvents(const std::vector<opentera_webrtc_ros_msgs::ParticipantEvent>& events);
        virtual void onStopSessionEvents(const std::vector<opentera_webrtc_ros_msgs::StopSessionEvent>& events);
        virtual void onUserEvents(const std::vector<opentera_webrtc_ros_msgs::UserEvent>& events);

        virtual void connectSignalingClientEvents();
        virtual void onSignalingConnectionOpened();
        virtual void onSignalingConnectionClosed();
        virtual void onSignalingConnectionError(const std::string& msg);
        virtual void onRoomClientsChanged(const std::vector<RoomClient>& roomClients);
        virtual bool callAcceptor(const Client& client);
        virtual void onCallRejected(const Client& client);
        virtual void onClientConnected(const Client& client);
        virtual void onClientDisconnected(const Client& client);
        virtual void onError(const std::string& error);

        template<typename PeerMsg>
        void publishPeerFrame(ros::Publisher& publisher, const Client& client, const decltype(PeerMsg::frame)& frame);
        void publishPeerStatus(const Client& client, int status);

    public:
        RosWebRTCBridge(const ros::NodeHandle& nh);
        virtual ~RosWebRTCBridge() = 0;

        void run();
    };

    /**
     * @brief construct a topic streamer node
     *
     * @param nh node handle.
     */
    template<typename T>
    RosWebRTCBridge<T>::RosWebRTCBridge(const ros::NodeHandle& nh) : m_signalingClient(nullptr),
                                                                     m_nh(nh)
    {
        // On CTRL+C exit
        signal(SIGINT, [](int sig) { ros::requestShutdown(); });

        nodeName = ros::this_node::getName();

        if (!RosNodeParameters::isStandAlone())
        {
            m_eventSubscriber = m_nh.subscribe("events", 10, &RosWebRTCBridge::onEvent, this);
        }

        m_peerStatusPublisher = m_nh.advertise<opentera_webrtc_ros_msgs::PeerStatus>("webrtc_peer_status", 10, false);
    }

    /**
     * @brief Close signaling client connection when this object is destroyed
     */
    template<typename T>
    RosWebRTCBridge<T>::~RosWebRTCBridge()
    {
        ROS_INFO("ROS is shutting down, closing signaling client connection.");
        disconnect();
        ROS_INFO("Signaling client disconnected, goodbye.");
    }

    /**
     * @brief Connect the Signaling client
     */
    template<typename T>
    void RosWebRTCBridge<T>::connect()
    {
        connectSignalingClientEvents();
        if (m_signalingClient != nullptr)
        {
            ROS_INFO_STREAM(
                nodeName << " --> "
                         << "Connecting to signaling server...");
            m_signalingClient->connect();
        }
        else
        {
            ROS_ERROR_STREAM(
                nodeName << " --> "
                         << "Signaling client is nullptr, cannot connect.");
        }
    }

    /**
     * @brief Disconnect the Signaling client
     */
    template<typename T>
    void RosWebRTCBridge<T>::disconnect()
    {
        if (m_signalingClient != nullptr)
        {
            ROS_INFO_STREAM(
                nodeName << " --> "
                         << "Disconnecting from signaling server...");
            m_signalingClient->closeSync();

            // Reset client
            m_signalingClient = nullptr;
        }
        else
        {
            ROS_ERROR_STREAM(
                nodeName << " --> "
                         << "Signaling client is nullptr, cannot disconnect.");
        }
    }

    /**
     * @brief publish a Peer message using the given node publisher
     *
     * @param publisher ROS node publisher
     * @param client Client who sent the message
     * @param frame The message to peer with a client
     */
    template<typename T>
    template<typename PeerMsg>
    void RosWebRTCBridge<T>::publishPeerFrame(
        ros::Publisher& publisher,
        const Client& client,
        const decltype(PeerMsg::frame)& frame)
    {
        PeerMsg peerFrameMsg;
        peerFrameMsg.sender.id = client.id();
        peerFrameMsg.sender.name = client.name();
        peerFrameMsg.frame = frame;
        publisher.publish(peerFrameMsg);
    }

    template<typename T>
    void RosWebRTCBridge<T>::publishPeerStatus(const Client& client, int status)
    {
        opentera_webrtc_ros_msgs::PeerStatus msg;
        msg.sender.id = client.id();
        msg.sender.name = client.name();
        msg.status = status;
        m_peerStatusPublisher.publish(msg);
    }

    /**
     * @brief Connect to server and process images forever
     */
    template<typename T>
    void RosWebRTCBridge<T>::run()
    {
        ros::spin();
    }

    /************************************************************************
     * OPENTERA EVENTS
     ***********************************************************************/

    /**
     * @brief Callback called when receiving a message from the /events topic.
     *
     * @param event The message received.
     */
    template<typename T>
    void RosWebRTCBridge<T>::onEvent(const opentera_webrtc_ros_msgs::OpenTeraEventConstPtr& msg)
    {
        if (!msg->database_events.empty())
        {
            onDataBaseEvents(msg->database_events);
        }

        if (!msg->device_events.empty())
        {
            onDeviceEvents(msg->device_events);
        }

        if (!msg->join_session_events.empty())
        {
            for (auto i = 0; i < msg->join_session_events.size(); i++)
            {
                if (msg->join_session_events[i].session_devices.size() > 1)
                {
                    m_hasMultipleDevices = true;
                }
            }
            onJoinSessionEvents(msg->join_session_events);
        }

        if (!msg->join_session_reply_events.empty())
        {
            onJoinSessionReplyEvents(msg->join_session_reply_events);
        }

        if (!msg->leave_session_events.empty())
        {
            onLeaveSessionEvents(msg->leave_session_events);
        }

        if (!msg->log_events.empty())
        {
            onLogEvents(msg->log_events);
        }

        if (!msg->participant_events.empty())
        {
            onParticipantEvents(msg->participant_events);
        }

        if (!msg->stop_session_events.empty())
        {
            onStopSessionEvents(msg->stop_session_events);
        }

        if (!msg->user_events.empty())
        {
            onUserEvents(msg->user_events);
        }
    }

    /**
     * @brief Callback that is call when their's a database event received
     *
     * @param events A vector of database event message
     */
    template<typename T>
    void RosWebRTCBridge<T>::onDataBaseEvents(const std::vector<opentera_webrtc_ros_msgs::DatabaseEvent>& events)
    {
    }

    /**
     * @brief Callback that is call when their's a device event received
     *
     * @param events A vector of device event message
     */
    template<typename T>
    void RosWebRTCBridge<T>::onDeviceEvents(const std::vector<opentera_webrtc_ros_msgs::DeviceEvent>& events)
    {
    }

    /**
     * @brief Callback that is call when their's a join session event received
     *
     * @param events A vector of join session event message
     */
    template<typename T>
    void RosWebRTCBridge<T>::onJoinSessionEvents(const std::vector<opentera_webrtc_ros_msgs::JoinSessionEvent>& events)
    {
    }

    /**
     * @brief Callback that is call when their's a join session reply event received
     *
     * @param events A vector of join session reply event message
     */
    template<typename T>
    void RosWebRTCBridge<T>::onJoinSessionReplyEvents(
        const std::vector<opentera_webrtc_ros_msgs::JoinSessionReplyEvent>& events)
    {
    }

    /**
     * @brief Callback that is call when their's a leave session event received
     *
     * @param events A vector of leave session event message
     */
    template<typename T>
    void
        RosWebRTCBridge<T>::onLeaveSessionEvents(const std::vector<opentera_webrtc_ros_msgs::LeaveSessionEvent>& events)
    {
    }

    /**
     * @brief Callback that is call when their's a log event event received
     *
     * @param events A vector of log session event message
     */
    template<typename T>
    void RosWebRTCBridge<T>::onLogEvents(const std::vector<opentera_webrtc_ros_msgs::LogEvent>& events)
    {
    }

    /**
     * @brief Callback that is call when their's a participant event received
     *
     * @param events A vector of participant event message
     */
    template<typename T>
    void RosWebRTCBridge<T>::onParticipantEvents(const std::vector<opentera_webrtc_ros_msgs::ParticipantEvent>& events)
    {
    }

    /**
     * @brief Callback that is call when their's a stop session event received
     *
     * @param events A vector of stop session event message
     */
    template<typename T>
    void RosWebRTCBridge<T>::onStopSessionEvents(const std::vector<opentera_webrtc_ros_msgs::StopSessionEvent>& events)
    {
    }

    /**
     * @brief Callback that is call when their's a user event received
     *
     * @param events A vector of user event message
     */
    template<typename T>
    void RosWebRTCBridge<T>::onUserEvents(const std::vector<opentera_webrtc_ros_msgs::UserEvent>& events)
    {
    }

    /************************************************************************
     * SIGNALING CLIENT CALLBACK
     ***********************************************************************/

    /**
     * @brief Connect signaling client callback
     */
    template<typename T>
    void RosWebRTCBridge<T>::connectSignalingClientEvents()
    {
        if (m_signalingClient != nullptr)
        {
            m_signalingClient->setOnSignalingConnectionOpened(
                std::bind(&RosWebRTCBridge<T>::onSignalingConnectionOpened, this));
            m_signalingClient->setOnSignalingConnectionClosed(
                std::bind(&RosWebRTCBridge<T>::onSignalingConnectionClosed, this));
            m_signalingClient->setOnSignalingConnectionError(
                std::bind(&RosWebRTCBridge<T>::onSignalingConnectionError, this, std::placeholders::_1));

            m_signalingClient->setOnRoomClientsChanged(
                std::bind(&RosWebRTCBridge<T>::onRoomClientsChanged, this, std::placeholders::_1));

            m_signalingClient->setCallAcceptor(
                std::bind(&RosWebRTCBridge<T>::callAcceptor, this, std::placeholders::_1));
            m_signalingClient->setOnCallRejected(
                std::bind(&RosWebRTCBridge<T>::onCallRejected, this, std::placeholders::_1));

            m_signalingClient->setOnClientConnected(
                std::bind(&RosWebRTCBridge<T>::onClientConnected, this, std::placeholders::_1));
            m_signalingClient->setOnClientDisconnected(
                std::bind(&RosWebRTCBridge<T>::onClientDisconnected, this, std::placeholders::_1));

            m_signalingClient->setOnError(std::bind(&RosWebRTCBridge<T>::onError, this, std::placeholders::_1));
        }
    }

    /**
     * @brief Callback that is call when the signaling client is opened
     */
    template<typename T>
    void RosWebRTCBridge<T>::onSignalingConnectionOpened()
    {
        ROS_INFO_STREAM(
            nodeName << " --> "
                     << "Signaling connection opened, streaming topic...");
    }

    /**
     * @brief Callback that is call when the signaling client is closed
     */
    template<typename T>
    void RosWebRTCBridge<T>::onSignalingConnectionClosed()
    {
        ROS_WARN_STREAM(
            nodeName << " --> "
                     << "Signaling connection closed.");
    }

    /**
     * @brief Callback that is call when their's an error with the signaling client
     *
     * @param msg The error message
     */
    template<typename T>
    void RosWebRTCBridge<T>::onSignalingConnectionError(const std::string& msg)
    {
        ROS_ERROR_STREAM(
            nodeName << " --> "
                     << "Signaling connection error " << msg.c_str() << ", shutting down...");
        ros::requestShutdown();
    }

    /**
     * @brief Callback that is call when their's a change in the room
     *
     * @param roomClients A vector of client in the room
     */
    template<typename T>
    void RosWebRTCBridge<T>::onRoomClientsChanged(const std::vector<RoomClient>& roomClients)
    {
        std::string log = nodeName + " --> Signaling on room clients changed:";
        bool clientNotConnected = false;
        for (const auto& client : roomClients)
        {
            log += "\n\tid: " + client.id() + ", name: " + client.name() +
                   ", isConnected: " + (client.isConnected() ? "true" : "false");
            if (!client.isConnected())
            {
                clientNotConnected = true;
            }
        }
        if (clientNotConnected && m_hasMultipleDevices)
        {
            m_signalingClient->callAll();
            m_hasMultipleDevices = false;
        }
        ROS_INFO_STREAM(log);
    }

    /**
     * @brief Callback that is call before a call
     *
     * @param client The client who sent the call
     */
    template<typename T>
    bool RosWebRTCBridge<T>::callAcceptor(const Client& client)
    {
        // TODO
        return true;
    }

    /**
     * @brief Callback that is call when a call his rejected
     *
     * @param client The client who rejected the call
     */
    template<typename T>
    void RosWebRTCBridge<T>::onCallRejected(const Client& client)
    {
        // TODO
    }

    /**
     * @brief Callback that is call when a client his connected
     *
     * @param client The client who his connected
     */
    template<typename T>
    void RosWebRTCBridge<T>::onClientConnected(const Client& client)
    {
        publishPeerStatus(client, opentera_webrtc_ros_msgs::PeerStatus::STATUS_CLIENT_CONNECTED);
        ROS_INFO_STREAM(
            nodeName << " --> "
                     << "Signaling on client connected: "
                     << "id: " << client.id() << ", name: " << client.name());
    }

    /**
     * @brief Callback that is call when a client his disconnected
     *
     * @param client The client who his diconnected
     */
    template<typename T>
    void RosWebRTCBridge<T>::onClientDisconnected(const Client& client)
    {
        publishPeerStatus(client, opentera_webrtc_ros_msgs::PeerStatus::STATUS_CLIENT_DISCONNECTED);
        ROS_INFO_STREAM(
            nodeName << " --> "
                     << "Signaling on client disconnected: "
                     << "id: " << client.id() << ", name: " << client.name());
    }

    /**
     * @brief Callback that is call when their's an error with the signaling client
     *
     * @param error The error message
     */
    template<typename T>
    void RosWebRTCBridge<T>::onError(const std::string& error)
    {
        ROS_ERROR_STREAM(nodeName << " --> " << error);
    }
}

#endif
