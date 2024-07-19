#ifndef OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_WEBRTC_BRIDGE_H
#define OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_WEBRTC_BRIDGE_H

#include <rclcpp/rclcpp.hpp>
#include <signal.h>

#include <OpenteraWebrtcNativeClient/Configurations/SignalingServerConfiguration.h>
#include <OpenteraWebrtcNativeClient/Utils/Client.h>
#include <OpenteraWebrtcNativeClient/WebrtcClient.h>

#include <opentera_webrtc_ros_msgs/msg/open_tera_event.hpp>
#include <opentera_webrtc_ros_msgs/msg/database_event.hpp>
#include <opentera_webrtc_ros_msgs/msg/device_event.hpp>
#include <opentera_webrtc_ros_msgs/msg/join_session_event.hpp>
#include <opentera_webrtc_ros_msgs/msg/join_session_reply_event.hpp>
#include <opentera_webrtc_ros_msgs/msg/leave_session_event.hpp>
#include <opentera_webrtc_ros_msgs/msg/log_event.hpp>
#include <opentera_webrtc_ros_msgs/msg/participant_event.hpp>
#include <opentera_webrtc_ros_msgs/msg/stop_session_event.hpp>
#include <opentera_webrtc_ros_msgs/msg/user_event.hpp>
#include <opentera_webrtc_ros_msgs/msg/peer_status.hpp>

#include <opentera_webrtc_ros/RosNodeParameters.h>
#include <opentera_webrtc_ros/RosSignalingServerConfiguration.h>
#include <opentera_webrtc_ros/utils.h>

namespace opentera
{

    /**
     * @brief Interface a ROS node to bridge WebRTC to ROS topics
     */
    template<typename T>
    class RosWebRTCBridge : public rclcpp::Node
    {
        static_assert(std::is_base_of<WebrtcClient, T>::value, "T must inherit from opentera::WebrtcClient");

    private:
        rclcpp::Subscription<opentera_webrtc_ros_msgs::msg::OpenTeraEvent>::SharedPtr m_eventSubscriber;

    protected:
        RosNodeParameters m_nodeParameters;
        rclcpp::Publisher<opentera_webrtc_ros_msgs::msg::PeerStatus>::SharedPtr m_peerStatusPublisher;
        std::unique_ptr<T> m_signalingClient;

        virtual void connect();
        virtual void disconnect();

        virtual void onEvent(const opentera_webrtc_ros_msgs::msg::OpenTeraEvent::ConstSharedPtr& msg);
        virtual void onDataBaseEvents(const std::vector<opentera_webrtc_ros_msgs::msg::DatabaseEvent>& events);
        virtual void onDeviceEvents(const std::vector<opentera_webrtc_ros_msgs::msg::DeviceEvent>& events);
        virtual void onJoinSessionEvents(const std::vector<opentera_webrtc_ros_msgs::msg::JoinSessionEvent>& events);
        virtual void
            onJoinSessionReplyEvents(const std::vector<opentera_webrtc_ros_msgs::msg::JoinSessionReplyEvent>& events);
        virtual void onLeaveSessionEvents(const std::vector<opentera_webrtc_ros_msgs::msg::LeaveSessionEvent>& events);
        virtual void onLogEvents(const std::vector<opentera_webrtc_ros_msgs::msg::LogEvent>& events);
        virtual void onParticipantEvents(const std::vector<opentera_webrtc_ros_msgs::msg::ParticipantEvent>& events);
        virtual void onStopSessionEvents(const std::vector<opentera_webrtc_ros_msgs::msg::StopSessionEvent>& events);
        virtual void onUserEvents(const std::vector<opentera_webrtc_ros_msgs::msg::UserEvent>& events);

        virtual void connectSignalingClientEvents();
        virtual void onSignalingConnectionOpened();
        virtual void onSignalingConnectionClosed();
        virtual void onSignalingConnectionError(const std::string& msg);
        virtual void onRoomClientsChanged(const std::vector<RoomClient>& roomClients);
        virtual bool callAcceptor(const Client& client);
        virtual void onCallRejected(const Client& client);
        virtual void onClientConnected(const Client& client);
        virtual void onClientDisconnected(const Client& client);
        virtual void onClientConnectionFailed(const Client& client);
        virtual void onError(const std::string& error);

        template<typename PeerMsg>
        void publishPeerFrame(
            rclcpp::Publisher<PeerMsg>& publisher,
            const Client& client,
            const decltype(PeerMsg::frame)& frame);
        void publishPeerStatus(const Client& client, int status);

    public:
        explicit RosWebRTCBridge(const std::string& nodeName);
        virtual ~RosWebRTCBridge() = 0;

        void run();
    };

    /**
     * @brief construct a topic streamer node
     *
     * @param nh node handle.
     */
    template<typename T>
    RosWebRTCBridge<T>::RosWebRTCBridge(const std::string& nodeName)
        : rclcpp::Node(nodeName),
          m_nodeParameters(*this),
          m_signalingClient(nullptr)
    {
        // On CTRL+C exit
        signal(
            SIGINT,
            [](int sig)
            {
                (void)sig;
                rclcpp::shutdown();
            });

        if (!m_nodeParameters.isStandAlone())
        {
            m_eventSubscriber = this->create_subscription<opentera_webrtc_ros_msgs::msg::OpenTeraEvent>(
                "events",
                10,
                bind_this<opentera_webrtc_ros_msgs::msg::OpenTeraEvent>(this, &RosWebRTCBridge::onEvent));
        }

        m_peerStatusPublisher =
            this->create_publisher<opentera_webrtc_ros_msgs::msg::PeerStatus>("webrtc_peer_status", 10);
    }

    /**
     * @brief Close signaling client connection when this object is destroyed
     */
    template<typename T>
    RosWebRTCBridge<T>::~RosWebRTCBridge()
    {
        RCLCPP_INFO(this->get_logger(), "ROS is shutting down, closing signaling client connection.");
        disconnect();
        RCLCPP_INFO(this->get_logger(), "Signaling client disconnected, goodbye.");
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
            RCLCPP_INFO(this->get_logger(), " --> Connecting to signaling server...");
            m_signalingClient->connect();
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), " --> Signaling client is nullptr, cannot connect.");
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
            RCLCPP_INFO(this->get_logger(), " --> Disconnecting from signaling server...");
            m_signalingClient->closeSync();

            // Reset client
            m_signalingClient = nullptr;
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), " --> Signaling client is nullptr, cannot disconnect.");
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
        rclcpp::Publisher<PeerMsg>& publisher,
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
        opentera_webrtc_ros_msgs::msg::PeerStatus msg;
        msg.sender.id = client.id();
        msg.sender.name = client.name();
        msg.status = status;
        m_peerStatusPublisher->publish(msg);
    }

    /**
     * @brief Connect to server and process images forever
     */
    template<typename T>
    void RosWebRTCBridge<T>::run()
    {
        rclcpp::spin(this->shared_from_this());
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
    void RosWebRTCBridge<T>::onEvent(const opentera_webrtc_ros_msgs::msg::OpenTeraEvent::ConstSharedPtr& msg)
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
    void RosWebRTCBridge<T>::onDataBaseEvents(const std::vector<opentera_webrtc_ros_msgs::msg::DatabaseEvent>& events)
    {
        (void)events;
    }

    /**
     * @brief Callback that is call when their's a device event received
     *
     * @param events A vector of device event message
     */
    template<typename T>
    void RosWebRTCBridge<T>::onDeviceEvents(const std::vector<opentera_webrtc_ros_msgs::msg::DeviceEvent>& events)
    {
        (void)events;
    }

    /**
     * @brief Callback that is call when their's a join session event received
     *
     * @param events A vector of join session event message
     */
    template<typename T>
    void RosWebRTCBridge<T>::onJoinSessionEvents(
        const std::vector<opentera_webrtc_ros_msgs::msg::JoinSessionEvent>& events)
    {
        (void)events;
    }

    /**
     * @brief Callback that is call when their's a join session reply event received
     *
     * @param events A vector of join session reply event message
     */
    template<typename T>
    void RosWebRTCBridge<T>::onJoinSessionReplyEvents(
        const std::vector<opentera_webrtc_ros_msgs::msg::JoinSessionReplyEvent>& events)
    {
        (void)events;
    }

    /**
     * @brief Callback that is call when their's a leave session event received
     *
     * @param events A vector of leave session event message
     */
    template<typename T>
    void RosWebRTCBridge<T>::onLeaveSessionEvents(
        const std::vector<opentera_webrtc_ros_msgs::msg::LeaveSessionEvent>& events)
    {
        (void)events;
    }

    /**
     * @brief Callback that is call when their's a log event event received
     *
     * @param events A vector of log session event message
     */
    template<typename T>
    void RosWebRTCBridge<T>::onLogEvents(const std::vector<opentera_webrtc_ros_msgs::msg::LogEvent>& events)
    {
        (void)events;
    }

    /**
     * @brief Callback that is call when their's a participant event received
     *
     * @param events A vector of participant event message
     */
    template<typename T>
    void RosWebRTCBridge<T>::onParticipantEvents(
        const std::vector<opentera_webrtc_ros_msgs::msg::ParticipantEvent>& events)
    {
        (void)events;
    }

    /**
     * @brief Callback that is call when their's a stop session event received
     *
     * @param events A vector of stop session event message
     */
    template<typename T>
    void RosWebRTCBridge<T>::onStopSessionEvents(
        const std::vector<opentera_webrtc_ros_msgs::msg::StopSessionEvent>& events)
    {
        (void)events;
    }

    /**
     * @brief Callback that is call when their's a user event received
     *
     * @param events A vector of user event message
     */
    template<typename T>
    void RosWebRTCBridge<T>::onUserEvents(const std::vector<opentera_webrtc_ros_msgs::msg::UserEvent>& events)
    {
        (void)events;
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
            m_signalingClient->setOnClientConnectionFailed(
                std::bind(&RosWebRTCBridge<T>::onClientConnectionFailed, this, std::placeholders::_1));

            m_signalingClient->setOnError(std::bind(&RosWebRTCBridge<T>::onError, this, std::placeholders::_1));
        }
    }

    /**
     * @brief Callback that is called when the signaling client is opened
     */
    template<typename T>
    void RosWebRTCBridge<T>::onSignalingConnectionOpened()
    {
        RCLCPP_INFO(this->get_logger(), " --> Signaling connection opened, streaming topic...");
    }

    /**
     * @brief Callback that is called when the signaling client is closed
     */
    template<typename T>
    void RosWebRTCBridge<T>::onSignalingConnectionClosed()
    {
        RCLCPP_WARN(this->get_logger(), " --> Signaling connection closed.");
    }

    /**
     * @brief Callback that is called when there is an error with the signaling client
     *
     * @param msg The error message
     */
    template<typename T>
    void RosWebRTCBridge<T>::onSignalingConnectionError(const std::string& msg)
    {
        RCLCPP_ERROR_STREAM(
            this->get_logger(),
            " --> Signaling connection error " << msg.c_str() << ", shutting down...");
        rclcpp::shutdown();
    }

    /**
     * @brief Callback that is called when there is a change in the room
     *
     * @param roomClients A vector of clients in the room
     */
    template<typename T>
    void RosWebRTCBridge<T>::onRoomClientsChanged(const std::vector<RoomClient>& roomClients)
    {
        RCLCPP_INFO(this->get_logger(), " --> Signaling on room clients changed:\n");
        bool allClientsConnected = true;
        for (const auto& client : roomClients)
        {
            RCLCPP_INFO_STREAM(
                this->get_logger(),
                "\tid: " << client.id() << ", name: " << client.name() << ", isConnected: " << std::boolalpha
                         << client.isConnected() << "\n");
            allClientsConnected &= client.isConnected();
        }
        if (!allClientsConnected)
        {
            m_signalingClient->callAll();
        }
    }

    /**
     * @brief Callback that is called before a call
     *
     * @param client The client who sent the call
     */
    template<typename T>
    bool RosWebRTCBridge<T>::callAcceptor(const Client& client)
    {
        (void)client;
        // TODO
        return true;
    }

    /**
     * @brief Callback that is called when a call is rejected
     *
     * @param client The client who rejected the call
     */
    template<typename T>
    void RosWebRTCBridge<T>::onCallRejected(const Client& client)
    {
        (void)client;
        // TODO
    }

    /**
     * @brief Callback that is called when a client is connected
     *
     * @param client The client who is connected
     */
    template<typename T>
    void RosWebRTCBridge<T>::onClientConnected(const Client& client)
    {
        publishPeerStatus(client, opentera_webrtc_ros_msgs::msg::PeerStatus::STATUS_CLIENT_CONNECTED);
        RCLCPP_INFO_STREAM(
            this->get_logger(),
            " --> Signaling on client connected: " << "id: " << client.id() << ", name: " << client.name());
    }

    /**
     * @brief Callback that is called when a client is disconnected
     *
     * @param client The client who is diconnected
     */
    template<typename T>
    void RosWebRTCBridge<T>::onClientDisconnected(const Client& client)
    {
        publishPeerStatus(client, opentera_webrtc_ros_msgs::msg::PeerStatus::STATUS_CLIENT_DISCONNECTED);
        RCLCPP_INFO_STREAM(
            this->get_logger(),
            " --> Signaling on client disconnected: " << "id: " << client.id() << ", name: " << client.name());
    }

    /**
     * @brief Callback that is called when a client is disconnected
     *
     * @param client The client who is diconnected
     */
    template<typename T>
    void RosWebRTCBridge<T>::onClientConnectionFailed(const Client& client)
    {
        publishPeerStatus(client, opentera_webrtc_ros_msgs::msg::PeerStatus::STATUS_CLIENT_CONNECTION_FAILED);
        RCLCPP_WARN_STREAM(
            this->get_logger(),
            " --> Signaling on client connection failed: " << "id: " << client.id() << ", name: " << client.name());
    }

    /**
     * @brief Callback that is called when there is an error with the signaling client
     *
     * @param error The error message
     */
    template<typename T>
    void RosWebRTCBridge<T>::onError(const std::string& error)
    {
        RCLCPP_ERROR_STREAM(this->get_logger(), " --> " << error);
    }
}

#endif
