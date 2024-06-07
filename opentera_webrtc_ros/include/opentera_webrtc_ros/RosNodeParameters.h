#ifndef OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_NODE_PARAMETERS_H
#define OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_NODE_PARAMETERS_H

#include <OpenteraWebrtcNativeClient/Configurations/VideoStreamConfiguration.h>

#include <rclcpp/rclcpp.hpp>

#include <unordered_set>

namespace opentera
{
    /**
     * @brief Utility to load ROS parameter
     */
    class RosNodeParameters
    {
    public:
        RosNodeParameters(rclcpp::Node& node);

        bool isStandAlone() const;

        void loadSignalingParams(std::string& clientName, std::string& room) const;
        void loadSignalingParams(
            std::string& serverUrl,
            std::string& clientName,
            std::string& room,
            std::string& password) const;

        void loadSignalingParamsVerifySSL(bool& verifySSL) const;

        void loadVideoStreamParams(
            bool& canSendVideoStream,
            bool& canReceiveVideoStream,
            bool& denoise,
            bool& screencast) const;

        void loadVideoCodecParams(
            std::unordered_set<opentera::VideoStreamCodec>& forcedCodecs,
            bool& forceGStreamerHardwareAcceleration,
            bool& useGStreamerSoftwareEncoderDecoder) const;

        void loadAudioStreamParams(
            bool& canSendAudioStream,
            bool& canReceiveAudioStream,
            unsigned int& soundCardTotalDelayMs,
            bool& echoCancellation,
            bool& autoGainControl,
            bool& noiseSuppression,
            bool& highPassFilter,
            bool& stereoSwapping,
            bool& transientSuppression) const;

        const rclcpp::Node& node() const {return m_node; }
        rclcpp::Node& node() {return m_node; }

    private:
        rclcpp::Node& m_node;
    };
}

#endif
