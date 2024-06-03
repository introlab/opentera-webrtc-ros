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
        // template<typename T>
        // static bool isInParams(rclcpp::Node& node, const std::string& key, const std::map<std::string, T>& dict);

        static bool isStandAlone(rclcpp::Node& node);

        static void loadSignalingParams(rclcpp::Node& node, std::string& clientName, std::string& room);
        static void loadSignalingParams(
            rclcpp::Node& node,
            std::string& serverUrl,
            std::string& clientName,
            std::string& room,
            std::string& password);

        static void loadSignalingParamsVerifySSL(rclcpp::Node& node, bool& verifySSL);

        static void loadVideoStreamParams(
            rclcpp::Node& node,
            bool& canSendVideoStream,
            bool& canReceiveVideoStream,
            bool& denoise,
            bool& screencast);

        static void loadVideoCodecParams(
            rclcpp::Node& node,
            std::unordered_set<opentera::VideoStreamCodec>& forcedCodecs,
            bool& forceGStreamerHardwareAcceleration,
            bool& useGStreamerSoftwareEncoderDecoder);

        static void loadAudioStreamParams(
            rclcpp::Node& node,
            bool& canSendAudioStream,
            bool& canReceiveAudioStream,
            unsigned int& soundCardTotalDelayMs,
            bool& echoCancellation,
            bool& autoGainControl,
            bool& noiseSuppression,
            bool& highPassFilter,
            bool& stereoSwapping,
            bool& transientSuppression);
    };

    /**
     * @brief Check if the key his in the parameters
     *
     * @param key Key value of the element to search
     * @param dict Dictionary (std::map) to search in
     * @return true if found otherwise false
     */
    // template<typename T>
    // bool RosNodeParameters::isInParams(const std::string& key, const std::map<std::string, T>& dict)
    // {
    //     return dict.find(key) != dict.end();
    // }
}

#endif
