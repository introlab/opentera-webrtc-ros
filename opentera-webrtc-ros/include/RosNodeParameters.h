#ifndef OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_NODE_PARAMETERS_H
#define OPENTERA_WEBRTC_NATIVE_CLIENT_ROS_NODE_PARAMETERS_H

namespace opentera {

    /**
     * @brief Utility to load ROS parameter
     */
    class RosNodeParameters
    {

    public:
        template <typename T>
        static bool isInParams(const std::string &key, const std::map<std::string, T> &dict);

        static void loadStreamParams(bool &canSendStream,
            bool &canReceiveStream,
            bool &denoise,
            bool &screencast);

        static void loadSignalingParams(std::string &serverUrl,
            std::string &clientName,
            std::string &room,
            std::string &password);
    };

    /**
     * @brief Check if the key his in the parameters
     * 
     * @param key Key value of the element to search
     * @param dict Dictionary (std::map) to search in
     * @return true if found otherwise false
     */
    template <typename T>
    bool RosNodeParameters::isInParams(const std::string &key, const std::map<std::string, T> &dict)
    {
        return dict.find(key) != dict.end();
    }
}

#endif