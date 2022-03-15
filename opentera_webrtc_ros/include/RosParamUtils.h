#ifndef OPENTERA_WEBRTC_NATIVE_CLIENT_SOURCES_ROS_PARAM_UTILS_H
#define OPENTERA_WEBRTC_NATIVE_CLIENT_SOURCES_ROS_PARAM_UTILS_H

#include <ros/ros.h>

namespace opentera
{
    namespace param
    {
        /** \brief Get a {double, float, int, bool, string} map value from the parameter
         * server.
         *
         * Values of a type not compatible with the map are ignored.
         *
         * \param key The key to be used in the parameter server's dictionary
         * \param[out] map Storage for the retrieved value.
         *
         * \return true if the parameter value was retrieved, false otherwise
         * \throws InvalidNameException If the parameter key begins with a tilde, or is an
         * otherwise invalid graph resource name
         */
        bool getParam(const ros::NodeHandle& nh, const std::string& key,
                      std::map<std::string, double>& map);
        bool getParam(const ros::NodeHandle& nh, const std::string& key,
                      std::map<std::string, float>& map);
        bool getParam(const ros::NodeHandle& nh, const std::string& key,
                      std::map<std::string, int>& map);
        bool getParam(const ros::NodeHandle& nh, const std::string& key,
                      std::map<std::string, bool>& map);
        bool getParam(const ros::NodeHandle& nh, const std::string& key,
                      std::map<std::string, std::string>& map);

        /** \brief Get a {double, float, int, bool, string} map value from the parameter
         * server using the cache.
         *
         * Values of a type not compatible with the map are ignored.
         *
         * \param key The key to be used in the parameter server's dictionary
         * \param[out] map Storage for the retrieved value.
         *
         * \return true if the parameter value was retrieved, false otherwise
         * \throws InvalidNameException If the parameter key begins with a tilde, or is an
         * otherwise invalid graph resource name
         */
        bool getParamCached(const ros::NodeHandle& nh, const std::string& key,
                            std::map<std::string, double>& map);
        bool getParamCached(const ros::NodeHandle& nh, const std::string& key,
                            std::map<std::string, float>& map);
        bool getParamCached(const ros::NodeHandle& nh, const std::string& key,
                            std::map<std::string, int>& map);
        bool getParamCached(const ros::NodeHandle& nh, const std::string& key,
                            std::map<std::string, bool>& map);
        bool getParamCached(const ros::NodeHandle& nh, const std::string& key,
                            std::map<std::string, std::string>& map);

        /** \brief Get a {double, float, int, bool, string} vector value from the
         * parameter server.
         *
         * Values of a type not compatible with the vector are ignored.
         *
         * \param key The key to be used in the parameter server's dictionary
         * \param[out] vec Storage for the retrieved value.
         *
         * \return true if the parameter value was retrieved, false otherwise
         * \throws InvalidNameException If the parameter key begins with a tilde, or is an
         * otherwise invalid graph resource name
         */
        bool getParam(const ros::NodeHandle& nh, const std::string& key,
                      std::vector<double>& vec);
        bool getParam(const ros::NodeHandle& nh, const std::string& key,
                      std::vector<float>& vec);
        bool getParam(const ros::NodeHandle& nh, const std::string& key,
                      std::vector<int>& vec);
        bool getParam(const ros::NodeHandle& nh, const std::string& key,
                      std::vector<bool>& vec);
        bool getParam(const ros::NodeHandle& nh, const std::string& key,
                      std::vector<std::string>& vec);

        /** \brief Get a {double, float, int, bool, string} vector value from the
         * parameter server using the cache.
         *
         * Values of a type not compatible with the vector are ignored.
         *
         * \param key The key to be used in the parameter server's dictionary
         * \param[out] vec Storage for the retrieved value.
         *
         * \return true if the parameter value was retrieved, false otherwise
         * \throws InvalidNameException If the parameter key begins with a tilde, or is an
         * otherwise invalid graph resource name
         */
        bool getParamCached(const ros::NodeHandle& nh, const std::string& key,
                            std::vector<double>& vec);
        bool getParamCached(const ros::NodeHandle& nh, const std::string& key,
                            std::vector<float>& vec);
        bool getParamCached(const ros::NodeHandle& nh, const std::string& key,
                            std::vector<int>& vec);
        bool getParamCached(const ros::NodeHandle& nh, const std::string& key,
                            std::vector<bool>& vec);
        bool getParamCached(const ros::NodeHandle& nh, const std::string& key,
                            std::vector<std::string>& vec);
    }
}

#endif
