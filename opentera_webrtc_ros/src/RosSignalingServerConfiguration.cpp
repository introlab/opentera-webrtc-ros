#include <rclcpp/rclcpp.hpp>
#include <opentera_webrtc_ros/RosSignalingServerConfiguration.h>
#include <opentera_webrtc_ros/RosStreamBridge.h>
#include <opentera_webrtc_ros/RosNodeParameters.h>

using namespace opentera;
using namespace std;
using namespace rclcpp;

std::string
    replaceProtocol(const std::string& url, const char* from0, const char* to0, const char* from1, const char* to1)
{
    size_t p = url.find("://");
    if (p == std::string::npos)
    {
        return url;
    }

    std::string protocol = url.substr(0, p);
    transform(protocol.begin(), protocol.end(), protocol.begin(), ptr_fun<int, int>(tolower));
    if (protocol == from0)
    {
        return to0 + url.substr(p);
    }
    else if (protocol == from1)
    {
        return to1 + url.substr(p);
    }
    else
    {
        return url;
    }
}

std::string httpToWs(const std::string& url)
{
    return replaceProtocol(url, "http", "ws", "https", "wss");
}

std::string wsToHttp(const std::string& url)
{
    return replaceProtocol(url, "ws", "http", "wss", "https");
}

/**
 * @brief Build a signaling server configuration from the ROS parameter server
 * Parameters are retrieved from the signaling namespace under the node private namespace
 *
 * @param defaultClientName Default name for the webrtc peer
 * @return The signaling server configuration
 */
SignalingServerConfiguration RosSignalingServerConfiguration::fromRosParam(RosNodeParameters& nodeParameters)
{
    string serverUrl, clientName, room, password;
    nodeParameters.loadSignalingParams(serverUrl, clientName, room, password);

    return SignalingServerConfiguration::create(httpToWs(serverUrl) + "/signaling", clientName, room, password);
}

/**
 * @brief Build a signaling server configuration from the given Url.
 * Parameters are retrieved from the queries and from the signaling namespace under the node private namespace.
 *
 * @param url An url to parse.
 *
 * @return The signaling server configuration.
 */
SignalingServerConfiguration
    RosSignalingServerConfiguration::fromUrl(RosNodeParameters& nodeParameters, const std::string& url)
{
    string address = httpToWs(RosSignalingServerConfiguration::getBaseUrl(nodeParameters, url)) + "/signaling";

    size_t pos1 = url.find("?");
    string queries = url.substr(pos1);
    string password = getQueryFrom("pwd", queries);
    string clientName, roomName;

    nodeParameters.loadSignalingParams(clientName, roomName);

    return SignalingServerConfiguration::create(address, clientName, roomName, password);
}


/**
 * @brief Get a query value among the queries arguments.
 * Queries format: "?query1=123&query2=abc"
 *
 * @param query The parameters that contain de value.
 * @param queries The full queries in which the query argument his.
 *
 * @return The value of the query
 */
std::string RosSignalingServerConfiguration::getQueryFrom(const std::string& query, const std::string& queries)
{
    size_t pos1, pos2;

    pos1 = queries.find(query + "=");
    string remaining = queries.substr(pos1);
    pos2 = remaining.find_first_of("&");

    return remaining.substr(0, pos2).substr(query.size() + 1);
}

/**
 * @brief Get the ice servers url from url
 *
 * @param url The full url to extract from.
 * @return std::string The ice server url.
 */
std::string RosSignalingServerConfiguration::getIceServerUrl(RosNodeParameters& nodeParameters, const std::string& url)
{
    RCLCPP_INFO_STREAM(nodeParameters.node().get_logger(), "getIceServerUrl from url:" << url);
    return wsToHttp(RosSignalingServerConfiguration::getBaseUrl(nodeParameters, url)) + "/iceservers";
}

/**
 * @brief Get base url from url
 *
 * @param url The full url to extract from.
 * @return std::string The base url (up to the last / of the path).
 */
std::string RosSignalingServerConfiguration::getBaseUrl(RosNodeParameters& nodeParameters, const std::string& url)
{
    const string prot_end("://");
    string::const_iterator prot_i = search(url.begin(), url.end(), prot_end.begin(), prot_end.end());

    std::string protocol;
    protocol.reserve(distance(url.begin(), prot_i));


    transform(url.begin(), prot_i, back_inserter(protocol), ptr_fun<int, int>(tolower));  // protocol is icase

    if (prot_i == url.end())
    {
        RCLCPP_ERROR_STREAM(nodeParameters.node().get_logger(), "No protocol defined in url: " << url);
        return url;
    }

    advance(prot_i, prot_end.length());

    string::const_iterator path_i = std::find(prot_i, url.end(), '/');

    auto pos = url.find_last_of('/');
    string::const_iterator last_i = url.begin();
    advance(last_i, pos);

    if (path_i < last_i)
    {
        path_i = last_i;
    }

    std::string host;
    host.reserve(distance(prot_i, path_i));
    transform(prot_i, path_i, back_inserter(host), ptr_fun<int, int>(tolower));  // host is icase

    return protocol + prot_end + host;
}
