#include <ros/ros.h>
#include <RosNodeParameters.h>

using namespace opentera;
using namespace ros;
using namespace std;

bool RosNodeParameters::isStandAlone() 
{
    NodeHandle pnh("~");
    return pnh.param<bool>("is_stand_alone", true);
}

/**
 * @brief Load stream parameters from ROS parameter server
 *
 * @param canSendAudioStream whether the node can send audio stream to the signaling server
 * @param canReceiveAudioStream whether the node can received audio stream from the signaling server
 * @param canSendVideoStream whether the node can send video stream to the signaling server
 * @param canReceiveVideoStream whether the node can received video stream from the signaling server
 * @param denoise whether the images require denoising
 * @param screencast whether the images are a screen capture
 */
void RosNodeParameters::loadStreamParams(bool &canSendAudioStream, bool &canReceiveAudioStream, 
    bool &canSendVideoStream, bool &canReceiveVideoStream, bool &denoise, bool &screencast)
{
    NodeHandle pnh("~");

    std::map<std::string, bool> dict;
    pnh.getParam("stream", dict);

    canSendAudioStream = isInParams("can_send_audio_stream", dict) ? dict["can_send_audio_stream"] : false;
    canReceiveAudioStream = isInParams("can_receive_audio_stream", dict) ? dict["can_receive_audio_stream"] : false;
    canSendVideoStream = isInParams("can_send_video_stream", dict) ? dict["can_send_video_stream"] : false;
    canReceiveVideoStream = isInParams("can_receive_video_stream", dict) ? dict["can_receive_video_stream"] : false;
    denoise = isInParams("needs_denoising", dict) ? dict["needs_denoising"] : false;
    screencast = isInParams("is_screen_cast", dict) ? dict["is_screen_cast"] : false;
}

/**
 * @brief Load signaling parameters from ROS parameter server
 *
 * @param clientName Client's name
 * @param room Room's name
 */
void RosNodeParameters::loadSignalingParams(std::string &clientName, std::string &room) 
{
    NodeHandle pnh("~");

    std::map<std::string, std::string> dict;
    pnh.getParam("signaling", dict);

    clientName = isInParams("client_name", dict) ? dict["client_name"] : "streamer";
    room = isInParams("room_name", dict) ? dict["room_name"] : "chat";
}

/**
 * @brief Load signaling parameters from ROS parameter server
 *
 * @param serverUrl Signaling server Url
 * @param clientName Client's name
 * @param room Room's name
 * @param password Room's password
 */
void RosNodeParameters::loadSignalingParams(std::string &serverUrl, std::string &clientName, std::string &room, std::string &password)
{
    NodeHandle pnh("~");

    std::map<std::string, std::string> dict;
    pnh.getParam("signaling", dict);

    serverUrl = isInParams("server_url", dict) ? dict["server_url"] : "http://localhost:8080";
    clientName = isInParams("client_name", dict) ? dict["client_name"] : "streamer";
    room = isInParams("room_name", dict) ? dict["room_name"] : "chat";
    password = isInParams("room_password", dict) ? dict["room_password"] : "abc";
}
