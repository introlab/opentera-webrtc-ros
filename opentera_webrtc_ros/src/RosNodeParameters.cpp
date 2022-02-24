#include <ros/ros.h>
#include <RosNodeParameters.h>
#include <RosParamUtils.h>

using namespace opentera;
using namespace ros;
using namespace std;

bool RosNodeParameters::isStandAlone()
{
    NodeHandle pnh("~");
    return pnh.param<bool>("is_stand_alone", true);
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
    opentera::param::getParam(pnh, "signaling", dict);

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
void RosNodeParameters::loadSignalingParams(std::string &serverUrl, std::string &clientName,
    std::string &room, std::string &password)
{
    NodeHandle pnh("~");

    //String parameters
    std::map<std::string, std::string> dictString;
    opentera::param::getParam(pnh, "signaling", dictString);
    serverUrl = isInParams("server_url", dictString) ? dictString["server_url"] : "http://localhost:8080";
    clientName = isInParams("client_name", dictString) ? dictString["client_name"] : "streamer";
    room = isInParams("room_name", dictString) ? dictString["room_name"] : "chat";
    password = isInParams("room_password", dictString) ? dictString["room_password"] : "abc";
}

/**
 * @brief Load signaling parameters (verify_ssl) from ROS parameter server
 *
 * @param verifySSL Verify SSL peer
 */
void RosNodeParameters::loadSignalingParamsVerifySSL(bool &verifySSL)
{
    NodeHandle pnh("~");
    //Bool parameters
    std::map <std::string, bool> dictBool;
    opentera::param::getParam(pnh, "signaling", dictBool);
    verifySSL = isInParams("verify_ssl", dictBool) ? dictBool["verify_ssl"] : true;
}

/**
 * @brief Load video stream parameters from ROS parameter server
 *
 * @param canSendVideoStream whether the node can send video stream to the signaling server
 * @param canReceiveVideoStream whether the node can received video stream from the signaling server
 * @param denoise whether the images require denoising
 * @param screencast whether the images are a screen capture
 */
void RosNodeParameters::loadVideoStreamParams(
            bool &canSendVideoStream,
            bool &canReceiveVideoStream,
            bool &denoise,
            bool &screencast)
{
    NodeHandle pnh("~");

    std::map<std::string, bool> dict;
    opentera::param::getParam(pnh, "videoStream", dict);

    canSendVideoStream = isInParams("can_send_video_stream", dict) ? dict["can_send_video_stream"] : false;
    canReceiveVideoStream = isInParams("can_receive_video_stream", dict) ? dict["can_receive_video_stream"] : false;
    denoise = isInParams("needs_denoising", dict) ? dict["needs_denoising"] : false;
    screencast = isInParams("is_screen_cast", dict) ? dict["is_screen_cast"] : false;
}

/**
 * @brief Load audio stream parameters from ROS parameter server
 *
 * @param canSendAudioStream whether the node can send audio stream to the signaling server
 * @param canReceiveAudioStream whether the node can received audio stream from the signaling server
 */
void RosNodeParameters::loadAudioStreamParams(
            bool &canSendAudioStream,
            bool &canReceiveAudioStream,
            unsigned int &soundCardTotalDelayMs,
            bool &echoCancellation,
            bool &autoGainControl,
            bool &noiseSuppression,
            bool &highPassFilter,
            bool &stereoSwapping,
            bool &typingDetection,
            bool &residualEchoDetector,
            bool &transientSuppression)
{
    NodeHandle pnh("~");

    std::map<std::string, bool> dict;
    opentera::param::getParam(pnh, "audioStream", dict);

    std::map<std::string, int> dictInt;
    opentera::param::getParam(pnh, "audioStream", dictInt);


    canSendAudioStream = isInParams("can_send_audio_stream", dict) ? dict["can_send_audio_stream"] : false;
    canReceiveAudioStream = isInParams("can_receive_audio_stream", dict) ? dict["can_receive_audio_stream"] : false;
    soundCardTotalDelayMs= isInParams("sound_card_total_delay_ms", dictInt) ? dictInt["sound_card_total_delay_ms"] : 40;
    echoCancellation= isInParams("echo_cancellation", dict) ? dict["echo_cancellation"] : true;
    autoGainControl= isInParams("auto_gain_control", dict) ? dict["auto_gain_control"] : true;
    noiseSuppression= isInParams("noise_suppression", dict) ? dict["noise_suppression"] : true;
    highPassFilter= isInParams("high_pass_filter", dict) ? dict["high_pass_filter"] : false;
    stereoSwapping= isInParams("stereo_swapping", dict) ? dict["stereo_swapping"] : false;
    typingDetection= isInParams("typing_detection", dict) ? dict["typing_detection"] : false;
    residualEchoDetector= isInParams("residual_echo_detector", dict) ? dict["residual_echo_detector"] : true;
    transientSuppression= isInParams("transient_suppression", dict) ? dict["transient_suppression"] : true;
}