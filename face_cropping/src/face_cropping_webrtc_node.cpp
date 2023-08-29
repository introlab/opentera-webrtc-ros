#include "OpencvFaceDetector.h"
#include "TorchFaceDetector.h"
#include "FaceCropper.h"
#include "FaceCroppingNodeConfiguration.h"
#include "OpencvUtils.h"

#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/Image.h>
#include <opentera_webrtc_ros_msgs/PeerImage.h>
#include <opentera_webrtc_ros_msgs/PeerStatus.h>
#include <cv_bridge/cv_bridge.h>

#include <unordered_map>


class FaceCroppingWebRtcNode
{
    ros::NodeHandle m_nodeHandle;
    FaceCroppingNodeConfiguration m_configuration;

    ros::Subscriber m_inputImageSubscriber;
    ros::Subscriber m_peerStatusSubscriber;
    ros::Publisher m_outputImagePublisher;

    ros::Subscriber m_enableCroppingSubscriber;
    bool m_enabled;

    std::shared_ptr<FaceDetector> m_faceDetector;
    std::unordered_map<std::string, FaceCropper> m_faceCroppersByPeerId;
    cv_bridge::CvImage m_outputImage;

public:
    FaceCroppingWebRtcNode(FaceCroppingNodeConfiguration configuration)
        : m_configuration(std::move(configuration)),
          m_enabled(true),
          m_faceDetector(createFaceDetector(m_configuration.faceDetectionModel, m_configuration.useGpuIfAvailable))
    {
        m_inputImageSubscriber =
            m_nodeHandle.subscribe("input_image", 10, &FaceCroppingWebRtcNode::inputImageCallback, this);
        m_peerStatusSubscriber =
            m_nodeHandle.subscribe("webrtc_peer_status", 10, &FaceCroppingWebRtcNode::peerStatusCallback, this);
        m_outputImagePublisher = m_nodeHandle.advertise<opentera_webrtc_ros_msgs::PeerImage>("output_image", 10);

        m_enableCroppingSubscriber =
            m_nodeHandle.subscribe("enable_face_cropping", 10, &FaceCroppingWebRtcNode::enableFaceCroppingCallback, this);

        m_outputImage.encoding = sensor_msgs::image_encodings::BGR8;
    }

    void run()
    {
        ros::spin();
    }

private:
    void inputImageCallback(const opentera_webrtc_ros_msgs::PeerImageConstPtr& msg)
    {
        if (!m_enabled)
        {
            m_outputImagePublisher.publish(msg);
            return;
        }

        if (m_faceCroppersByPeerId.find(msg->sender.id) == m_faceCroppersByPeerId.end())
        {
            m_faceCroppersByPeerId.insert({msg->sender.id, FaceCropper(m_faceDetector,
                m_configuration.minFaceWidth,
                m_configuration.minFaceHeight,
                m_configuration.outputWidth,
                m_configuration.outputHeight)});
        }

        cv_bridge::CvImagePtr cvPtr = cv_bridge::toCvCopy(msg->frame, sensor_msgs::image_encodings::BGR8);

        m_outputImage.header = msg->frame.header;
        m_faceCroppersByPeerId.at(msg->sender.id).crop(cvPtr->image, m_outputImage.image);
        if (m_configuration.adjustBrightness)
        {
            adjustBrightness(m_outputImage.image);
        }

        opentera_webrtc_ros_msgs::PeerImage outputMsg;
        outputMsg.sender = msg->sender;
        outputMsg.frame = *m_outputImage.toImageMsg();
        m_outputImagePublisher.publish(outputMsg);
    }

    void peerStatusCallback(const opentera_webrtc_ros_msgs::PeerStatusConstPtr& msg)
    {
        if (msg->status == opentera_webrtc_ros_msgs::PeerStatus::STATUS_REMOTE_STREAM_REMOVED)
        {
            m_faceCroppersByPeerId.erase(msg->sender.id);
        }
    }

    void enableFaceCroppingCallback(const std_msgs::Bool& msg)
    {
        if (m_enabled != msg.data)
        {
            m_faceCroppersByPeerId.clear();
        }
        m_enabled = msg.data;
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "face_cropping_webrtc_node");

    ros::NodeHandle privateNodeHandle("~");

    auto configuration = FaceCroppingNodeConfiguration::fromRosParameters(privateNodeHandle);
    if (configuration == std::nullopt)
    {
        ROS_ERROR("Configuration creation failed");
        return -1;
    }

    try
    {
        FaceCroppingWebRtcNode node(*configuration);
        node.run();
    }
    catch (const std::exception& e)
    {
        ROS_ERROR("%s", e.what());
        return -1;
    }

    return 0;
}
