#include "FaceCropper.h"
#include "FaceCroppingNodeConfiguration.h"
#include "OpencvUtils.h"
#include "utils.h"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opentera_webrtc_ros_msgs/msg/peer_image.hpp>
#include <opentera_webrtc_ros_msgs/msg/peer_status.hpp>
#include <cv_bridge/cv_bridge.h>

#include <unordered_map>


class FaceCroppingWebRtcNode : public rclcpp::Node
{
    FaceCroppingNodeConfiguration m_configuration;

    bool m_enabled;

    std::shared_ptr<FaceDetector> m_faceDetector;
    std::unordered_map<std::string, FaceCropper> m_faceCroppersByPeerId;
    cv_bridge::CvImage m_outputImage;

    rclcpp::Publisher<opentera_webrtc_ros_msgs::msg::PeerImage>::SharedPtr m_outputImagePublisher;

    rclcpp::Subscription<opentera_webrtc_ros_msgs::msg::PeerImage>::SharedPtr m_inputImageSubscriber;
    rclcpp::Subscription<opentera_webrtc_ros_msgs::msg::PeerStatus>::SharedPtr m_peerStatusSubscriber;

    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr m_enableCroppingSubscriber;

public:
    FaceCroppingWebRtcNode(FaceCroppingNodeConfiguration configuration)
        : rclcpp::Node("face_cropping_webrtc_node"),
          m_configuration(std::move(configuration)),
          m_enabled(true),
          m_faceDetector(
              createFaceDetector(*this, m_configuration.faceDetectionModel, m_configuration.useGpuIfAvailable)),
          m_outputImagePublisher(this->create_publisher<opentera_webrtc_ros_msgs::msg::PeerImage>("output_image", 10)),
          m_inputImageSubscriber(this->create_subscription<opentera_webrtc_ros_msgs::msg::PeerImage>(
              "input_image",
              10,
              bind_this<opentera_webrtc_ros_msgs::msg::PeerImage>(this, &FaceCroppingWebRtcNode::inputImageCallback))),
          m_peerStatusSubscriber(this->create_subscription<opentera_webrtc_ros_msgs::msg::PeerStatus>(
              "webrtc_peer_status",
              10,
              bind_this<opentera_webrtc_ros_msgs::msg::PeerStatus>(this, &FaceCroppingWebRtcNode::peerStatusCallback))),
          m_enableCroppingSubscriber(this->create_subscription<std_msgs::msg::Bool>(
              "enable_face_cropping",
              10,
              bind_this<std_msgs::msg::Bool>(this, &FaceCroppingWebRtcNode::enableFaceCroppingCallback)))
    {
        m_outputImage.encoding = sensor_msgs::image_encodings::BGR8;
    }

    void run() { rclcpp::spin(this->shared_from_this()); }

private:
    void inputImageCallback(const opentera_webrtc_ros_msgs::msg::PeerImage::ConstSharedPtr& msg)
    {
        if (!m_enabled)
        {
            m_outputImagePublisher->publish(*msg);
            return;
        }

        if (m_faceCroppersByPeerId.find(msg->sender.id) == m_faceCroppersByPeerId.end())
        {
            m_faceCroppersByPeerId.insert(
                {msg->sender.id,
                 FaceCropper(
                     m_faceDetector,
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

        opentera_webrtc_ros_msgs::msg::PeerImage outputMsg;
        outputMsg.sender = msg->sender;
        outputMsg.frame = *m_outputImage.toImageMsg();
        m_outputImagePublisher->publish(outputMsg);
    }

    void peerStatusCallback(const opentera_webrtc_ros_msgs::msg::PeerStatus::ConstSharedPtr& msg)
    {
        if (msg->status == opentera_webrtc_ros_msgs::msg::PeerStatus::STATUS_REMOTE_STREAM_REMOVED)
        {
            m_faceCroppersByPeerId.erase(msg->sender.id);
        }
    }

    void enableFaceCroppingCallback(const std_msgs::msg::Bool::ConstSharedPtr& msg)
    {
        if (m_enabled != msg->data)
        {
            m_faceCroppersByPeerId.clear();
        }
        m_enabled = msg->data;
    }
};


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    auto nodeHandle = std::make_shared<rclcpp::Node>("__face_cropping_webrtc_node_configuration");

    auto configuration = FaceCroppingNodeConfiguration::fromRosParameters(*nodeHandle);
    if (configuration == std::nullopt)
    {
        RCLCPP_ERROR(nodeHandle->get_logger(), "Configuration creation failed");
        return -1;
    }

    nodeHandle.reset();

    auto node = std::make_shared<FaceCroppingWebRtcNode>(*configuration);

    try
    {
        node->run();
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR(node->get_logger(), "%s", e.what());
        return -1;
    }

    return 0;
}
