#include "OpencvFaceDetector.h"
#include "TorchFaceDetector.h"
#include "FaceCropper.h"
#include "FaceCroppingNodeConfiguration.h"
#include "OpencvUtils.h"
#include "utils.h"


#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>

class FaceCroppingNode
{
    std::shared_ptr<rclcpp::Node> m_node;

    image_transport::ImageTransport m_imageTransport;
    image_transport::Subscriber m_inputImageSubscriber;
    image_transport::Publisher m_outputImagePublisher;

    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr m_enableCroppingSubscriber;
    bool m_enabled;

    FaceCropper m_faceCropper;
    cv_bridge::CvImage m_outputImage;

    bool m_adjustBrightness;

public:
    FaceCroppingNode(const FaceCroppingNodeConfiguration& configuration)
        : m_node{std::make_shared<rclcpp::Node>("face_cropping_node")},
          m_imageTransport{m_node},
          m_inputImageSubscriber{
              m_imageTransport.subscribe("input_image", 1, &FaceCroppingNode::inputImageCallback, this)},
          m_outputImagePublisher{m_imageTransport.advertise("output_image", 1)},
          m_enableCroppingSubscriber{m_node->create_subscription<std_msgs::msg::Bool>(
              "enable_face_cropping",
              10,
              bind_this<std_msgs::msg::Bool>(this, &FaceCroppingNode::enableFaceCroppingCallback))},
          m_enabled{true},
          m_faceCropper{
              createFaceDetector(*m_node, configuration.faceDetectionModel, configuration.useGpuIfAvailable),
              configuration.minFaceWidth,
              configuration.minFaceHeight,
              configuration.outputWidth,
              configuration.outputHeight},
          m_adjustBrightness{configuration.adjustBrightness}
    {
        m_outputImage.encoding = sensor_msgs::image_encodings::BGR8;
    }

    void run() { rclcpp::spin(m_node); }

    rclcpp::Node& get_node() { return *m_node; }

private:
    void inputImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
    {
        if (!m_enabled)
        {
            m_outputImagePublisher.publish(msg);
            return;
        }

        cv_bridge::CvImagePtr cvPtr = cv_bridge::toCvCopy(*msg, sensor_msgs::image_encodings::BGR8);

        m_outputImage.header = msg->header;
        m_faceCropper.crop(cvPtr->image, m_outputImage.image);
        if (m_adjustBrightness)
        {
            adjustBrightness(m_outputImage.image);
        }

        m_outputImagePublisher.publish(m_outputImage.toImageMsg());
    }

    void enableFaceCroppingCallback(const std_msgs::msg::Bool::ConstSharedPtr& msg)
    {
        if (m_enabled != msg->data)
        {
            m_faceCropper.reset();
        }
        m_enabled = msg->data;
    }
};


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    auto nodeHandle = std::make_shared<rclcpp::Node>("__face_cropping_node_configuration");

    auto configuration = FaceCroppingNodeConfiguration::fromRosParameters(*nodeHandle);
    if (configuration == std::nullopt)
    {
        RCLCPP_ERROR(nodeHandle->get_logger(), "Configuration creation failed");
        return -1;
    }

    nodeHandle.reset();

    FaceCroppingNode node{*configuration};
    try
    {
        node.run();
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR(node.get_node().get_logger(), "%s", e.what());
        return -1;
    }

    return 0;
}
