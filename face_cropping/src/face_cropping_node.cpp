#include "OpencvFaceDetector.h"
#include "TorchFaceDetector.h"
#include "FaceCropper.h"
#include "FaceCroppingNodeConfiguration.h"
#include "OpencvUtils.h"

#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

class FaceCroppingNode
{
    ros::NodeHandle m_nodeHandle;

    image_transport::ImageTransport m_imageTransport;
    image_transport::Subscriber m_inputImageSubscriber;
    image_transport::Publisher m_outputImagePublisher;

    ros::Subscriber m_enableCroppingSubscriber;
    bool m_enabled;

    FaceCropper m_faceCropper;
    cv_bridge::CvImage m_outputImage;

    bool m_adjustBrightness;

public:
    FaceCroppingNode(const FaceCroppingNodeConfiguration& configuration)
        : m_imageTransport(m_nodeHandle),
          m_enabled(true),
          m_faceCropper(createFaceDetector(configuration.faceDetectionModel, configuration.useGpuIfAvailable),
              configuration.minFaceWidth,
              configuration.minFaceHeight,
              configuration.outputWidth,
              configuration.outputHeight),
          m_adjustBrightness(configuration.adjustBrightness)
    {
        m_inputImageSubscriber =
            m_imageTransport.subscribe("input_image", 1, &FaceCroppingNode::inputImageCallback, this);
        m_outputImagePublisher = m_imageTransport.advertise("output_image", 1);

        m_enableCroppingSubscriber =
            m_nodeHandle.subscribe("enable_face_cropping", 10, &FaceCroppingNode::enableFaceCroppingCallback, this);

        m_outputImage.encoding = sensor_msgs::image_encodings::BGR8;
    }

    void run()
    {
        ros::spin();
    }

private:
    void inputImageCallback(const sensor_msgs::ImageConstPtr& msg)
    {
        if (!m_enabled)
        {
            m_outputImagePublisher.publish(msg);
            return;
        }

        cv_bridge::CvImagePtr cvPtr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

        m_outputImage.header = msg->header;
        m_faceCropper.crop(cvPtr->image, m_outputImage.image);
        if (m_adjustBrightness)
        {
            adjustBrightness(m_outputImage.image);
        }

        m_outputImagePublisher.publish(m_outputImage.toImageMsg());
    }

    void enableFaceCroppingCallback(const std_msgs::Bool& msg)
    {
        if (m_enabled != msg.data)
        {
            m_faceCropper.reset();
        }
        m_enabled = msg.data;
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "face_cropping_node");

    ros::NodeHandle privateNodeHandle("~");

    auto configuration = FaceCroppingNodeConfiguration::fromRosParameters(privateNodeHandle);
    if (configuration == std::nullopt)
    {
        ROS_ERROR("Configuration creation failed");
        return -1;
    }

    try
    {
        FaceCroppingNode node(*configuration);
        node.run();
    }
    catch (const std::exception& e)
    {
        ROS_ERROR("%s", e.what());
        return -1;
    }

    return 0;
}
