#include "OpencvFaceDetector.h"
#include "TorchFaceDetector.h"
#include "FaceCropper.h"

#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

struct FaceCroppingNodeConfiguration
{
    std::string faceDetectionModel;
    bool useGpuIfAvailable;

    float minFaceWidth;
    float minFaceHeight;
    int outputWidth;
    int outputHeight;

    bool adjustBrightness;
};

std::unique_ptr<FaceDetector> createFaceDetector(const std::string& name, bool useGpuIfAvailable)
{
    if (name == "haarcascade")
    {
        return std::make_unique<HaarFaceDetector>();
    }
    else if (name == "lbpcascade")
    {
        return std::make_unique<LbpFaceDetector>();
    }
    #ifndef NO_TORCH
    else if (name == "small_yunet_0.25_160")
    {
        return std::make_unique<SmallYunet025Silu160FaceDetector>(useGpuIfAvailable);
    }
    else if (name == "small_yunet_0.25_320")
    {
        return std::make_unique<SmallYunet025Silu320FaceDetector>(useGpuIfAvailable);
    }
    else if (name == "small_yunet_0.25_640")
    {
        return std::make_unique<SmallYunet025Silu640FaceDetector>(useGpuIfAvailable);
    }
    else if (name == "small_yunet_0.5_160")
    {
        return std::make_unique<SmallYunet05Silu160FaceDetector>(useGpuIfAvailable);
    }
    else if (name == "small_yunet_0.5_320")
    {
        return std::make_unique<SmallYunet05Silu320FaceDetector>(useGpuIfAvailable);
    }
    else if (name == "small_yunet_0.5_640")
    {
        return std::make_unique<SmallYunet05Silu640FaceDetector>(useGpuIfAvailable);
    }
    #endif
    else
    {
        throw std::runtime_error("Not supported face detector (" + name + ")");
    }
}


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
            m_imageTransport.subscribe("input_image",1, &FaceCroppingNode::inputImageCallback, this);
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
            return;
        }

        cv_bridge::CvImagePtr cvPtr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

        m_outputImage.header = msg->header;
        m_faceCropper.crop(cvPtr->image, m_outputImage.image);
        if (m_adjustBrightness)
        {
            cv::Scalar outputMean = cv::mean(m_outputImage.image);
            double colorScale = 3 * 128 / (outputMean.val[0] + outputMean.val[1] + outputMean.val[2]);
            cv::convertScaleAbs(m_outputImage.image, m_outputImage.image, colorScale);
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

    FaceCroppingNodeConfiguration configuration;

    if (!privateNodeHandle.getParam("face_detection_model", configuration.faceDetectionModel))
    {
        ROS_ERROR("The parameter face_detection_model is required.");
        return -1;
    }
    privateNodeHandle.param("use_gpu_if_available", configuration.useGpuIfAvailable, false);


    if (!privateNodeHandle.getParam("min_face_width", configuration.minFaceWidth))
    {
        ROS_ERROR("The parameter min_face_width is required.");
        return -1;
    }
    if (!privateNodeHandle.getParam("min_face_height", configuration.minFaceHeight))
    {
        ROS_ERROR("The parameter min_face_height is required.");
        return -1;
    }
    if (!privateNodeHandle.getParam("output_width", configuration.outputWidth))
    {
        ROS_ERROR("The parameter output_width is required.");
        return -1;
    }
    if (!privateNodeHandle.getParam("output_height", configuration.outputHeight))
    {
        ROS_ERROR("The parameter output_height is required.");
        return -1;
    }
    privateNodeHandle.param("adjust_brightness", configuration.adjustBrightness, true);

    ROS_INFO_STREAM("Face Cropping Configuration:");
    ROS_INFO_STREAM("\tface_detection_model=" << configuration.faceDetectionModel);
    ROS_INFO_STREAM("\tuse_gpu_if_available=" << configuration.useGpuIfAvailable);
    ROS_INFO_STREAM("\tmin_face_width=" << configuration.minFaceWidth);
    ROS_INFO_STREAM("\tmin_face_height=" << configuration.minFaceHeight);
    ROS_INFO_STREAM("\toutput_width=" << configuration.outputWidth);
    ROS_INFO_STREAM("\toutput_height=" << configuration.outputHeight);
    ROS_INFO_STREAM("\tadjust_brightness: " << configuration.adjustBrightness);

    try
    {
        FaceCroppingNode node(configuration);
        node.run();
    }
    catch (const std::exception& e)
    {
        ROS_ERROR("%s", e.what());
        return -1;
    }

    return 0;
}
