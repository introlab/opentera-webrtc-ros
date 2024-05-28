#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <opencv4/opencv2/core.hpp>
#include <rclcpp/node.hpp>
#include <typeindex>

#include <memory>


inline std::string getPackagePath()
{
    return ament_index_cpp::get_package_share_directory("face_cropping");
}

struct FaceDetection
{
    float xCenter;
    float yCenter;
    float width;
    float height;

    FaceDetection(float xCenter, float yCenter, float width, float height);
};

inline FaceDetection::FaceDetection(float xCenter, float yCenter, float width, float height)
    : xCenter(xCenter),
      yCenter(yCenter),
      width(width),
      height(height)
{
}

class FaceDetector
{
public:
    FaceDetector() = default;
    virtual ~FaceDetector() = default;

    virtual std::vector<FaceDetection> detect(const cv::Mat& bgrImage) = 0;
    [[nodiscard]] virtual std::type_index type() const = 0;
};

std::shared_ptr<FaceDetector>
    createFaceDetector(rclcpp::Node& nodeHandle, const std::string& name, bool useGpuIfAvailable);

#endif
