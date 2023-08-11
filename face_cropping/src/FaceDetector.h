#ifndef FILE_DETECTOR_H
#define FILE_DETECTOR_H

#include <opencv4/opencv2/core.hpp>
#include <typeindex>

#include <ros/package.h>


inline std::string getPackagePath()
{
    return ros::package::getPath("face_cropping");
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

#endif
