#ifndef OPENCV_UTILS_H
#define OPENCV_UTILS_H

#include <opencv4/opencv2/core.hpp>

inline void adjustBrightness(cv::Mat& image)
{
    cv::Scalar outputMean = cv::mean(image);
    double colorScale = 3 * 128 / (outputMean.val[0] + outputMean.val[1] + outputMean.val[2]);
    cv::convertScaleAbs(image, image, colorScale);
}

#endif
