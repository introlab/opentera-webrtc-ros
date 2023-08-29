#ifndef OPENCV_FACE_DETECTOR_H
#define OPENCV_FACE_DETECTOR_H

#include "FaceDetector.h"

#include <opencv4/opencv2/objdetect.hpp>

class OpencvFaceDetector : public FaceDetector
{
    int m_maxWidth;
    int m_maxHeight;
    cv::CascadeClassifier m_detector;

    cv::Mat m_resizedImage;
    cv::Mat m_resizedGrayImage;

protected:
    OpencvFaceDetector(int maxWidth, int maxHeight, const std::string& modelPath);

public:
    ~OpencvFaceDetector() override = default;

    std::vector<FaceDetection> detect(const cv::Mat& bgrImage) override;
};


class HaarFaceDetector : public OpencvFaceDetector
{
    constexpr static const char* MODEL_SUBPATH = "/models/haarcascade_frontalface_default.xml";

public:
    HaarFaceDetector();

    [[nodiscard]] std::type_index type() const override { return typeid(HaarFaceDetector); }
};


class LbpFaceDetector : public OpencvFaceDetector
{
    constexpr static const char* MODEL_SUBPATH = "/models/lbpcascade_frontalface_improved.xml";

public:
    LbpFaceDetector();

    [[nodiscard]] std::type_index type() const override { return typeid(LbpFaceDetector); }
};

#endif
