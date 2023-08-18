#include "OpencvFaceDetector.h"

#include <opencv4/opencv2/imgproc.hpp>

OpencvFaceDetector::OpencvFaceDetector(int maxWidth, int maxHeight, const std::string& modelPath)
    : m_maxWidth(maxWidth),
      m_maxHeight(maxHeight)
{
    m_detector.load(modelPath);
}

std::vector<FaceDetection> OpencvFaceDetector::detect(const cv::Mat& bgrImage)
{
    float scale = std::min(float(m_maxWidth) / float(bgrImage.cols), float(m_maxHeight) / float(bgrImage.rows));
    cv::resize(bgrImage, m_resizedImage, cv::Size(), scale, scale, cv::INTER_LINEAR);
    cv::cvtColor(m_resizedImage, m_resizedGrayImage, cv::COLOR_BGR2GRAY, 0);

    std::vector<cv::Rect> faceRects;

    m_detector.detectMultiScale(
        m_resizedGrayImage,
        faceRects,
        1.1,
        3,
        cv::CASCADE_SCALE_IMAGE,
        cv::Size(m_maxWidth / 20, m_maxHeight / 20));

    std::vector<FaceDetection> faceDetections;
    faceDetections.reserve(faceRects.size());

    for (cv::Rect& r : faceRects)
    {
        faceDetections.emplace_back(
            (float(r.x) + float(r.width) / 2) / scale,
            (float(r.y) + float(r.height) / 2) / scale,
            float(r.width) / scale,
            float(r.height) / scale);
    }

    return faceDetections;
}


HaarFaceDetector::HaarFaceDetector()
    : OpencvFaceDetector(
          640,
          640,
          getPackagePath() + "/models/haarcascade_frontalface_default.xml")
{
}


LbpFaceDetector::LbpFaceDetector()
    : OpencvFaceDetector(
          640,
          640,
          getPackagePath() + "/models/lbpcascade_frontalface_improved.xml")
{
}
